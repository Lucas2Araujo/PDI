"""
batch.py — Motor de Processamento de Imagens em Lote.

Responsável por varrer um diretório de entrada ou lista de imagens (em disco ou memória),
aplicar downscaling preventivo (máx 800×800 px), converter para escala de cinza e quantizar
todas as imagens de forma assíncrona/resiliente com coleta explícita de lixo para Web/WASM.
"""

import gc
import io
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from src.core.grayscale import (
    GrayscaleMethod,
    colorize_channel,
    is_channel_isolation,
    to_grayscale,
)
from src.core.histogram import ImageMetrics, calculate_metrics
from src.core.image_io import (
    MAX_IMAGE_DIMENSION,
    MAX_THUMBNAIL_DIMENSION,
    array_to_png_bytes,
    make_thumbnail_png,
    open_and_downscale_image,
)
from src.core.quantization import (
    QuantizationTechnique,
    quantize,
    technique_label,
)

# Formatos de arquivo de imagem suportados pelo módulo
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)


# ---------------------------------------------------------------------------
# Estruturas de Dados Públicas
# ---------------------------------------------------------------------------


@dataclass
class BatchItemResult:
    """
    Resultado detalhado do processamento de uma imagem individual dentro do lote.
    """

    filename: str
    source_path: Path | None = None
    output_path: Path | None = None
    source_bytes: bytes | None = None
    quantized_bytes: bytes | None = None
    source_thumb_bytes: bytes | None = None
    quantized_thumb_bytes: bytes | None = None
    raw_array: np.ndarray | None = None
    gray_array: np.ndarray | None = None
    quantized_array: np.ndarray | None = None
    metrics: ImageMetrics | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    success: bool = True


@dataclass
class BatchResult:
    """
    Resultado global do processamento em lote com estatísticas agregadas.
    """

    total: int = 0
    processed: int = 0
    failed: list[tuple[Path | str, str]] = field(default_factory=list)
    output_dir: Path = field(default_factory=Path)
    items: list[BatchItemResult] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    @property
    def success_count(self) -> int:
        return self.processed

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    @property
    def avg_mse(self) -> float:
        valid = [it for it in self.items if it.success and it.metrics is not None]
        if not valid:
            return 0.0
        return sum(it.metrics.mse for it in valid) / len(valid)

    @property
    def avg_psnr(self) -> float:
        valid = [it for it in self.items if it.success and it.metrics is not None]
        if not valid:
            return 0.0
        return sum(it.metrics.psnr for it in valid) / len(valid)

    @property
    def avg_savings_pct(self) -> float:
        valid = [it for it in self.items if it.success and it.metrics is not None]
        if not valid:
            return 0.0
        return sum((1.0 - it.metrics.bits / 8.0) * 100.0 for it in valid) / len(valid)


ProgressCallback = Callable[[int, int, str], None]
ItemDoneCallback = Callable[[BatchItemResult], None]


# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------


def discover_images(directory: Path) -> list[Path]:
    """Retorna a lista de arquivos de imagem suportados em um diretório."""
    if not directory.is_dir():
        raise ValueError(f"Diretório de entrada inválido: {directory}")

    found = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return found


def _process_batch_item(
    source: Path | bytes,
    filename: str,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    convert_to_grayscale: bool = True,
    max_dim: int = MAX_IMAGE_DIMENSION,
    output_dir: Path | None = None,
    source_path: Path | None = None,
) -> BatchItemResult:
    """Executa o pipeline completo de quantização e métricas para um único item do lote."""
    item_start = time.perf_counter()
    try:
        raw_array = open_and_downscale_image(source, max_dim=max_dim)
        source_bytes = array_to_png_bytes(raw_array)

        if convert_to_grayscale:
            gray_array = to_grayscale(raw_array, method=grayscale_method)
            quantized_raw = quantize(gray_array, bits=bits, technique=technique)
            if is_channel_isolation(grayscale_method):
                quantized_array = colorize_channel(quantized_raw, grayscale_method)
            else:
                quantized_array = quantized_raw
            target_original = gray_array
        else:
            gray_array = None
            rgb_input = raw_array[:, :, :3] if (raw_array.ndim == 3 and raw_array.shape[2] >= 3) else raw_array
            quantized_raw = quantize(rgb_input, bits=bits, technique=technique)
            quantized_array = quantized_raw
            target_original = rgb_input

        if output_dir is not None and source_path is not None:
            out_path = _build_output_path(
                output_dir=output_dir,
                source_path=source_path,
                technique=technique,
                bits=bits,
            )
            out_pil = Image.fromarray(quantized_array)
            out_pil.save(out_path)
            out_pil.close()
        else:
            stem = Path(filename).stem
            technique_slug = (
                technique_label(technique)
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "")
            )
            color_slug = "cinza" if convert_to_grayscale else "rgb"
            out_path = Path(f"{stem}_{technique_slug}_{color_slug}_{bits}bits.png")

        quantized_bytes = array_to_png_bytes(quantized_array)
        metrics = calculate_metrics(target_original, quantized_raw, bits)
        elapsed = time.perf_counter() - item_start

        return BatchItemResult(
            filename=filename,
            source_path=source_path,
            output_path=out_path,
            source_bytes=source_bytes,
            quantized_bytes=quantized_bytes,
            source_thumb_bytes=make_thumbnail_png(raw_array, max_size=MAX_THUMBNAIL_DIMENSION),
            quantized_thumb_bytes=make_thumbnail_png(quantized_array, max_size=MAX_THUMBNAIL_DIMENSION),
            raw_array=raw_array,
            gray_array=gray_array,
            quantized_array=quantized_array,
            metrics=metrics,
            elapsed_seconds=elapsed,
            success=True,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - item_start
        return BatchItemResult(
            filename=filename,
            source_path=source_path,
            elapsed_seconds=elapsed,
            error=str(exc),
            success=False,
        )
    finally:
        gc.collect()


def process_file_list(
    images: list[Path],
    output_dir: Path | None = None,
    technique: QuantizationTechnique = QuantizationTechnique.UNIFORM,
    bits: int = 4,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    convert_to_grayscale: bool = True,
    progress_callback: ProgressCallback | None = None,
    item_callback: ItemDoneCallback | None = None,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> BatchResult:
    """Processa uma lista explícita de arquivos de imagem e salva os resultados em output_dir."""
    if output_dir is None:
        if images and images[0].parent.exists():
            output_dir = images[0].parent / "lote_resultado"
        else:
            output_dir = Path("lote_resultado")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = BatchResult(total=len(images), output_dir=output_dir)
    batch_start = time.perf_counter()

    for idx, img_path in enumerate(images, start=1):
        item_res = _process_batch_item(
            source=img_path,
            filename=img_path.name,
            technique=technique,
            bits=bits,
            grayscale_method=grayscale_method,
            convert_to_grayscale=convert_to_grayscale,
            max_dim=max_dim,
            output_dir=output_dir,
            source_path=img_path,
        )
        result.items.append(item_res)
        if item_res.success:
            result.processed += 1
        else:
            result.failed.append((img_path, item_res.error or "Erro desconhecido"))

        if item_callback is not None:
            item_callback(item_res)
        if progress_callback is not None:
            progress_callback(idx, len(images), str(img_path.name))

    result.total_elapsed_seconds = time.perf_counter() - batch_start
    return result


def process_bytes_batch(
    images: list[tuple[str, bytes]],
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    convert_to_grayscale: bool = True,
    progress_callback: ProgressCallback | None = None,
    item_callback: ItemDoneCallback | None = None,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> BatchResult:
    """Processa uma lista de imagens em memória retornando um BatchResult rico com métricas."""
    result = BatchResult(total=len(images))
    batch_start = time.perf_counter()

    for idx, (name, raw_bytes) in enumerate(images, start=1):
        item_res = _process_batch_item(
            source=raw_bytes,
            filename=name,
            technique=technique,
            bits=bits,
            grayscale_method=grayscale_method,
            convert_to_grayscale=convert_to_grayscale,
            max_dim=max_dim,
        )
        result.items.append(item_res)
        if item_res.success:
            result.processed += 1
        else:
            result.failed.append((name, item_res.error or "Erro desconhecido"))

        if item_callback is not None:
            item_callback(item_res)
        if progress_callback is not None:
            progress_callback(idx, len(images), name)

    result.total_elapsed_seconds = time.perf_counter() - batch_start
    return result


def process_bytes_list(
    images: list[tuple[str, bytes]],
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    convert_to_grayscale: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]]]:
    """Processa uma lista de imagens em memória (retrocompatibilidade de 2-tupla)."""
    batch_res = process_bytes_batch(
        images=images,
        technique=technique,
        bits=bits,
        grayscale_method=grayscale_method,
        convert_to_grayscale=convert_to_grayscale,
        progress_callback=progress_callback,
    )

    results: list[tuple[str, bytes]] = []
    for item in batch_res.items:
        if item.success and item.quantized_bytes is not None:
            results.append((str(item.output_path), item.quantized_bytes))

    failures: list[tuple[str, str]] = [
        (str(name), err) for name, err in batch_res.failed
    ]

    return results, failures


def process_batch(
    input_dir: Path,
    output_dir: Path,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    convert_to_grayscale: bool = True,
    progress_callback: ProgressCallback | None = None,
    item_callback: ItemDoneCallback | None = None,
) -> BatchResult:
    """Processa todas as imagens de `input_dir` e salva os resultados em `output_dir`."""
    images = discover_images(input_dir)
    return process_file_list(
        images=images,
        output_dir=output_dir,
        technique=technique,
        bits=bits,
        grayscale_method=grayscale_method,
        convert_to_grayscale=convert_to_grayscale,
        progress_callback=progress_callback,
        item_callback=item_callback,
    )


def process_batch_async(
    input_dir: Path,
    output_dir: Path,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    convert_to_grayscale: bool = True,
    progress_callback: ProgressCallback | None = None,
    done_callback: Callable[[BatchResult], None] | None = None,
) -> threading.Thread:
    """Executa o processamento em lote em uma thread separada (não bloqueante)."""
    def _worker() -> None:
        result = process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            technique=technique,
            bits=bits,
            grayscale_method=grayscale_method,
            convert_to_grayscale=convert_to_grayscale,
            progress_callback=progress_callback,
        )
        if done_callback is not None:
            done_callback(result)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Funções Privadas de Apoio
# ---------------------------------------------------------------------------


def _build_output_path(
    output_dir: Path,
    source_path: Path,
    technique: QuantizationTechnique,
    bits: int,
) -> Path:
    """Gera o caminho final de salvamento para a imagem quantizada."""
    technique_slug = (
        technique_label(technique)
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
    )
    suffix = f"_{technique_slug}_{bits}bits"
    return output_dir / f"{source_path.stem}{suffix}.png"


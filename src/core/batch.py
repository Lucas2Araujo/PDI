"""
batch.py — Motor de Processamento de Imagens em Lote.

Responsável por varrer um diretório de entrada ou lista de imagens (em disco ou memória),
aplicar a conversão para escala de cinza e a técnica de quantização desejada a todas as
imagens suportadas, calcular métricas didáticas individuais (MSE, PSNR, economia de memória)
e salvar os resultados em um diretório de saída ou disponibilizá-los para download.

O processamento é executado de forma resiliente, reportando o progresso via callback.
"""

import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
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


def make_thumbnail_png(array: np.ndarray, max_size: int = 180) -> bytes:
    """Gera bytes PNG compactos de miniatura para exibição ultra-rápida e leve na UI."""
    if array.ndim == 3 and array.shape[2] == 4:
        pil_img = Image.fromarray(array, mode="RGBA")
    elif array.ndim == 3 and array.shape[2] == 3:
        pil_img = Image.fromarray(array, mode="RGB")
    else:
        pil_img = Image.fromarray(array, mode="L")
    pil_img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@dataclass
class BatchItemResult:
    """
    Resultado detalhado do processamento de uma imagem individual dentro do lote.

    Attributes:
        filename: Nome do arquivo original.
        source_path: Caminho original do arquivo no disco (se aplicável).
        output_path: Caminho de destino do arquivo quantizado no disco (se aplicável).
        source_bytes: Bytes originais da imagem (PNG completo para zoom e download).
        quantized_bytes: Bytes da imagem quantizada (PNG completo para zoom e download).
        source_thumb_bytes: Bytes PNG compactos da miniatura original para a galeria.
        quantized_thumb_bytes: Bytes PNG compactos da miniatura quantizada para a galeria.
        raw_array: Matriz NumPy uint8 da imagem de entrada (RGB ou Cinza).
        gray_array: Matriz NumPy uint8 após conversão em escala de cinza/canal.
        quantized_array: Matriz NumPy uint8 após o processo de quantização.
        metrics: Métricas de fidelidade e compressão (MSE, PSNR, economia).
        elapsed_seconds: Tempo de execução desta imagem em segundos.
        error: Mensagem de erro caso o processamento falhe.
        success: Indicador booleano de sucesso.
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

    Attributes:
        total: Número total de imagens encontradas na entrada.
        processed: Número de imagens processadas com sucesso.
        failed: Lista de tuplas (caminho_do_arquivo_ou_nome, mensagem_de_erro).
        output_dir: Diretório onde os resultados foram salvos (se aplicável).
        items: Lista detalhada de BatchItemResult para cada imagem do lote.
        total_elapsed_seconds: Duração total do processamento em segundos.
    """

    total: int = 0
    processed: int = 0
    failed: list[tuple[Path | str, str]] = field(default_factory=list)
    output_dir: Path = field(default_factory=Path)
    items: list[BatchItemResult] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    @property
    def success_count(self) -> int:
        """Número de imagens processadas com sucesso."""
        return self.processed

    @property
    def failure_count(self) -> int:
        """Número de imagens que falharam no processamento."""
        return len(self.failed)

    @property
    def avg_mse(self) -> float:
        """Média do Erro Quadrático Médio (MSE) das imagens bem-sucedidas."""
        valid = [it for it in self.items if it.success and it.metrics is not None]
        if not valid:
            return 0.0
        return sum(it.metrics.mse for it in valid) / len(valid)

    @property
    def avg_psnr(self) -> float:
        """Média da Relação Sinal-Ruído de Pico (PSNR) das imagens bem-sucedidas."""
        valid = [it for it in self.items if it.success and it.metrics is not None]
        if not valid:
            return 0.0
        return sum(it.metrics.psnr for it in valid) / len(valid)

    @property
    def avg_savings_pct(self) -> float:
        """Média percentual de economia de memória teórica."""
        valid = [it for it in self.items if it.success and it.metrics is not None]
        if not valid:
            return 0.0
        return sum((1.0 - it.metrics.bits / 8.0) * 100.0 for it in valid) / len(valid)


# Tipo do callback de progresso: recebe (imagens_processadas, total, caminho_ou_nome_atual)
ProgressCallback = Callable[[int, int, str], None]
# Callback chamado imediatamente após o término individual de cada item
ItemDoneCallback = Callable[[BatchItemResult], None]


# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------


def discover_images(directory: Path) -> list[Path]:
    """
    Retorna a lista de arquivos de imagem suportados em um diretório.

    Busca apenas no nível raiz do diretório (sem recursão).

    Args:
        directory: Caminho para o diretório de entrada.

    Returns:
        Lista de objetos Path para os arquivos de imagem encontrados,
        ordenados pelo nome do arquivo.

    Raises:
        ValueError: Se `directory` não existir ou não for um diretório.
    """
    if not directory.is_dir():
        raise ValueError(f"Diretório de entrada inválido: {directory}")

    found = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return found


def process_file_list(
    images: list[Path],
    output_dir: Path | None = None,
    technique: QuantizationTechnique = QuantizationTechnique.UNIFORM,
    bits: int = 4,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    progress_callback: ProgressCallback | None = None,
    item_callback: ItemDoneCallback | None = None,
) -> BatchResult:
    """
    Processa uma lista explícita de arquivos de imagem e salva os resultados em output_dir.

    Args:
        images: Lista de Paths de arquivos de imagem.
        output_dir: Diretório de destino onde os arquivos quantizados serão salvos.
        technique: Técnica de quantização (Uniforme, K-Means, etc.).
        bits: Profundidade de bits (1 a 8).
        grayscale_method: Método de conversão para escala de cinza/canal.
        progress_callback: Callback opcional de progresso (processadas, total, nome_arquivo).
        item_callback: Callback opcional chamado logo que cada item finaliza.

    Returns:
        BatchResult com as estatísticas e itens detalhados do processamento.
    """
    if output_dir is None:
        if images and images[0].parent.exists():
            output_dir = images[0].parent / "lote_resultado"
        else:
            output_dir = Path("lote_resultado")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = BatchResult(total=len(images), output_dir=output_dir)
    batch_start = time.perf_counter()

    for idx, img_path in enumerate(images, start=1):
        item_start = time.perf_counter()
        try:
            with Image.open(img_path) as pil_img:
                if pil_img.mode in ("RGBA", "LA", "P"):
                    pil_img = pil_img.convert("RGB")
                raw_array = np.array(pil_img)

            # Gera bytes da imagem original
            src_buf = io.BytesIO()
            Image.fromarray(raw_array).save(src_buf, format="PNG")
            source_bytes = src_buf.getvalue()

            gray_array = to_grayscale(raw_array, method=grayscale_method)
            quantized_raw = quantize(gray_array, bits=bits, technique=technique)

            if is_channel_isolation(grayscale_method):
                quantized_array = colorize_channel(quantized_raw, grayscale_method)
            else:
                quantized_array = quantized_raw

            out_path = _build_output_path(
                output_dir=output_dir,
                source_path=img_path,
                technique=technique,
                bits=bits,
            )
            out_pil = Image.fromarray(quantized_array)
            out_pil.save(out_path)

            q_buf = io.BytesIO()
            out_pil.save(q_buf, format="PNG")
            quantized_bytes = q_buf.getvalue()

            metrics = calculate_metrics(gray_array, quantized_raw, bits)
            elapsed = time.perf_counter() - item_start

            item_result = BatchItemResult(
                filename=img_path.name,
                source_path=img_path,
                output_path=out_path,
                source_bytes=source_bytes,
                quantized_bytes=quantized_bytes,
                source_thumb_bytes=make_thumbnail_png(raw_array, max_size=180),
                quantized_thumb_bytes=make_thumbnail_png(quantized_array, max_size=180),
                raw_array=raw_array,
                gray_array=gray_array,
                quantized_array=quantized_array,
                metrics=metrics,
                elapsed_seconds=elapsed,
                success=True,
            )
            result.items.append(item_result)
            result.processed += 1

            if item_callback is not None:
                item_callback(item_result)

        except Exception as exc:
            elapsed = time.perf_counter() - item_start
            err_msg = str(exc)
            result.failed.append((img_path, err_msg))
            item_result = BatchItemResult(
                filename=img_path.name,
                source_path=img_path,
                elapsed_seconds=elapsed,
                error=err_msg,
                success=False,
            )
            result.items.append(item_result)
            if item_callback is not None:
                item_callback(item_result)
        finally:
            if progress_callback is not None:
                progress_callback(idx, len(images), str(img_path.name))

    result.total_elapsed_seconds = time.perf_counter() - batch_start
    return result


def process_bytes_batch(
    images: list[tuple[str, bytes]],
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    progress_callback: ProgressCallback | None = None,
    item_callback: ItemDoneCallback | None = None,
) -> BatchResult:
    """
    Processa uma lista de imagens em memória retornando um BatchResult rico com métricas.

    Args:
        images: Lista de tuplas (nome_arquivo, bytes_da_imagem).
        technique: Técnica de quantização.
        bits: Profundidade de bits (1 a 8).
        grayscale_method: Método de conversão para escala de cinza/canal.
        progress_callback: Callback opcional de progresso.
        item_callback: Callback opcional para cada item concluído.

    Returns:
        BatchResult com os resultados em memória.
    """
    result = BatchResult(total=len(images))
    batch_start = time.perf_counter()

    for idx, (name, raw_bytes) in enumerate(images, start=1):
        item_start = time.perf_counter()
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                if pil_img.mode in ("RGBA", "LA", "P"):
                    pil_img = pil_img.convert("RGB")
                raw_array = np.array(pil_img)

            # Normaliza bytes de origem para PNG
            src_buf = io.BytesIO()
            Image.fromarray(raw_array).save(src_buf, format="PNG")
            source_bytes = src_buf.getvalue()

            gray_array = to_grayscale(raw_array, method=grayscale_method)
            quantized_raw = quantize(gray_array, bits=bits, technique=technique)

            if is_channel_isolation(grayscale_method):
                quantized_array = colorize_channel(quantized_raw, grayscale_method)
            else:
                quantized_array = quantized_raw

            q_buf = io.BytesIO()
            Image.fromarray(quantized_array).save(q_buf, format="PNG")
            quantized_bytes = q_buf.getvalue()

            technique_slug = (
                technique_label(technique)
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "")
            )
            stem = Path(name).stem
            out_name = f"{stem}_{technique_slug}_{bits}bits.png"

            metrics = calculate_metrics(gray_array, quantized_raw, bits)
            elapsed = time.perf_counter() - item_start

            item_result = BatchItemResult(
                filename=name,
                source_bytes=source_bytes,
                quantized_bytes=quantized_bytes,
                source_thumb_bytes=make_thumbnail_png(raw_array, max_size=180),
                quantized_thumb_bytes=make_thumbnail_png(quantized_array, max_size=180),
                raw_array=raw_array,
                gray_array=gray_array,
                quantized_array=quantized_array,
                metrics=metrics,
                elapsed_seconds=elapsed,
                output_path=Path(out_name),
                success=True,
            )
            result.items.append(item_result)
            result.processed += 1

            if item_callback is not None:
                item_callback(item_result)

        except Exception as exc:
            elapsed = time.perf_counter() - item_start
            err_msg = str(exc)
            result.failed.append((name, err_msg))
            item_result = BatchItemResult(
                filename=name,
                elapsed_seconds=elapsed,
                error=err_msg,
                success=False,
            )
            result.items.append(item_result)
            if item_callback is not None:
                item_callback(item_result)
        finally:
            if progress_callback is not None:
                progress_callback(idx, len(images), name)

    result.total_elapsed_seconds = time.perf_counter() - batch_start
    return result


def process_bytes_list(
    images: list[tuple[str, bytes]],
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]]]:
    """
    Processa uma lista de imagens em memória (retrocompatibilidade de 2-tupla).

    Returns:
        Tupla (resultados, falhas):
        - resultados: list[(nome_saída_png, bytes_png)]
        - falhas: list[(nome_original, mensagem_de_erro)]
    """
    batch_res = process_bytes_batch(
        images=images,
        technique=technique,
        bits=bits,
        grayscale_method=grayscale_method,
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
    progress_callback: ProgressCallback | None = None,
    item_callback: ItemDoneCallback | None = None,
) -> BatchResult:
    """
    Processa todas as imagens de `input_dir` e salva os resultados em `output_dir`.
    """
    images = discover_images(input_dir)
    return process_file_list(
        images=images,
        output_dir=output_dir,
        technique=technique,
        bits=bits,
        grayscale_method=grayscale_method,
        progress_callback=progress_callback,
        item_callback=item_callback,
    )


def process_batch_async(
    input_dir: Path,
    output_dir: Path,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    progress_callback: ProgressCallback | None = None,
    done_callback: Callable[[BatchResult], None] | None = None,
) -> threading.Thread:
    """
    Executa o processamento em lote em uma thread separada (não bloqueante).
    """
    def _worker() -> None:
        result = process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            technique=technique,
            bits=bits,
            grayscale_method=grayscale_method,
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


def _process_single_image(
    image_path: Path,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod,
) -> np.ndarray:
    """
    Carrega, converte para cinza/canal e quantiza uma única imagem.
    """
    with Image.open(image_path) as pil_img:
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
        image_array = np.array(pil_img)
    gray = to_grayscale(image_array, method=grayscale_method)
    quantized = quantize(gray, bits=bits, technique=technique)
    if is_channel_isolation(grayscale_method):
        return colorize_channel(quantized, grayscale_method)
    return quantized


def _build_output_path(
    output_dir: Path,
    source_path: Path,
    technique: QuantizationTechnique,
    bits: int,
) -> Path:
    """
    Constrói o caminho do arquivo de saída com sufixo padronizado.
    """
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

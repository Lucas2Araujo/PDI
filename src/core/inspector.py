"""
inspector.py — Módulo Didático de Inspeção do Pipeline de PDI ("Entranhas do Processo").

Extrai telemetria detalhada de cada etapa do processamento:
  1. Matriz de entrada original e amostras numéricas.
  2. Aritmética passo a passo da conversão para escala de cinza/canal.
  3. Mecânica interna da quantização (tabela de decisão, centróides e quantis).
  4. Auditoria de erro com matriz residual, métricas e mapa térmico (Heatmap) gerado em puro NumPy.
"""

from dataclasses import dataclass, field
import gc
import io
from typing import Any

import numpy as np
from PIL import Image

from src.core.grayscale import (
    GrayscaleMethod,
    get_channel_index,
    is_channel_isolation,
    method_label,
)
from src.core.quantization import (
    QuantizationTechnique,
    technique_label,
)


@dataclass
class MatrixSample:
    """Amostra de uma região central da imagem com coordenadas e valores numéricos."""
    start_row: int
    start_col: int
    rows: int
    cols: int
    values: np.ndarray  # (rows, cols) ou (rows, cols, channels)


@dataclass
class QuantizationStepInfo:
    """Informações detalhadas sobre as faixas ou centróides de quantização."""
    technique_name: str
    bits: int
    n_levels: int
    step_size: float | None = None
    table_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineTelemetry:
    """Conjunto completo de dados didáticos extraídos do pipeline para a imagem processada."""
    # Informações Gerais
    image_shape: tuple[int, ...]
    is_color: bool
    grayscale_method_name: str
    quantization_technique_name: str
    bits: int
    n_levels: int

    # Amostragem Matricial (Região Central 5x5)
    sample_coords: tuple[int, int]  # (start_row, start_col)
    sample_raw: np.ndarray          # 5x5x3 ou 5x5
    sample_gray: np.ndarray         # 5x5
    sample_quantized: np.ndarray    # 5x5
    sample_error: np.ndarray        # 5x5 (|gray - quant|)

    # Cálculos detalhados da conversão cinza para a amostra
    pixel_calculations: list[str]

    # Mecânica da Quantização
    quant_info: QuantizationStepInfo

    # Métricas de Qualidade e Compressão
    mse: float
    psnr: float
    max_error: int
    mean_error: float
    compression_bpp: int
    original_bpp: int
    memory_savings_pct: float

    # Mapa de Calor de Erro Residual (Bytes PNG)
    heatmap_bytes: bytes


def _format_pixel_calc(
    abs_r: int,
    abs_c: int,
    raw_pixel: np.ndarray | None,
    gray_val: int,
    is_color: bool,
    method: GrayscaleMethod,
) -> str:
    """Formata a equação passo a passo para um único pixel."""
    if not is_color or raw_pixel is None:
        return f"Pixel({abs_r},{abs_c}): Valor já monocromático = {gray_val}"

    r_val = int(raw_pixel[0])
    g_val = int(raw_pixel[1])
    b_val = int(raw_pixel[2])

    if method == GrayscaleMethod.LUMINANCE:
        y_calc = round(0.2989 * r_val + 0.5870 * g_val + 0.1140 * b_val)
        return f"Pixel({abs_r},{abs_c}): 0.2989×{r_val} + 0.5870×{g_val} + 0.1140×{b_val} = {y_calc} (Cinza: {gray_val})"
    if method == GrayscaleMethod.AVERAGE:
        y_calc = round((r_val + g_val + b_val) / 3.0)
        return f"Pixel({abs_r},{abs_c}): ({r_val} + {g_val} + {b_val}) / 3 = {y_calc} (Cinza: {gray_val})"
    if method == GrayscaleMethod.CHANNEL_R:
        return f"Pixel({abs_r},{abs_c}): Canal R isolado = {r_val}"
    if method == GrayscaleMethod.CHANNEL_G:
        return f"Pixel({abs_r},{abs_c}): Canal G isolado = {g_val}"
    if method == GrayscaleMethod.CHANNEL_B:
        return f"Pixel({abs_r},{abs_c}): Canal B isolado = {b_val}"

    return f"Pixel({abs_r},{abs_c}): Valor = {gray_val}"


def _build_pixel_calculations(
    sample_raw: np.ndarray,
    sample_gray: np.ndarray,
    start_r: int,
    start_c: int,
    is_color: bool,
    method: GrayscaleMethod,
) -> list[str]:
    """Gera a lista de strings com equações aritméticas para os pixels da amostra."""
    calcs: list[str] = []
    r_rows, r_cols = sample_gray.shape
    for i in range(r_rows):
        for j in range(r_cols):
            abs_r = start_r + i
            abs_c = start_c + j
            raw_p = sample_raw[i, j] if is_color else None
            calcs.append(
                _format_pixel_calc(
                    abs_r=abs_r,
                    abs_c=abs_c,
                    raw_pixel=raw_p,
                    gray_val=int(sample_gray[i, j]),
                    is_color=is_color,
                    method=method,
                )
            )
    return calcs


def _build_uniform_quant_table(
    gray_image: np.ndarray,
    n_levels: int,
    step_size: float,
) -> list[dict[str, Any]]:
    """Gera linhas da tabela de quantização para a técnica uniforme."""
    rows: list[dict[str, Any]] = []
    total = gray_image.size
    for idx in range(n_levels):
        low = int(idx * step_size)
        high = 255 if idx == n_levels - 1 else int((idx + 1) * step_size - 1)
        reconstruction = int(np.clip((idx + 0.5) * step_size, 0, 255))
        count = int(np.sum((gray_image >= low) & (gray_image <= high)))
        pct = (count / total) * 100.0 if total > 0 else 0.0
        rows.append({
            "index": idx,
            "range": f"[{low:03d} – {high:03d}]",
            "reconstruction": reconstruction,
            "count": count,
            "pct": f"{pct:.1f}%",
        })
    return rows


def _build_discrete_quant_table(
    q_2d: np.ndarray,
    total_pixels: int,
    label_prefix: str,
) -> list[dict[str, Any]]:
    """Gera linhas da tabela para técnicas discretas (KMeans, Dither, Histograma)."""
    rows: list[dict[str, Any]] = []
    unique_vals = np.sort(np.unique(q_2d))
    for idx, val in enumerate(unique_vals):
        count = int(np.sum(q_2d == val))
        pct = (count / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        rows.append({
            "index": idx,
            "range": f"{label_prefix} {idx + 1}",
            "reconstruction": int(val),
            "count": count,
            "pct": f"{pct:.1f}%",
        })
    return rows


def extract_pipeline_telemetry(
    raw_image: np.ndarray,
    gray_image: np.ndarray,
    quantized_image: np.ndarray,
    bits: int,
    technique: QuantizationTechnique | str,
    method: GrayscaleMethod,
) -> PipelineTelemetry:
    """
    Extrai telemetria didática completa de cada estágio do pipeline de PDI.

    Args:
        raw_image: Imagem original de entrada (H, W, 3) ou (H, W).
        gray_image: Imagem em escala de cinza intermediária (H, W).
        quantized_image: Imagem quantizada resultante (H, W) ou (H, W, 3).
        bits: Número de bits utilizado.
        technique: Técnica de quantização utilizada.
        method: Método de conversão para escala de cinza.

    Returns:
        Instância de PipelineTelemetry com todas as análises e gráficos didáticos.
    """
    h, w = gray_image.shape[:2]
    is_color = bool(raw_image.ndim == 3 and raw_image.shape[2] >= 3)
    n_levels = 2 ** bits

    # 1. Região central para amostragem matricial 5x5
    sample_size = 5
    start_r = max(0, (h // 2) - (sample_size // 2))
    start_c = max(0, (w // 2) - (sample_size // 2))
    end_r = min(h, start_r + sample_size)
    end_c = min(w, start_c + sample_size)

    sample_raw = raw_image[start_r:end_r, start_c:end_c]
    sample_gray = gray_image[start_r:end_r, start_c:end_c]

    # Assegura que imagem quantizada 2D é usada para cálculo de erro escalar ou 3D
    if gray_image.ndim == 3:
        sample_quant = quantized_image[start_r:end_r, start_c:end_c]
        diff_samp = sample_gray.astype(np.float32) - sample_quant.astype(np.float32)
        sample_error = np.clip(np.mean(np.abs(diff_samp), axis=2), 0, 255).astype(np.uint8)
        diff = gray_image.astype(np.float64) - quantized_image.astype(np.float64)
        abs_error_map = np.clip(np.mean(np.abs(diff), axis=2), 0, 255).astype(np.uint8)
    else:
        if quantized_image.ndim == 3:
            if is_channel_isolation(method):
                ch_idx = get_channel_index(method) or 0
                q_2d = quantized_image[:, :, ch_idx]
            else:
                q_2d = quantized_image[:, :, 0]
        else:
            q_2d = quantized_image

        sample_quant = q_2d[start_r:end_r, start_c:end_c]
        sample_error = np.abs(sample_gray.astype(np.int32) - sample_quant.astype(np.int32)).astype(np.uint8)
        diff = gray_image.astype(np.float64) - q_2d.astype(np.float64)
        abs_error_map = np.abs(diff).astype(np.uint8)

    # 2. Cálculos Aritméticos Passo a Passo da Conversão Cinza
    pixel_calcs = _build_pixel_calculations(
        sample_raw=sample_raw,
        sample_gray=sample_gray,
        start_r=start_r,
        start_c=start_c,
        is_color=is_color,
        method=method,
    )

    # 3. Mecânica de Quantização
    tech_name = technique_label(technique) if isinstance(technique, QuantizationTechnique) else str(technique)
    table_rows: list[dict[str, Any]] = []
    step_size: float | None = None

    if technique in (QuantizationTechnique.UNIFORM, "BOTH"):
        step_size = 256.0 / n_levels
        table_rows = _build_uniform_quant_table(gray_image if gray_image.ndim == 2 else gray_image[:, :, 0], n_levels, step_size)
    elif technique == QuantizationTechnique.FLOYD_STEINBERG:
        step_size = 255.0 / (n_levels - 1) if n_levels > 1 else 255.0
        table_rows = _build_discrete_quant_table(quantized_image if quantized_image.ndim == 2 else quantized_image[:, :, 0], gray_image.size, "Nível Dither")
    elif technique == QuantizationTechnique.KMEANS:
        table_rows = _build_discrete_quant_table(quantized_image if quantized_image.ndim == 2 else quantized_image[:, :, 0], gray_image.size, "Cluster")
    elif technique == QuantizationTechnique.HISTOGRAM:
        table_rows = _build_discrete_quant_table(quantized_image if quantized_image.ndim == 2 else quantized_image[:, :, 0], gray_image.size, "Faixa Quantil")

    quant_info = QuantizationStepInfo(
        technique_name=tech_name,
        bits=bits,
        n_levels=n_levels,
        step_size=step_size,
        table_rows=table_rows,
    )

    # 4. Métricas e Auditoria de Erro Residual
    mse = float(np.mean(diff ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 1e-10 else float("inf")
    max_error = int(np.max(abs_error_map)) if abs_error_map.size > 0 else 0
    mean_error = float(np.mean(abs_error_map)) if abs_error_map.size > 0 else 0.0

    orig_bpp = 24 if is_color else 8
    quant_bpp = bits
    savings_pct = (1.0 - (quant_bpp / orig_bpp)) * 100.0

    # 5. Geração ultra-rápida do Mapa Térmico de Erro Residual via NumPy LUT
    heatmap_bytes = _generate_heatmap_bytes(abs_error_map)

    del diff
    gc.collect()

    return PipelineTelemetry(
        image_shape=raw_image.shape,
        is_color=is_color,
        grayscale_method_name=method_label(method),
        quantization_technique_name=tech_name,
        bits=bits,
        n_levels=n_levels,
        sample_coords=(start_r, start_c),
        sample_raw=sample_raw,
        sample_gray=sample_gray,
        sample_quantized=sample_quant,
        sample_error=sample_error,
        pixel_calculations=pixel_calcs,
        quant_info=quant_info,
        mse=mse,
        psnr=psnr,
        max_error=max_error,
        mean_error=mean_error,
        compression_bpp=quant_bpp,
        original_bpp=orig_bpp,
        memory_savings_pct=savings_pct,
        heatmap_bytes=heatmap_bytes,
    )


def _generate_heatmap_bytes(error_map: np.ndarray) -> bytes:
    """
    Gera a figura PNG do mapa térmico de erro residual usando uma LUT de cores inferno
    em puro NumPy e Pillow, sem qualquer dependência ou overhead do Matplotlib.
    """
    max_val = int(np.max(error_map)) if error_map.size > 0 else 0
    if max_val == 0:
        norm = np.zeros_like(error_map, dtype=np.uint8)
    else:
        norm = ((error_map.astype(np.float32) / max_val) * 255.0).astype(np.uint8)

    # 256 RGB colors aproximando o mapa térmico inferno (escuro/roxo -> vermelho -> amarelo -> branco)
    t = np.linspace(0, 1, 256, dtype=np.float32)
    r = np.clip(np.sin(t * np.pi * 0.9) * 1.5 - 0.2 + (t > 0.6) * (t - 0.6) * 2.5, 0.0, 1.0)
    g = np.clip((t - 0.2) * 1.2 * (t > 0.2) + (t > 0.7) * (t - 0.7) * 2.0, 0.0, 1.0)
    b = np.clip(np.sin(t * np.pi) * 0.8 * (t < 0.6) + (t > 0.85) * (t - 0.85) * 4.0, 0.0, 1.0)
    lut = (np.column_stack([r, g, b]) * 255.0).astype(np.uint8)

    colored_heatmap = lut[norm]
    pil_img = Image.fromarray(colored_heatmap, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    result = buf.getvalue()
    buf.close()
    pil_img.close()
    del norm, lut, colored_heatmap
    return result

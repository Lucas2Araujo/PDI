"""
inspector.py — Módulo Didático de Inspeção do Pipeline de PDI ("Entranhas do Processo").

Extrai telemetria detalhada de cada etapa do processamento:
  1. Matriz de entrada original e amostras numéricas.
  2. Aritmética passo a passo da conversão para escala de cinza/canal.
  3. Mecânica interna da quantização (tabela de decisão, centróides e quantis).
  4. Auditoria de erro com matriz residual, métricas e mapa de calor (Heatmap).
"""

from dataclasses import dataclass, field
import io
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.core.grayscale import (
    GrayscaleMethod,
    _LUMINANCE_WEIGHTS,
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

    # Assegura que imagem quantizada 2D é usada para cálculo de erro escalar
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

    # 2. Cálculos Aritméticos Passo a Passo da Conversão Cinza
    pixel_calcs: list[str] = []
    r_rows, r_cols = sample_gray.shape
    for i in range(r_rows):
        for j in range(r_cols):
            abs_r = start_r + i
            abs_c = start_c + j
            if is_color:
                r_val = int(sample_raw[i, j, 0])
                g_val = int(sample_raw[i, j, 1])
                b_val = int(sample_raw[i, j, 2])
                if method == GrayscaleMethod.LUMINANCE:
                    y_calc = round(0.2989 * r_val + 0.5870 * g_val + 0.1140 * b_val)
                    pixel_calcs.append(
                        f"Pixel({abs_r},{abs_c}): 0.2989×{r_val} + 0.5870×{g_val} + 0.1140×{b_val} = {y_calc} (Cinza: {sample_gray[i, j]})"
                    )
                elif method == GrayscaleMethod.AVERAGE:
                    y_calc = round((r_val + g_val + b_val) / 3.0)
                    pixel_calcs.append(
                        f"Pixel({abs_r},{abs_c}): ({r_val} + {g_val} + {b_val}) / 3 = {y_calc} (Cinza: {sample_gray[i, j]})"
                    )
                elif method == GrayscaleMethod.CHANNEL_R:
                    pixel_calcs.append(f"Pixel({abs_r},{abs_c}): Canal R isolado = {r_val}")
                elif method == GrayscaleMethod.CHANNEL_G:
                    pixel_calcs.append(f"Pixel({abs_r},{abs_c}): Canal G isolado = {g_val}")
                elif method == GrayscaleMethod.CHANNEL_B:
                    pixel_calcs.append(f"Pixel({abs_r},{abs_c}): Canal B isolado = {b_val}")
            else:
                pixel_calcs.append(f"Pixel({abs_r},{abs_c}): Valor já monocromático = {sample_gray[i, j]}")

    # 3. Mecânica de Quantização
    tech_name = technique_label(technique) if isinstance(technique, QuantizationTechnique) else str(technique)
    table_rows: list[dict[str, Any]] = []
    step_size = None

    if technique == QuantizationTechnique.UNIFORM or technique == "BOTH":
        step_size = 256 // n_levels
        scale_factor = 255 // (n_levels - 1) if n_levels > 1 else 0
        for idx in range(n_levels):
            low = idx * step_size
            high = 255 if idx == n_levels - 1 else ((idx + 1) * step_size - 1)
            reconstruction = idx * scale_factor
            count = int(np.sum((gray_image >= low) & (gray_image <= high)))
            pct = (count / gray_image.size) * 100.0
            table_rows.append({
                "index": idx,
                "range": f"[{low:03d} – {high:03d}]",
                "reconstruction": reconstruction,
                "count": count,
                "pct": f"{pct:.1f}%",
            })
    elif technique == QuantizationTechnique.KMEANS:
        unique_centroids = np.sort(np.unique(q_2d))
        for idx, cent in enumerate(unique_centroids):
            count = int(np.sum(q_2d == cent))
            pct = (count / gray_image.size) * 100.0
            table_rows.append({
                "index": idx,
                "range": f"Cluster {idx + 1}",
                "reconstruction": int(cent),
                "count": count,
                "pct": f"{pct:.1f}%",
            })
    elif technique == QuantizationTechnique.HISTOGRAM:
        unique_vals = np.sort(np.unique(q_2d))
        for idx, val in enumerate(unique_vals):
            count = int(np.sum(q_2d == val))
            pct = (count / gray_image.size) * 100.0
            table_rows.append({
                "index": idx,
                "range": f"Faixa Quantil {idx + 1}",
                "reconstruction": int(val),
                "count": count,
                "pct": f"{pct:.1f}%",
            })

    quant_info = QuantizationStepInfo(
        technique_name=tech_name,
        bits=bits,
        n_levels=n_levels,
        step_size=step_size,
        table_rows=table_rows,
    )

    # 4. Métricas e Auditoria de Erro Residual
    diff = gray_image.astype(np.float64) - q_2d.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 1e-10 else float("inf")
    abs_error_map = np.abs(diff).astype(np.uint8)
    max_error = int(np.max(abs_error_map))
    mean_error = float(np.mean(abs_error_map))

    orig_bpp = 24 if is_color else 8
    quant_bpp = bits
    savings_pct = (1.0 - (quant_bpp / orig_bpp)) * 100.0

    # 5. Geração do Mapa de Calor de Erro Residual (Heatmap)
    heatmap_bytes = _generate_heatmap_figure(gray_image, q_2d, abs_error_map, bits, tech_name, mse, psnr)

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


def _generate_heatmap_figure(
    gray_image: np.ndarray,
    quantized_image: np.ndarray,
    error_map: np.ndarray,
    bits: int,
    tech_name: str,
    mse: float,
    psnr: float,
) -> bytes:
    """Gera a figura Matplotlib com a comparação da imagem e o mapa térmico de erro residual."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Auditoria Didática do Erro Residual — {tech_name} ({bits} bits)\nMSE: {mse:.2f} · PSNR: {psnr:.2f} dB",
        fontsize=13,
        fontweight="bold",
    )

    # Painel 1: Imagem Tons de Cinza Original
    axes[0].imshow(gray_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Entrada (8 bits / 256 tons)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Painel 2: Imagem Quantizada
    axes[1].imshow(quantized_image, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Quantizada ({bits} bits / {2**bits} tons)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Painel 3: Mapa de Calor do Erro Residual
    im_err = axes[2].imshow(error_map, cmap="inferno", vmin=0, vmax=max(1, int(np.max(error_map))))
    axes[2].set_title("3. Mapa de Calor do Erro |I - Q|\n(Amarelo/Branco = Maior Perda)", fontsize=11, fontweight="bold", color="#d32f2f")
    axes[2].axis("off")

    cbar = fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Erro Absoluto (Intensidade)", fontsize=9)

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


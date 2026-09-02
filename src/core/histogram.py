"""
histogram.py — Módulo de Histogramas e Métricas de Qualidade de Imagem.

Responsável por:
  - Calcular histogramas numéricos de frequência de intensidades de forma ultra-rápida via NumPy.
  - Calcular histogramas cromáticos RGB sobrepostos.
  - Calcular métricas objetivas de fidelidade: MSE e PSNR.
  - Gerar figuras legadas sob demanda (lazy loading do Matplotlib).

Referências:
    - Gonzalez & Woods, "Digital Image Processing", Cap. 3 (Histogramas) e Cap. 8.
    - Salomon, D. "Data Compression" — PSNR como métrica de qualidade de imagem.
"""

from dataclasses import dataclass
import gc
import io
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Estruturas de Dados Públicas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageMetrics:
    """
    Contém as métricas objetivas de qualidade entre imagem original e quantizada.

    Attributes:
        mse: Mean Squared Error — erro médio quadrático por pixel.
        psnr: Peak Signal-to-Noise Ratio em dB.
        unique_levels: Número de níveis de intensidade únicos na imagem quantizada.
        bits: Número de bits usado na quantização.
    """

    mse: float
    psnr: float
    unique_levels: int
    bits: int


@dataclass(frozen=True)
class HistogramData:
    """
    Dados brutos do histograma de uma imagem em escala de cinza.

    Attributes:
        counts: Array com a contagem de pixels para cada intensidade (0–255).
        bin_edges: Bordas dos 256 intervalos de intensidade.
    """

    counts: np.ndarray
    bin_edges: np.ndarray


# ---------------------------------------------------------------------------
# API Pública de Cálculo Numérico
# ---------------------------------------------------------------------------


def compute_histogram(image: np.ndarray) -> HistogramData:
    """
    Calcula o histograma numérico de frequências de intensidade de uma imagem em escala de cinza ou canal único.

    Args:
        image: Array NumPy (H, W) ou (H, W, 3) dtype uint8.

    Returns:
        HistogramData com os arrays `counts` (256 bins) e `bin_edges` (257 bordas).
    """
    if image.ndim == 3:
        # Se for um canal isolado (ex: R ativo, G=0, B=0), detecta o canal com dados
        channel_maxes = [image[:, :, c].max() for c in range(min(3, image.shape[2]))]
        best_channel = int(np.argmax(channel_maxes)) if max(channel_maxes) > 0 else 0
        arr = image[:, :, best_channel]
    else:
        arr = image

    counts, bin_edges = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    return HistogramData(counts=counts, bin_edges=bin_edges)


def compute_rgb_histogram(image: np.ndarray) -> dict[str, HistogramData]:
    """
    Calcula os histogramas individuais para cada um dos canais R, G e B de uma imagem colorida.

    Args:
        image: Array NumPy (H, W, 3) ou (H, W, 4) ou (H, W).

    Returns:
        Dicionário com as chaves "R", "G", "B" contendo suas respectivas instâncias de HistogramData.
    """
    if image.ndim == 2:
        h = compute_histogram(image)
        return {"R": h, "G": h, "B": h}

    rgb = image[:, :, :3]
    if rgb.dtype != np.uint8 and np.issubdtype(rgb.dtype, np.floating):
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    counts_r, bin_edges_r = np.histogram(rgb[:, :, 0].ravel(), bins=256, range=(0, 256))
    counts_g, bin_edges_g = np.histogram(rgb[:, :, 1].ravel(), bins=256, range=(0, 256))
    counts_b, bin_edges_b = np.histogram(rgb[:, :, 2].ravel(), bins=256, range=(0, 256))

    return {
        "R": HistogramData(counts=counts_r, bin_edges=bin_edges_r),
        "G": HistogramData(counts=counts_g, bin_edges=bin_edges_g),
        "B": HistogramData(counts=counts_b, bin_edges=bin_edges_b),
    }


def calculate_metrics(original: np.ndarray, quantized: np.ndarray, bits: int) -> ImageMetrics:
    """
    Calcula as métricas objetivas de fidelidade (MSE e PSNR) entre a imagem original e a quantizada,
    com suporte transparente para matrizes 2D (tons de cinza) e tensores 3D (RGB).

    Fórmulas:
        MSE = np.mean((orig.astype(np.float64) - quant.astype(np.float64)) ** 2)
        PSNR = 10.0 * np.log10((255.0 ** 2) / MSE) if MSE > 0 else float("inf")

    Args:
        original: Array NumPy (H, W) ou (H, W, 3) uint8.
        quantized: Array NumPy (H, W) ou (H, W, 3) uint8.
        bits: Número de bits de quantização utilizado.

    Returns:
        ImageMetrics com MSE, PSNR e contagem de níveis / cores únicas.
    """
    orig_f = original.astype(np.float64)
    quant_f = quantized.astype(np.float64)

    diff = orig_f - quant_f
    mse = float(np.mean(diff ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 0.0 else float("inf")

    if quantized.ndim == 3:
        # Em imagens RGB, conta as combinações únicas de cores [R, G, B] na paleta final
        unique_levels = int(len(np.unique(quantized.reshape(-1, 3), axis=0)))
    else:
        unique_levels = int(np.unique(quantized).size)

    del orig_f, quant_f, diff
    gc.collect()

    return ImageMetrics(mse=mse, psnr=psnr, unique_levels=unique_levels, bits=bits)


# ---------------------------------------------------------------------------
# Funções de Renderização Legadas (Matplotlib sob demanda)
# ---------------------------------------------------------------------------


def generate_comparison_figure(
    original: np.ndarray,
    quantized: np.ndarray,
    bits: int,
    technique_name: str,
    gray_method_name: str | None = None,
    hist_color: str = "#4a90d9",
    orig_hist_color: str = "#555555",
) -> bytes:
    """Gera uma figura Matplotlib comparativa para exportação ou relatórios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tons = 2 ** bits
    is_rgb = bool(original.ndim == 3 and original.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name and not is_rgb else (" | Modo RGB" if is_rgb else "")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Comparação de Quantização — {technique_name}{method_str} {info_bits}",
        fontsize=14,
        fontweight="bold",
    )

    _plot_image(axes[0, 0], original, "Original (8 bits)")
    _plot_image(axes[0, 1], quantized, f"Quantizada via {technique_name}\n{info_bits}")

    if is_rgb:
        _plot_rgb_histogram(axes[1, 0], original, "Histograma RGB — Original")
        _plot_rgb_histogram(axes[1, 1], quantized, f"Histograma RGB — {technique_name}")
    else:
        hist_original = compute_histogram(original)
        hist_quantized = compute_histogram(quantized)
        _plot_histogram(axes[1, 0], hist_original, "Histograma — Original (8 bits)", color=orig_hist_color)
        _plot_histogram(axes[1, 1], hist_quantized, f"Histograma — {technique_name}", color=hist_color)

    plt.tight_layout()
    figure_bytes = _figure_to_bytes(fig)
    plt.close(fig)
    del fig, axes
    gc.collect()
    return figure_bytes


def generate_full_comparison_figure(
    original: np.ndarray,
    uniform: np.ndarray,
    kmeans: np.ndarray,
    bits: int,
    gray_method_name: str | None = None,
    hist_color_unif: str = "#4a90d9",
    hist_color_km: str = "#e8624a",
    orig_hist_color: str = "#555555",
) -> bytes:
    """Gera a figura completa de comparação das 3 imagens (Original, Uniforme, K-Means)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tons = 2 ** bits
    is_rgb = bool(original.ndim == 3 and original.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name and not is_rgb else (" | Modo RGB" if is_rgb else "")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(
        f"Comparação Completa (Uniforme × K-Means){method_str} — {info_bits}",
        fontsize=14,
        fontweight="bold",
    )

    _plot_image(axes[0, 0], original, "Original (8 bits)")
    _plot_image(axes[0, 1], uniform, f"Quantização Uniforme {info_bits}")
    _plot_image(axes[0, 2], kmeans, f"Quantização K-Means {info_bits}")

    if is_rgb:
        _plot_rgb_histogram(axes[1, 0], original, "Histograma RGB — Original")
        _plot_rgb_histogram(axes[1, 1], uniform, "Histograma RGB — Uniforme")
        _plot_rgb_histogram(axes[1, 2], kmeans, "Histograma RGB — K-Means")
    else:
        hist_orig = compute_histogram(original)
        hist_unif = compute_histogram(uniform)
        hist_km = compute_histogram(kmeans)
        _plot_histogram(axes[1, 0], hist_orig, "Histograma — Original", color=orig_hist_color)
        _plot_histogram(axes[1, 1], hist_unif, "Histograma — Uniforme", color=hist_color_unif)
        _plot_histogram(axes[1, 2], hist_km, "Histograma — K-Means", color=hist_color_km)

    plt.tight_layout()
    figure_bytes = _figure_to_bytes(fig)
    plt.close(fig)
    del fig, axes
    gc.collect()
    return figure_bytes


def generate_color_comparison_figure(
    color_image: np.ndarray,
    quantized: np.ndarray,
    bits: int,
    technique_name: str,
    gray_image: np.ndarray | None = None,
    gray_method_name: str | None = None,
) -> bytes:
    """Gera uma figura comparativa destacando a imagem original colorida (RGB)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tons = 2 ** bits
    is_quant_rgb = bool(quantized.ndim == 3 and quantized.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_quant_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name else ""

    if gray_image is not None:
        hist_gray = compute_histogram(gray_image)
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle(
            f"Comparação Colorida vs Quantizada — {technique_name}{method_str} {info_bits}",
            fontsize=14,
            fontweight="bold",
        )

        _plot_color_image(axes[0, 0], color_image, "1. Original Colorida (RGB)")
        _plot_image(axes[0, 1], gray_image, f"2. Escala de Cinza ({gray_method_name or '8 bits'})")
        _plot_image(axes[0, 2], quantized, f"3. Quantizada via {technique_name}\n{info_bits}")

        _plot_rgb_histogram(axes[1, 0], color_image, "Histograma de Cores (RGB)")
        _plot_histogram(axes[1, 1], hist_gray, "Histograma — Cinza", color="#555555")
        if is_quant_rgb:
            _plot_rgb_histogram(axes[1, 2], quantized, f"Histograma RGB — {technique_name}")
        else:
            hist_quantized = compute_histogram(quantized)
            _plot_histogram(axes[1, 2], hist_quantized, f"Histograma — {technique_name}", color="#e8624a")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"Comparação Colorida vs Quantizada — {technique_name}{method_str} {info_bits}",
            fontsize=14,
            fontweight="bold",
        )

        _plot_color_image(axes[0, 0], color_image, "Original Colorida (RGB)")
        _plot_image(axes[0, 1], quantized, f"Quantizada via {technique_name}\n{info_bits}")

        _plot_rgb_histogram(axes[1, 0], color_image, "Histograma de Cores (RGB)")
        if is_quant_rgb:
            _plot_rgb_histogram(axes[1, 1], quantized, f"Histograma RGB — {technique_name}")
        else:
            hist_quantized = compute_histogram(quantized)
            _plot_histogram(axes[1, 1], hist_quantized, f"Histograma — {technique_name}", color="#e8624a")

    plt.tight_layout()
    figure_bytes = _figure_to_bytes(fig)
    plt.close(fig)
    del fig, axes
    gc.collect()
    return figure_bytes


def generate_dither_comparison_figure(
    original_gray: np.ndarray,
    direct_quantized: np.ndarray,
    dither_quantized: np.ndarray,
    bits: int,
    gray_method_name: str | None = None,
    mse_direct: float | None = None,
    psnr_direct: float | None = None,
    mse_dither: float | None = None,
    psnr_dither: float | None = None,
) -> bytes:
    """Gera uma figura comparativa destacando Quantização Direta vs Floyd-Steinberg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tons = 2 ** bits
    is_rgb = bool(original_gray.ndim == 3 and original_gray.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name and not is_rgb else (" | Modo RGB" if is_rgb else "")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(
        f"Comparativo de Pós-Processamento: Direta vs Floyd-Steinberg{method_str} — {info_bits}",
        fontsize=14,
        fontweight="bold",
    )

    t_dir = f"2. Quantização Direta {info_bits}"
    if mse_direct is not None and psnr_direct is not None:
        t_dir += f"\nMSE: {mse_direct:.1f} | PSNR: {psnr_direct:.1f} dB"

    t_dit = f"3. Com Floyd-Steinberg {info_bits}"
    if mse_dither is not None and psnr_dither is not None:
        t_dit += f"\nMSE: {mse_dither:.1f} | PSNR: {psnr_dither:.1f} dB"

    _plot_image(axes[0, 0], original_gray, "1. Entrada Original (8 bits)")
    _plot_image(axes[0, 1], direct_quantized, t_dir)
    _plot_image(axes[0, 2], dither_quantized, t_dit)

    if is_rgb:
        _plot_rgb_histogram(axes[1, 0], original_gray, "Histograma RGB — Entrada")
        _plot_rgb_histogram(axes[1, 1], direct_quantized, "Histograma RGB — Direta")
        _plot_rgb_histogram(axes[1, 2], dither_quantized, "Histograma RGB — Floyd-Steinberg")
    else:
        hist_orig = compute_histogram(original_gray)
        hist_dir = compute_histogram(direct_quantized)
        hist_dit = compute_histogram(dither_quantized)
        _plot_histogram(axes[1, 0], hist_orig, "Histograma — Entrada Original", color="#555555")
        _plot_histogram(axes[1, 1], hist_dir, "Histograma — Quantização Direta", color="#4a90d9")
        _plot_histogram(axes[1, 2], hist_dit, "Histograma — Com Floyd-Steinberg", color="#e8624a")

    plt.tight_layout()
    figure_bytes = _figure_to_bytes(fig)
    plt.close(fig)
    del fig, axes
    gc.collect()
    return figure_bytes



# ---------------------------------------------------------------------------
# Funções Auxiliares Privadas
# ---------------------------------------------------------------------------


def _compute_psnr(mse: float, max_value: float = 255.0) -> float:
    """Calcula o PSNR (Peak Signal-to-Noise Ratio) em dB."""
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))


def _plot_image(ax: Any, image: np.ndarray, title: str) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        img_display = image[:, :, :3] if image.shape[2] >= 3 else image
        if img_display.dtype != np.uint8 and np.issubdtype(img_display.dtype, np.floating):
            img_display = np.clip(img_display, 0.0, 1.0)
        ax.imshow(img_display)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _plot_color_image(ax: Any, image: np.ndarray, title: str) -> None:
    img_display = image[:, :, :3] if image.ndim == 3 and image.shape[2] >= 3 else image
    if img_display.dtype != np.uint8 and np.issubdtype(img_display.dtype, np.floating):
        img_display = np.clip(img_display, 0.0, 1.0)
    ax.imshow(img_display)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _plot_histogram(ax: Any, hist: HistogramData, title: str, color: str) -> None:
    ax.bar(
        hist.bin_edges[:-1],
        hist.counts,
        width=1.0,
        color=color,
        alpha=0.85,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Intensidade")
    ax.set_ylabel("Frequência (Pixels)")


def _plot_rgb_histogram(ax: Any, image: np.ndarray, title: str) -> None:
    if image.ndim == 2:
        hist = compute_histogram(image)
        _plot_histogram(ax, hist, title, color="#555555")
        return

    colors = ("#e53935", "#43a047", "#1e88e5")
    labels = ("R", "G", "B")
    img_rgb = image[:, :, :3]
    if img_rgb.dtype != np.uint8 and np.issubdtype(img_rgb.dtype, np.floating):
        img_rgb = (np.clip(img_rgb, 0.0, 1.0) * 255).astype(np.uint8)

    for i, (col, lbl) in enumerate(zip(colors, labels)):
        counts, bin_edges = np.histogram(img_rgb[:, :, i].ravel(), bins=256, range=(0, 256))
        ax.plot(bin_edges[:-1], counts, color=col, label=lbl, alpha=0.85, linewidth=1.5)

    ax.set_title(title, fontsize=11)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Intensidade (0–255)")
    ax.set_ylabel("Frequência")
    ax.legend(loc="upper right", fontsize=9)


def _figure_to_bytes(fig: Any) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()

"""
histogram.py — Módulo de Histogramas e Métricas de Qualidade de Imagem.

Responsável por:
  - Calcular histogramas de frequência de intensidades.
  - Gerar figuras comparativas (Original × Quantizada) com imagens e histogramas.
  - Calcular métricas objetivas de qualidade de quantização: MSE e PSNR.

Referências:
    - Gonzalez & Woods, "Digital Image Processing", Cap. 3 (Histogramas) e Cap. 8.
    - Salomon, D. "Data Compression" — PSNR como métrica de qualidade de imagem.
"""

import io
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Usa o backend não-interativo Agg para gerar figuras em memória (sem abrir janelas)
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Estruturas de Dados Públicas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageMetrics:
    """
    Contém as métricas objetivas de qualidade entre imagem original e quantizada.

    Attributes:
        mse: Mean Squared Error — erro médio quadrático por pixel.
             Quanto menor, mais próximas as imagens são.
        psnr: Peak Signal-to-Noise Ratio em dB.
              Quanto maior, melhor a qualidade preservada.
              Valores típicos: >40 dB (excelente), 30–40 dB (boa), <30 dB (perceptível).
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
# API Pública
# ---------------------------------------------------------------------------


def compute_histogram(image: np.ndarray) -> HistogramData:
    """
    Calcula o histograma de frequências de intensidade de uma imagem em escala de cinza ou canal único.

    Args:
        image: Array NumPy (H, W) ou (H, W, 3) dtype uint8.

    Returns:
        HistogramData com os arrays `counts` e `bin_edges`.
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


def calculate_metrics(original: np.ndarray, quantized: np.ndarray, bits: int) -> ImageMetrics:
    """
    Calcula as métricas objetivas de qualidade entre a imagem original e a quantizada.

    Args:
        original: Array NumPy (H, W) ou (H, W, 3) uint8.
        quantized: Array NumPy (H, W) ou (H, W, 3) uint8.
        bits: Número de bits de quantização utilizado.

    Returns:
        ImageMetrics com MSE, PSNR e contagem de níveis únicos.
    """
    orig_2d = original if original.ndim == 2 else original[:, :, np.argmax([original[:, :, c].max() for c in range(3)])]
    quant_2d = quantized if quantized.ndim == 2 else quantized[:, :, np.argmax([quantized[:, :, c].max() for c in range(3)])]

    orig_float = orig_2d.astype(np.float64)
    quant_float = quant_2d.astype(np.float64)

    mse = float(np.mean((orig_float - quant_float) ** 2))
    psnr = _compute_psnr(mse)
    unique_levels = int(np.unique(quant_2d).size)

    return ImageMetrics(mse=mse, psnr=psnr, unique_levels=unique_levels, bits=bits)


def generate_comparison_figure(
    original: np.ndarray,
    quantized: np.ndarray,
    bits: int,
    technique_name: str,
    hist_color: str = "#4a90d9",
    orig_hist_color: str = "#555555",
) -> bytes:
    """
    Gera uma figura Matplotlib com a comparação lado a lado:
      - Linha 1: Imagem Original | Imagem Quantizada
      - Linha 2: Histograma Original | Histograma Quantizado

    A figura é renderizada em memória e retornada como bytes PNG,
    compatível com exibição direta em interfaces Flet ou Web.

    Args:
        original: Array NumPy (H, W) ou (H, W, 3) uint8.
        quantized: Array NumPy (H, W) ou (H, W, 3) uint8.
        bits: Número de bits utilizado na quantização.
        technique_name: Nome da técnica para o título da figura.
        hist_color: Cor hexadecimal da barra do histograma quantizado.
        orig_hist_color: Cor hexadecimal da barra do histograma original.

    Returns:
        Bytes da figura no formato PNG.
    """
    n_tons = 2 ** bits
    hist_original = compute_histogram(original)
    hist_quantized = compute_histogram(quantized)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Comparação de Quantização — {technique_name} ({bits} bits / {n_tons} tons)",
        fontsize=14,
        fontweight="bold",
    )

    # Linha 1 — Imagens
    _plot_image(axes[0, 0], original, "Original (8 bits / 256 tons)")
    _plot_image(axes[0, 1], quantized, f"Quantizada ({bits} bits / {n_tons} tons)")

    # Linha 2 — Histogramas
    _plot_histogram(axes[1, 0], hist_original, "Histograma — Original", color=orig_hist_color)
    _plot_histogram(axes[1, 1], hist_quantized, f"Histograma — {technique_name}", color=hist_color)

    plt.tight_layout()

    figure_bytes = _figure_to_bytes(fig)
    plt.close(fig)

    return figure_bytes


def generate_full_comparison_figure(
    original: np.ndarray,
    uniform: np.ndarray,
    kmeans: np.ndarray,
    bits: int,
    hist_color_unif: str = "#4a90d9",
    hist_color_km: str = "#e8624a",
    orig_hist_color: str = "#555555",
) -> bytes:
    """
    Gera a figura completa de comparação das 3 imagens (Original, Uniforme, K-Means)
    com seus respectivos histogramas em uma grade 2×3.
    """
    n_tons = 2 ** bits
    hist_orig = compute_histogram(original)
    hist_unif = compute_histogram(uniform)
    hist_km = compute_histogram(kmeans)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(
        f"Comparação Completa — {bits} bits / {n_tons} tons",
        fontsize=14,
        fontweight="bold",
    )

    # Linha 1 — Imagens
    _plot_image(axes[0, 0], original, "Original (8 bits / 256 tons)")
    _plot_image(axes[0, 1], uniform, f"Quantização Uniforme ({bits} bits)")
    _plot_image(axes[0, 2], kmeans, f"Quantização K-Means ({bits} bits)")

    # Linha 2 — Histogramas
    _plot_histogram(axes[1, 0], hist_orig, "Histograma — Original", color=orig_hist_color)
    _plot_histogram(axes[1, 1], hist_unif, "Histograma — Uniforme", color=hist_color_unif)
    _plot_histogram(axes[1, 2], hist_km, "Histograma — K-Means", color=hist_color_km)

    plt.tight_layout()

    figure_bytes = _figure_to_bytes(fig)
    plt.close(fig)

    return figure_bytes



def generate_color_comparison_figure(
    color_image: np.ndarray,
    quantized: np.ndarray,
    bits: int,
    technique_name: str,
    gray_image: np.ndarray | None = None,
) -> bytes:
    """
    Gera uma figura comparativa destacando a imagem original colorida (RGB)
    e o resultado quantizado em tons de cinza.

    Se `gray_image` for fornecida, exibe uma grade 2×3 (Colorida, Cinza, Quantizada
    e seus respectivos histogramas). Caso contrário, exibe uma grade 2×2
    (Colorida vs Quantizada).

    Args:
        color_image: Array NumPy (H, W, 3/4) uint8 com imagem original colorida.
        quantized: Array NumPy (H, W) uint8 — imagem após quantização.
        bits: Número de bits utilizado na quantização.
        technique_name: Nome da técnica utilizada.
        gray_image: Opcional, imagem em escala de cinza intermediária.

    Returns:
        Bytes da figura no formato PNG.
    """
    n_tons = 2 ** bits
    hist_quantized = compute_histogram(quantized)

    if gray_image is not None:
        hist_gray = compute_histogram(gray_image)
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle(
            f"Comparação Colorida vs Quantizada — {technique_name} ({bits} bits / {n_tons} tons)",
            fontsize=14,
            fontweight="bold",
        )

        # Linha 1: Imagens
        _plot_color_image(axes[0, 0], color_image, "Original Colorida (RGB)")
        _plot_image(axes[0, 1], gray_image, "Escala de Cinza Original (8 bits)")
        _plot_image(axes[0, 2], quantized, f"Quantizada ({bits} bits / {n_tons} tons)")

        # Linha 2: Histogramas
        _plot_rgb_histogram(axes[1, 0], color_image, "Histograma de Cores (RGB)")
        _plot_histogram(axes[1, 1], hist_gray, "Histograma — Cinza", color="#555555")
        _plot_histogram(axes[1, 2], hist_quantized, f"Histograma — {technique_name}", color="#e8624a")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"Comparação Colorida vs Quantizada — {technique_name} ({bits} bits / {n_tons} tons)",
            fontsize=14,
            fontweight="bold",
        )

        _plot_color_image(axes[0, 0], color_image, "Original Colorida (RGB)")
        _plot_image(axes[0, 1], quantized, f"Quantizada ({bits} bits / {n_tons} tons)")

        _plot_rgb_histogram(axes[1, 0], color_image, "Histograma de Cores (RGB)")
        _plot_histogram(axes[1, 1], hist_quantized, f"Histograma — {technique_name}", color="#e8624a")

    plt.tight_layout()
    figure_bytes = _figure_to_bytes(fig)
    plt.close(fig)

    return figure_bytes


# ---------------------------------------------------------------------------
# Funções Privadas de Renderização
# ---------------------------------------------------------------------------


def _compute_psnr(mse: float, max_value: float = 255.0) -> float:
    """
    Calcula o PSNR (Peak Signal-to-Noise Ratio) em decibéis.

    PSNR = 20 · log10(MAX) − 10 · log10(MSE)

    Retorna `inf` quando MSE = 0 (imagens idênticas).
    """
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(max_value) - 10.0 * np.log10(mse)


def _plot_image(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    """Renderiza uma imagem em escala de cinza ou canal colorido isolado em um eixo Matplotlib."""
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        img_display = image[:, :, :3] if image.shape[2] >= 3 else image
        if img_display.dtype != np.uint8 and np.issubdtype(img_display.dtype, np.floating):
            img_display = np.clip(img_display, 0.0, 1.0)
        ax.imshow(img_display)
    ax.set_title(title, fontsize=11)
    ax.axis("off")



def _plot_color_image(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    """Renderiza uma imagem colorida (RGB/RGBA) em um eixo Matplotlib."""
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        # Se for RGBA ou tiver canais float/uint8
        img_display = image[:, :, :3] if image.shape[2] >= 3 else image
        if img_display.dtype != np.uint8 and np.issubdtype(img_display.dtype, np.floating):
            img_display = np.clip(img_display, 0.0, 1.0)
        ax.imshow(img_display)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _plot_histogram(ax: plt.Axes, hist: HistogramData, title: str, color: str) -> None:
    """Plota o histograma de frequências em um eixo Matplotlib."""
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
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))


def _plot_rgb_histogram(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    """Plota histogramas sobrepostos para os canais R, G e B."""
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
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))


def _figure_to_bytes(fig: plt.Figure) -> bytes:
    """Renderiza uma figura Matplotlib em um buffer de bytes no formato PNG."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()



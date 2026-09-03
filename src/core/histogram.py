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
from PIL import Image, ImageDraw, ImageFont


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

    @property
    def num_levels(self) -> int:
        """Alias de compatibilidade para unique_levels."""
        return self.unique_levels


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
# Funções de Renderização Analítica com Pillow Puro (Zero Dependência de Matplotlib)
# ---------------------------------------------------------------------------


def _get_font(size: int = 16, bold: bool = False) -> Any:
    """Busca e carrega fontes TrueType do sistema com suporte completo a caracteres UTF-8 e acentos."""
    candidates = [
        # Linux Adwaita / GNOME
        "AdwaitaSans-Bold.ttf" if bold else "AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Bold.ttf" if bold else "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        # Linux DejaVu / Debian / Ubuntu / Fedora
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        # Linux Carlito / Liberation
        "Carlito-Bold.ttf" if bold else "Carlito-Regular.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        # Windows / macOS
        "arialbd.ttf" if bold else "arial.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Converte string hexadecimal (#RRGGBB) para tupla RGB (r, g, b)."""
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) == 6:
        try:
            return (
                int(hex_clean[0:2], 16),
                int(hex_clean[2:4], 16),
                int(hex_clean[4:6], 16),
            )
        except ValueError:
            pass
    return (74, 144, 217)


def _draw_cell_image(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    font_title: Any,
) -> None:
    """Desenha um cartão com a imagem centralizada preservando proporções e fidelidade de pixels (NEAREST)."""
    draw.rectangle([x, y, x + w, y + h], fill=(36, 38, 46), outline=(68, 72, 88), width=2)
    draw.text((x + 18, y + 16), title, fill=(245, 245, 250), font=font_title)

    if image.ndim == 2:
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        disp = image[:, :, :3] if image.shape[2] >= 3 else image
        if disp.dtype != np.uint8 and np.issubdtype(disp.dtype, np.floating):
            disp = (np.clip(disp, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(disp)

    max_w = max(20, w - 36)
    max_h = max(20, h - 75)

    # Usa NEAREST para preservar 100% dos tons e degraus discretos da quantização sem borrar
    scale = min(max_w / pil_img.width, max_h / pil_img.height)
    new_w = max(1, int(round(pil_img.width * scale)))
    new_h = max(1, int(round(pil_img.height * scale)))
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)

    px = x + (w - pil_img.width) // 2
    py = y + 58 + (max_h - pil_img.height) // 2
    canvas.paste(pil_img, (px, py))


def _draw_cell_histogram(
    draw: ImageDraw.ImageDraw,
    counts: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    color_hex: str,
    font_title: Any,
    font_axis: Any,
) -> None:
    """Desenha o histograma de intensidades em escala de cinza com eixos e grid em alta resolução."""
    draw.rectangle([x, y, x + w, y + h], fill=(36, 38, 46), outline=(68, 72, 88), width=2)
    draw.text((x + 18, y + 16), title, fill=(245, 245, 250), font=font_title)

    ml, mr, mt, mb = 75, 24, 60, 52
    pw, ph = max(20, w - ml - mr), max(20, h - mt - mb)
    bx0, by0, bx1, by1 = x + ml, y + mt, x + w - mr, y + h - mb

    # Fundo do gráfico
    draw.rectangle([bx0, by0, bx1, by1], fill=(26, 28, 34), outline=(85, 90, 110), width=2)

    # Linhas de grade horizontais
    for s in (0.25, 0.5, 0.75):
        gy = int(by0 + ph * s)
        draw.line([(bx0, gy), (bx1, gy)], fill=(48, 52, 64), width=1)

    # Linhas e rótulos de grade verticais
    for tick in (64, 128, 192):
        gx = int(bx0 + pw * (tick / 256.0))
        draw.line([(gx, by0), (gx, by1)], fill=(48, 52, 64), width=1)
        draw.text((gx - 14, by1 + 8), str(tick), fill=(180, 180, 195), font=font_axis)
    draw.text((bx0 - 6, by1 + 8), "0", fill=(180, 180, 195), font=font_axis)
    draw.text((bx1 - 28, by1 + 8), "255", fill=(180, 180, 195), font=font_axis)
    draw.text((bx0 + pw // 2 - 40, by1 + 28), "Intensidade", fill=(160, 160, 175), font=font_axis)

    # Barras de frequência
    bar_rgb = _hex_to_rgb(color_hex)
    max_c = float(counts.max()) if counts.max() > 0 else 1.0
    for i in range(256):
        c = counts[i]
        if c <= 0:
            continue
        bar_h = int((c / max_c) * ph)
        rx0 = int(bx0 + i * (pw / 256.0))
        rx1 = max(rx0 + 1, int(bx0 + (i + 1) * (pw / 256.0)))
        ry1 = by1
        ry0 = ry1 - bar_h
        draw.rectangle([rx0, ry0, rx1, ry1], fill=bar_rgb)


def _draw_cell_rgb_histogram(
    draw: ImageDraw.ImageDraw,
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    font_title: Any,
    font_axis: Any,
) -> None:
    """Desenha histograma cromático com as curvas R, G e B sobrepostas em alta definição."""
    draw.rectangle([x, y, x + w, y + h], fill=(36, 38, 46), outline=(68, 72, 88), width=2)
    draw.text((x + 18, y + 16), title, fill=(245, 245, 250), font=font_title)

    ml, mr, mt, mb = 75, 24, 60, 52
    pw, ph = max(20, w - ml - mr), max(20, h - mt - mb)
    bx0, by0, bx1, by1 = x + ml, y + mt, x + w - mr, y + h - mb

    draw.rectangle([bx0, by0, bx1, by1], fill=(26, 28, 34), outline=(85, 90, 110), width=2)

    for s in (0.25, 0.5, 0.75):
        gy = int(by0 + ph * s)
        draw.line([(bx0, gy), (bx1, gy)], fill=(48, 52, 64), width=1)
    for tick in (64, 128, 192):
        gx = int(bx0 + pw * (tick / 256.0))
        draw.line([(gx, by0), (gx, by1)], fill=(48, 52, 64), width=1)
        draw.text((gx - 14, by1 + 8), str(tick), fill=(180, 180, 195), font=font_axis)
    draw.text((bx0 - 6, by1 + 8), "0", fill=(180, 180, 195), font=font_axis)
    draw.text((bx1 - 28, by1 + 8), "255", fill=(180, 180, 195), font=font_axis)
    draw.text((bx0 + pw // 2 - 40, by1 + 28), "Intensidade", fill=(160, 160, 175), font=font_axis)

    if image.ndim == 2:
        c_gray, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        max_c = float(c_gray.max()) if c_gray.max() > 0 else 1.0
        for i in range(256):
            if c_gray[i] > 0:
                bar_h = int((c_gray[i] / max_c) * ph)
                rx0 = int(bx0 + i * (pw / 256.0))
                rx1 = max(rx0 + 1, int(bx0 + (i + 1) * (pw / 256.0)))
                draw.rectangle([rx0, by1 - bar_h, rx1, by1], fill=(130, 130, 135))
        return

    rgb = image[:, :, :3]
    if rgb.dtype != np.uint8 and np.issubdtype(rgb.dtype, np.floating):
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    c_r, _ = np.histogram(rgb[:, :, 0].ravel(), bins=256, range=(0, 256))
    c_g, _ = np.histogram(rgb[:, :, 1].ravel(), bins=256, range=(0, 256))
    c_b, _ = np.histogram(rgb[:, :, 2].ravel(), bins=256, range=(0, 256))
    max_c = float(max(c_r.max(), c_g.max(), c_b.max(), 1))

    colors = [(235, 65, 60), (70, 175, 75), (40, 150, 245)]
    counts_list = [c_r, c_g, c_b]
    labels = ["R", "G", "B"]

    for color, cnt in zip(colors, counts_list):
        pts = []
        for i in range(256):
            px = int(bx0 + i * (pw / 255.0))
            py = int(by1 - (cnt[i] / max_c) * ph)
            pts.append((px, py))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)

    # Legenda RGB com indicador de cor e fonte TrueType
    lx = bx1 - 110
    ly = by0 + 12
    for idx, (lbl, col) in enumerate(zip(labels, colors)):
        rx = lx + idx * 36
        draw.line([(rx, ly + 8), (rx + 14, ly + 8)], fill=col, width=3)
        draw.text((rx + 18, ly), lbl, fill=(220, 220, 230), font=font_axis)


def generate_comparison_figure(
    original: np.ndarray,
    quantized: np.ndarray,
    bits: int,
    technique_name: str,
    gray_method_name: str | None = None,
    hist_color: str = "#4a90d9",
    orig_hist_color: str = "#555555",
) -> bytes:
    """Gera uma figura comparativa 2×2 em Ultra-HD via Pillow puro para exportação e relatórios."""
    n_tons = 2 ** bits
    is_rgb = bool(original.ndim == 3 and original.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name and not is_rgb else (" | Modo RGB" if is_rgb else "")

    total_w, total_h = 2200, 1450
    header_h = 70
    margin = 24
    grid_w = (total_w - margin * 3) // 2
    grid_h = (total_h - header_h - margin * 3) // 2

    canvas = Image.new("RGB", (total_w, total_h), (22, 24, 30))
    draw = ImageDraw.Draw(canvas)

    f_head = _get_font(28, bold=True)
    f_cell = _get_font(20, bold=True)
    f_axis = _get_font(15, bold=False)

    # Header
    title = f"Comparação de Quantização — {technique_name}{method_str} {info_bits}"
    draw.text((margin, 20), title, fill=(255, 255, 255), font=f_head)

    # Linha 1: Imagens
    x0 = margin
    x1 = margin * 2 + grid_w
    y_img = header_h + margin
    y_hist = header_h + margin * 2 + grid_h

    _draw_cell_image(draw, canvas, original, x0, y_img, grid_w, grid_h, "Original (8 bits)", f_cell)
    _draw_cell_image(draw, canvas, quantized, x1, y_img, grid_w, grid_h, f"Quantizada via {technique_name} {info_bits}", f_cell)

    # Linha 2: Histogramas
    if is_rgb:
        _draw_cell_rgb_histogram(draw, original, x0, y_hist, grid_w, grid_h, "Histograma RGB — Original", f_cell, f_axis)
        _draw_cell_rgb_histogram(draw, quantized, x1, y_hist, grid_w, grid_h, f"Histograma RGB — {technique_name}", f_cell, f_axis)
    else:
        h_orig = compute_histogram(original)
        h_quant = compute_histogram(quantized)
        _draw_cell_histogram(draw, h_orig.counts, x0, y_hist, grid_w, grid_h, "Histograma — Original (8 bits)", orig_hist_color, f_cell, f_axis)
        _draw_cell_histogram(draw, h_quant.counts, x1, y_hist, grid_w, grid_h, f"Histograma — {technique_name}", hist_color, f_cell, f_axis)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    del canvas, draw
    gc.collect()
    return buf.getvalue()


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
    """Gera a figura completa 2×3 em Ultra-HD de comparação das 3 imagens (Original, Uniforme, K-Means)."""
    n_tons = 2 ** bits
    is_rgb = bool(original.ndim == 3 and original.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name and not is_rgb else (" | Modo RGB" if is_rgb else "")

    total_w, total_h = 3000, 1450
    header_h = 70
    margin = 24
    grid_w = (total_w - margin * 4) // 3
    grid_h = (total_h - header_h - margin * 3) // 2

    canvas = Image.new("RGB", (total_w, total_h), (22, 24, 30))
    draw = ImageDraw.Draw(canvas)

    f_head = _get_font(28, bold=True)
    f_cell = _get_font(20, bold=True)
    f_axis = _get_font(15, bold=False)

    title = f"Comparação Completa (Uniforme × K-Means){method_str} — {info_bits}"
    draw.text((margin, 20), title, fill=(255, 255, 255), font=f_head)

    x0 = margin
    x1 = margin * 2 + grid_w
    x2 = margin * 3 + grid_w * 2
    y_img = header_h + margin
    y_hist = header_h + margin * 2 + grid_h

    # Linha 1: Imagens
    _draw_cell_image(draw, canvas, original, x0, y_img, grid_w, grid_h, "Original (8 bits)", f_cell)
    _draw_cell_image(draw, canvas, uniform, x1, y_img, grid_w, grid_h, f"Quantização Uniforme {info_bits}", f_cell)
    _draw_cell_image(draw, canvas, kmeans, x2, y_img, grid_w, grid_h, f"Quantização K-Means {info_bits}", f_cell)

    # Linha 2: Histogramas
    if is_rgb:
        _draw_cell_rgb_histogram(draw, original, x0, y_hist, grid_w, grid_h, "Histograma RGB — Original", f_cell, f_axis)
        _draw_cell_rgb_histogram(draw, uniform, x1, y_hist, grid_w, grid_h, "Histograma RGB — Uniforme", f_cell, f_axis)
        _draw_cell_rgb_histogram(draw, kmeans, x2, y_hist, grid_w, grid_h, "Histograma RGB — K-Means", f_cell, f_axis)
    else:
        h_orig = compute_histogram(original)
        h_unif = compute_histogram(uniform)
        h_km = compute_histogram(kmeans)
        _draw_cell_histogram(draw, h_orig.counts, x0, y_hist, grid_w, grid_h, "Histograma — Original", orig_hist_color, f_cell, f_axis)
        _draw_cell_histogram(draw, h_unif.counts, x1, y_hist, grid_w, grid_h, "Histograma — Uniforme", hist_color_unif, f_cell, f_axis)
        _draw_cell_histogram(draw, h_km.counts, x2, y_hist, grid_w, grid_h, "Histograma — K-Means", hist_color_km, f_cell, f_axis)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    del canvas, draw
    gc.collect()
    return buf.getvalue()


def generate_color_comparison_figure(
    color_image: np.ndarray,
    quantized: np.ndarray,
    bits: int,
    technique_name: str,
    gray_image: np.ndarray | None = None,
    gray_method_name: str | None = None,
) -> bytes:
    """Gera uma figura comparativa em Ultra-HD destacando a imagem original colorida (RGB)."""
    n_tons = 2 ** bits
    is_quant_rgb = bool(quantized.ndim == 3 and quantized.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_quant_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name else ""

    f_head = _get_font(28, bold=True)
    f_cell = _get_font(20, bold=True)
    f_axis = _get_font(15, bold=False)

    if gray_image is not None:
        total_w, total_h = 3000, 1450
        header_h = 70
        margin = 24
        grid_w = (total_w - margin * 4) // 3
        grid_h = (total_h - header_h - margin * 3) // 2

        canvas = Image.new("RGB", (total_w, total_h), (22, 24, 30))
        draw = ImageDraw.Draw(canvas)

        title = f"Comparação Colorida vs Quantizada — {technique_name}{method_str} {info_bits}"
        draw.text((margin, 20), title, fill=(255, 255, 255), font=f_head)

        x0 = margin
        x1 = margin * 2 + grid_w
        x2 = margin * 3 + grid_w * 2
        y_img = header_h + margin
        y_hist = header_h + margin * 2 + grid_h

        _draw_cell_image(draw, canvas, color_image, x0, y_img, grid_w, grid_h, "1. Original Colorida (RGB)", f_cell)
        _draw_cell_image(draw, canvas, gray_image, x1, y_img, grid_w, grid_h, f"2. Escala de Cinza ({gray_method_name or '8 bits'})", f_cell)
        _draw_cell_image(draw, canvas, quantized, x2, y_img, grid_w, grid_h, f"3. Quantizada via {technique_name} {info_bits}", f_cell)

        _draw_cell_rgb_histogram(draw, color_image, x0, y_hist, grid_w, grid_h, "Histograma de Cores (RGB)", f_cell, f_axis)
        h_gray = compute_histogram(gray_image)
        _draw_cell_histogram(draw, h_gray.counts, x1, y_hist, grid_w, grid_h, "Histograma — Cinza", "#555555", f_cell, f_axis)

        if is_quant_rgb:
            _draw_cell_rgb_histogram(draw, quantized, x2, y_hist, grid_w, grid_h, f"Histograma RGB — {technique_name}", f_cell, f_axis)
        else:
            h_q = compute_histogram(quantized)
            _draw_cell_histogram(draw, h_q.counts, x2, y_hist, grid_w, grid_h, f"Histograma — {technique_name}", "#e8624a", f_cell, f_axis)
    else:
        total_w, total_h = 2200, 1450
        header_h = 70
        margin = 24
        grid_w = (total_w - margin * 3) // 2
        grid_h = (total_h - header_h - margin * 3) // 2

        canvas = Image.new("RGB", (total_w, total_h), (22, 24, 30))
        draw = ImageDraw.Draw(canvas)

        title = f"Comparação Colorida vs Quantizada — {technique_name}{method_str} {info_bits}"
        draw.text((margin, 20), title, fill=(255, 255, 255), font=f_head)

        x0 = margin
        x1 = margin * 2 + grid_w
        y_img = header_h + margin
        y_hist = header_h + margin * 2 + grid_h

        _draw_cell_image(draw, canvas, color_image, x0, y_img, grid_w, grid_h, "Original Colorida (RGB)", f_cell)
        _draw_cell_image(draw, canvas, quantized, x1, y_img, grid_w, grid_h, f"Quantizada via {technique_name} {info_bits}", f_cell)

        _draw_cell_rgb_histogram(draw, color_image, x0, y_hist, grid_w, grid_h, "Histograma de Cores (RGB)", f_cell, f_axis)
        if is_quant_rgb:
            _draw_cell_rgb_histogram(draw, quantized, x1, y_hist, grid_w, grid_h, f"Histograma RGB — {technique_name}", f_cell, f_axis)
        else:
            h_q = compute_histogram(quantized)
            _draw_cell_histogram(draw, h_q.counts, x1, y_hist, grid_w, grid_h, f"Histograma — {technique_name}", "#e8624a", f_cell, f_axis)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    del canvas, draw
    gc.collect()
    return buf.getvalue()


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
    """Gera uma figura comparativa em Ultra-HD destacando Quantização Direta vs Floyd-Steinberg."""
    n_tons = 2 ** bits
    is_rgb = bool(original_gray.ndim == 3 and original_gray.shape[2] >= 3)
    info_bits = f"({bits} bits / {(2**bits)**3 if is_rgb else n_tons} tons)"
    method_str = f" | {gray_method_name}" if gray_method_name and not is_rgb else (" | Modo RGB" if is_rgb else "")

    total_w, total_h = 3000, 1450
    header_h = 70
    margin = 24
    grid_w = (total_w - margin * 4) // 3
    grid_h = (total_h - header_h - margin * 3) // 2

    canvas = Image.new("RGB", (total_w, total_h), (22, 24, 30))
    draw = ImageDraw.Draw(canvas)

    f_head = _get_font(28, bold=True)
    f_cell = _get_font(20, bold=True)
    f_axis = _get_font(15, bold=False)

    title = f"Comparativo: Direta vs Floyd-Steinberg{method_str} — {info_bits}"
    draw.text((margin, 20), title, fill=(255, 255, 255), font=f_head)

    t_dir = f"2. Quantização Direta {info_bits}"
    if mse_direct is not None and psnr_direct is not None:
        t_dir += f" (MSE: {mse_direct:.1f} | PSNR: {psnr_direct:.1f} dB)"

    t_dit = f"3. Floyd-Steinberg {info_bits}"
    if mse_dither is not None and psnr_dither is not None:
        t_dit += f" (MSE: {mse_dither:.1f} | PSNR: {psnr_dither:.1f} dB)"

    x0 = margin
    x1 = margin * 2 + grid_w
    x2 = margin * 3 + grid_w * 2
    y_img = header_h + margin
    y_hist = header_h + margin * 2 + grid_h

    # Linha 1: Imagens
    _draw_cell_image(draw, canvas, original_gray, x0, y_img, grid_w, grid_h, "1. Entrada Original (8 bits)", f_cell)
    _draw_cell_image(draw, canvas, direct_quantized, x1, y_img, grid_w, grid_h, t_dir, f_cell)
    _draw_cell_image(draw, canvas, dither_quantized, x2, y_img, grid_w, grid_h, t_dit, f_cell)

    # Linha 2: Histogramas
    if is_rgb:
        _draw_cell_rgb_histogram(draw, original_gray, x0, y_hist, grid_w, grid_h, "Histograma RGB — Entrada", f_cell, f_axis)
        _draw_cell_rgb_histogram(draw, direct_quantized, x1, y_hist, grid_w, grid_h, "Histograma RGB — Direta", f_cell, f_axis)
        _draw_cell_rgb_histogram(draw, dither_quantized, x2, y_hist, grid_w, grid_h, "Histograma RGB — Floyd-Steinberg", f_cell, f_axis)
    else:
        h_orig = compute_histogram(original_gray)
        h_dir = compute_histogram(direct_quantized)
        h_dit = compute_histogram(dither_quantized)
        _draw_cell_histogram(draw, h_orig.counts, x0, y_hist, grid_w, grid_h, "Histograma — Entrada Original", "#555555", f_cell, f_axis)
        _draw_cell_histogram(draw, h_dir.counts, x1, y_hist, grid_w, grid_h, "Histograma — Quantização Direta", "#4a90d9", f_cell, f_axis)
        _draw_cell_histogram(draw, h_dit.counts, x2, y_hist, grid_w, grid_h, "Histograma — Com Floyd-Steinberg", "#e8624a", f_cell, f_axis)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    del canvas, draw
    gc.collect()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Funções Auxiliares Privadas
# ---------------------------------------------------------------------------


def _compute_psnr(mse: float, max_value: float = 255.0) -> float:
    """Calcula o PSNR (Peak Signal-to-Noise Ratio) em dB."""
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))

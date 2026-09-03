"""
scripts/comparativo_graficos.py — Comparativo de Performance e Visual de Gráficos.

Compara 3 abordagens de renderização para histogramas no app PDI:
  1. Matplotlib (Atual) — gera figura estática completa em bytes PNG.
  2. Pillow ImageDraw (Alternativa Leve 1) — gera imagem PNG em bytes sem matplotlib (~40x mais rápido).
  3. Flet Canvas Nativo (Alternativa Leve 2) — renderização vetorial direto pela GPU/Flutter.

Uso:
  python scripts/comparativo_graficos.py           # Janela desktop
  python scripts/comparativo_graficos.py --web     # No navegador (modo Web)
"""

import argparse
import base64
import io
import sys
import time
from pathlib import Path

# Ajusta path para importar src
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import flet as ft
import flet.canvas as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.grayscale import to_grayscale
from src.core.image_io import open_and_downscale_image
from src.core.quantization import quantize_uniform


# ---------------------------------------------------------------------------
# 1. Implementação Matplotlib (Atual)
# ---------------------------------------------------------------------------
def render_matplotlib_histogram(counts: np.ndarray, title: str = "Histograma — Matplotlib", color: str = "#4a90d9") -> tuple[bytes, float]:
    """Gera o histograma usando matplotlib (abordagem atual do app)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=100)
    fig.patch.set_facecolor("#1e1e24")
    ax.set_facecolor("#282a36")

    ax.bar(range(256), counts, width=1.0, color=color, alpha=0.85)
    ax.set_title(title, fontsize=11, color="#f8f8f2", fontweight="bold")
    ax.set_xlim([0, 256])
    ax.set_xlabel("Intensidade (0–255)", color="#cccccc", fontsize=9)
    ax.set_ylabel("Frequência", color="#cccccc", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#44475a")
    ax.grid(True, linestyle="--", alpha=0.3, color="#6272a4")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    t1 = time.perf_counter()

    return buf.getvalue(), (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# 2. Implementação Pillow / ImageDraw (Alternativa Leve 1)
# ---------------------------------------------------------------------------
def render_pillow_histogram(
    counts: np.ndarray,
    title: str = "Histograma — Pillow (ImageDraw)",
    color: str = "#4a90d9",
    width: int = 550,
    height: int = 320,
) -> tuple[bytes, float]:
    """Gera o histograma diretamente com Pillow puro (sem Matplotlib)."""
    t0 = time.perf_counter()

    bg_color = (30, 30, 36)
    plot_bg = (40, 42, 54)
    grid_color = (60, 64, 80)
    axis_color = (90, 94, 115)
    text_color = (220, 220, 220)
    bar_rgb = (74, 144, 217) if color == "#4a90d9" else (232, 98, 74)

    im = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(im)

    margin_l, margin_r, margin_t, margin_b = 55, 20, 40, 45
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # Fundo do gráfico
    draw.rectangle([margin_l, margin_t, width - margin_r, height - margin_b], fill=plot_bg, outline=axis_color)

    # Grid vertical e horizontal
    for step in range(1, 4):
        gy = margin_t + int(plot_h * step / 4)
        draw.line([(margin_l, gy), (width - margin_r, gy)], fill=grid_color, width=1)

    for tick in [64, 128, 192]:
        gx = margin_l + int(plot_w * tick / 256)
        draw.line([(gx, margin_t), (gx, height - margin_b)], fill=grid_color, width=1)

    # Desenha as 256 barras do histograma
    max_c = float(counts.max()) if counts.max() > 0 else 1.0
    for i in range(256):
        c = counts[i]
        if c <= 0:
            continue
        bar_h = (c / max_c) * plot_h
        x0 = margin_l + int(i * plot_w / 256)
        x1 = margin_l + int((i + 1) * plot_w / 256)
        if x1 <= x0:
            x1 = x0 + 1
        y1 = height - margin_b
        y0 = int(y1 - bar_h)
        draw.rectangle([x0, y0, x1, y1], fill=bar_rgb)

    # Rótulos dos eixos e título
    draw.text((margin_l + 10, 12), title, fill=text_color)
    draw.text((margin_l, height - margin_b + 8), "0", fill=(160, 160, 160))
    draw.text((margin_l + int(plot_w * 64 / 256) - 8, height - margin_b + 8), "64", fill=(160, 160, 160))
    draw.text((margin_l + int(plot_w * 128 / 256) - 10, height - margin_b + 8), "128", fill=(160, 160, 160))
    draw.text((margin_l + int(plot_w * 192 / 256) - 10, height - margin_b + 8), "192", fill=(160, 160, 160))
    draw.text((width - margin_r - 20, height - margin_b + 8), "255", fill=(160, 160, 160))
    draw.text((margin_l + int(plot_w / 2) - 40, height - 16), "Intensidade", fill=(160, 160, 160))

    # Converte para bytes PNG
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    t1 = time.perf_counter()

    return buf.getvalue(), (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# 3. Implementação Flet Canvas Nativo (Alternativa Leve 2)
# ---------------------------------------------------------------------------
def build_flet_canvas_histogram(
    counts: np.ndarray,
    title: str = "Histograma — Flet Canvas Nativo",
    color: str = "#4a90d9",
    width: float = 550,
    height: float = 320,
) -> tuple[ft.Control, float]:
    """Gera o histograma vetorial renderizado diretamente pelo Flutter via flet.canvas."""
    t0 = time.perf_counter()

    margin_l, margin_r, margin_t, margin_b = 55.0, 20.0, 40.0, 45.0
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_c = float(counts.max()) if counts.max() > 0 else 1.0

    shapes: list[cv.Shape] = []

    # Fundo do gráfico
    shapes.append(
        cv.Rect(
            x=margin_l,
            y=margin_t,
            width=plot_w,
            height=plot_h,
            paint=ft.Paint(color="#282a36", style=ft.PaintingStyle.FILL),
        )
    )

    # Linhas de grade
    grid_paint = ft.Paint(color="#44475a", stroke_width=1, style=ft.PaintingStyle.STROKE)
    for step in range(1, 4):
        gy = margin_t + (plot_h * step / 4.0)
        shapes.append(cv.Line(x1=margin_l, y1=gy, x2=width - margin_r, y2=gy, paint=grid_paint))

    for tick in [64, 128, 192]:
        gx = margin_l + (plot_w * tick / 256.0)
        shapes.append(cv.Line(x1=gx, y1=margin_t, x2=gx, y2=height - margin_b, paint=grid_paint))

    # Polígono do Histograma usando Path vetorial contínuo
    path_elements = [cv.Path.MoveTo(margin_l, height - margin_b)]
    for i in range(256):
        c = counts[i]
        px = margin_l + (i * plot_w / 255.0)
        py = (height - margin_b) - ((c / max_c) * plot_h)
        path_elements.append(cv.Path.LineTo(px, py))
    path_elements.append(cv.Path.LineTo(width - margin_r, height - margin_b))
    path_elements.append(cv.Path.Close())

    fill_paint = ft.Paint(color=color, style=ft.PaintingStyle.FILL)
    shapes.append(cv.Path(elements=path_elements, paint=fill_paint))

    # Borda dos eixos
    border_paint = ft.Paint(color="#6272a4", stroke_width=1.5, style=ft.PaintingStyle.STROKE)
    shapes.append(
        cv.Rect(
            x=margin_l,
            y=margin_t,
            width=plot_w,
            height=plot_h,
            paint=border_paint,
        )
    )

    # Rótulos de escala
    label_style = ft.TextStyle(size=10, color="#8be9fd")
    shapes.append(cv.Text(x=margin_l, y=height - margin_b + 6, value="0", style=label_style))
    shapes.append(cv.Text(x=margin_l + (plot_w * 64 / 256.0) - 8, y=height - margin_b + 6, value="64", style=label_style))
    shapes.append(cv.Text(x=margin_l + (plot_w * 128 / 256.0) - 10, y=height - margin_b + 6, value="128", style=label_style))
    shapes.append(cv.Text(x=margin_l + (plot_w * 192 / 256.0) - 10, y=height - margin_b + 6, value="192", style=label_style))
    shapes.append(cv.Text(x=width - margin_r - 20, y=height - margin_b + 6, value="255", style=label_style))

    # Título
    title_style = ft.TextStyle(size=12, color="#f8f8f2", weight=ft.FontWeight.BOLD)
    shapes.append(cv.Text(x=margin_l + 8, y=12, value=title, style=title_style))

    canvas_widget = cv.Canvas(
        shapes=shapes,
        width=width,
        height=height,
    )
    t1 = time.perf_counter()

    return canvas_widget, (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# Aplicação Flet de Comparação
# ---------------------------------------------------------------------------
def main_app(page: ft.Page) -> None:
    page.title = "PDI — Comparativo de Métodos de Gráficos"
    page.theme_mode = ft.ThemeMode.DARK
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 24
    page.bgcolor = "#121318"

    # Carrega imagem base de teste
    sample_path = ROOT_DIR / "assets" / "sample_portrait.png"
    if sample_path.exists():
        raw_img = open_and_downscale_image(sample_path)
    else:
        # Imagem sintética gradiente
        raw_img = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))

    gray_img = to_grayscale(raw_img)

    # Estado reativo
    bits_slider = ft.Slider(min=1, max=8, divisions=7, value=3, label="{value} bits", width=220)
    bits_label = ft.Text("3 bits (8 tons)", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.AMBER_400)

    # Containers dos gráficos
    mpl_container = ft.Container()
    pil_container = ft.Container()
    canvas_container = ft.Container()

    # Cards de métricas
    mpl_metric = ft.Text(size=12)
    pil_metric = ft.Text(size=12)
    canvas_metric = ft.Text(size=12)

    def refresh_charts(_=None) -> None:
        bits = int(bits_slider.value)
        n_tons = 2 ** bits
        bits_label.value = f"{bits} bits ({n_tons} tons)"

        # Quantização uniforme
        quantized = quantize_uniform(gray_img, bits=bits)
        counts, _ = np.histogram(quantized.ravel(), bins=256, range=(0, 256))

        # 1. Matplotlib
        mpl_bytes, mpl_ms = render_matplotlib_histogram(counts, title=f"Matplotlib ({bits} bits / {n_tons} tons)")
        mpl_b64 = base64.b64encode(mpl_bytes).decode("utf-8")
        mpl_container.content = ft.Image(src=f"data:image/png;base64,{mpl_b64}", width=480, height=280, fit=ft.BoxFit.CONTAIN)
        mpl_metric.value = f"⏱️ Render: {mpl_ms:.1f} ms  |  📦 PNG: {len(mpl_bytes)/1024:.1f} KB"

        # 2. Pillow (ImageDraw)
        pil_bytes, pil_ms = render_pillow_histogram(counts, title=f"Pillow ImageDraw ({bits} bits / {n_tons} tons)", width=480, height=280)
        pil_b64 = base64.b64encode(pil_bytes).decode("utf-8")
        pil_container.content = ft.Image(src=f"data:image/png;base64,{pil_b64}", width=480, height=280, fit=ft.BoxFit.CONTAIN)
        speedup = (mpl_ms / pil_ms) if pil_ms > 0 else 1.0
        pil_metric.value = f"⏱️ Render: {pil_ms:.1f} ms ({speedup:.0f}x mais rápido!)  |  📦 PNG: {len(pil_bytes)/1024:.1f} KB"

        # 3. Flet Canvas Nativo
        canvas_widget, canvas_ms = build_flet_canvas_histogram(counts, title=f"Flet Canvas Nativo ({bits} bits / {n_tons} tons)", width=480, height=280)
        canvas_container.content = canvas_widget
        canvas_metric.value = f"⏱️ Tempo Python: {canvas_ms:.2f} ms  |  🚀 Render: GPU Flutter (0 KB de imagem!)"

        page.update()

    bits_slider.on_change = refresh_charts

    # Layout de cada card
    def build_card(title: str, subtitle: str, badge_color: str, container: ft.Container, metric: ft.Text, pros_cons: list[str]) -> ft.Container:
        return ft.Container(
            padding=16,
            border_radius=12,
            bgcolor="#1e1e24",
            border=ft.Border.all(1, "#333340"),
            content=ft.Column(
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Text(title, size=16, weight=ft.FontWeight.BOLD, color=badge_color),
                            ft.Container(
                                content=ft.Text(subtitle, size=11, weight=ft.FontWeight.W_600, color="#ffffff"),
                                bgcolor=badge_color,
                                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                border_radius=6,
                            ),
                        ],
                    ),
                    ft.Divider(height=12, color="#333340"),
                    ft.Container(
                        content=container,
                        alignment=ft.Alignment.CENTER,
                        height=290,
                        bgcolor="#18191f",
                        border_radius=8,
                        padding=4,
                    ),
                    ft.Container(
                        content=metric,
                        bgcolor="#282a36",
                        padding=8,
                        border_radius=6,
                    ),
                    ft.Column(
                        spacing=4,
                        controls=[
                            ft.Text(f"• {item}", size=12, color="#b0b0b8") for item in pros_cons
                        ],
                    ),
                ],
                spacing=10,
            ),
        )

    # Monta a tela
    card_mpl = build_card(
        title="1. Matplotlib",
        subtitle="Atual no App",
        badge_color="#bd93f9",
        container=mpl_container,
        metric=mpl_metric,
        pros_cons=[
            "Gera figura acadêmica padrão pronta para download",
            "Muito pesado no WebAssembly/Pyodide (+37 MB de download no GitHub Pages)",
            "Lento no cold start (~675 ms para importar)",
        ],
    )

    card_pil = build_card(
        title="2. Pillow (ImageDraw)",
        subtitle="Alternativa Leve 1",
        badge_color="#50fa7b",
        container=pil_container,
        metric=pil_metric,
        pros_cons=[
            "Gera bytes PNG normais (100% compatível com botão de download/ZIP)",
            "~40x mais rápido que Matplotlib (< 5 ms)",
            "Zero dependências extras (Pillow já está no app)",
            "Ideal para exportação sem inchar o site do GitHub Pages",
        ],
    )

    card_canvas = build_card(
        title="3. Flet Canvas Nativo",
        subtitle="Alternativa Leve 2",
        badge_color="#8be9fd",
        container=canvas_container,
        metric=canvas_metric,
        pros_cons=[
            "Renderizado direto pela GPU/Flutter via Canvas vetorial",
            "Zero tráfego de imagens PNG (transfere só números leves)",
            "Ideal para gráficos interativos e animações na UI",
            "Não exporta diretamente para PNG/ZIP sem código extra",
        ],
    )

    header = ft.Column(
        controls=[
            ft.Text("Comparativo de Renderização de Histogramas", size=24, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Teste prático comparando a abordagem atual (Matplotlib) com as duas alternativas mais leves (Pillow e Flet Canvas), "
                "especialmente focado na otimização de peso e velocidade para o GitHub Pages (WebAssembly).",
                size=13,
                color="#a0a0a8",
            ),
            ft.Row(
                controls=[
                    ft.Text("Nível de Bits:", size=13, weight=ft.FontWeight.BOLD),
                    bits_slider,
                    bits_label,
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ],
        spacing=8,
    )

    cards_row = ft.ResponsiveRow(
        controls=[
            ft.Container(card_mpl, col={"sm": 12, "md": 12, "lg": 4}),
            ft.Container(card_pil, col={"sm": 12, "md": 12, "lg": 4}),
            ft.Container(card_canvas, col={"sm": 12, "md": 12, "lg": 4}),
        ],
        run_spacing=16,
    )

    page.add(
        ft.Column(
            controls=[
                header,
                ft.Divider(height=20, color="#333340"),
                cards_row,
            ],
            spacing=16,
        )
    )

    # Primeira renderização
    refresh_charts()


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparativo de Métodos de Gráficos")
    parser.add_argument("--web", action="store_true", help="Executa no navegador Web")
    parser.add_argument("--port", type=int, default=8560, help="Porta para o servidor web (padrão: 8560)")
    parser.add_argument("--save", action="store_true", help="Salva imagens de exemplo em disco para inspeção")
    args = parser.parse_args()

    if args.save:
        print("Gerando amostras em disco...")
        sample_path = ROOT_DIR / "assets" / "sample_portrait.png"
        raw_img = open_and_downscale_image(sample_path) if sample_path.exists() else np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
        gray = to_grayscale(raw_img)
        quantized = quantize_uniform(gray, bits=3)
        counts, _ = np.histogram(quantized.ravel(), bins=256, range=(0, 256))

        mpl_bytes, mpl_ms = render_matplotlib_histogram(counts, title="Matplotlib (3 bits / 8 tons)")
        pil_bytes, pil_ms = render_pillow_histogram(counts, title="Pillow ImageDraw (3 bits / 8 tons)")
        canvas_widget, canvas_ms = build_flet_canvas_histogram(counts, title="Flet Canvas (3 bits / 8 tons)")

        out_mpl = ROOT_DIR / "assets" / "comparativo_demo_matplotlib.png"
        out_pil = ROOT_DIR / "assets" / "comparativo_demo_pillow.png"
        out_mpl.write_bytes(mpl_bytes)
        out_pil.write_bytes(pil_bytes)

        print(f"\n[+] Matplotlib salvo em: {out_mpl} ({len(mpl_bytes)/1024:.1f} KB | {mpl_ms:.1f} ms)")
        print(f"[+] Pillow salvo em:     {out_pil} ({len(pil_bytes)/1024:.1f} KB | {pil_ms:.1f} ms)")
        print(f"[+] Flet Canvas:         {len(canvas_widget.shapes)} shapes vetoriais | {canvas_ms:.2f} ms")
        print(f"[!] Ganho do Pillow:     {(mpl_ms/pil_ms):.0f}x mais rápido que Matplotlib!\n")
        return

    if args.web:
        print(f"Iniciando Comparativo em modo Web: http://localhost:{args.port}")
        ft.run(main=main_app, view=ft.AppView.WEB_BROWSER, port=args.port)
    else:
        print("Iniciando Comparativo em modo Desktop...")
        ft.run(main=main_app)


if __name__ == "__main__":
    main()

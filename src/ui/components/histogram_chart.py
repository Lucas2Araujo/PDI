"""
histogram_chart.py — Gráficos de Histograma Nativos do Flet (Flutter Canvas).

Renderiza histogramas de intensidade digitalmente acelerados por hardware via `flet.canvas.Canvas`,
eliminando 100% da dependência de renderização pesada do Matplotlib na interface gráfica.

Suporta:
- Histograma monocromático (escala de cinza e canais isolados) com área preenchida e borda.
- Histograma de Cores RGB sobreposto (canais R, G e B simultâneos com legenda).
- Histograma Quantizado com destaque dos níveis discretos (degraus de quantização).
- Grade de referência e eixos (0, 64, 128, 192, 255) com formatação numérica.
"""

from typing import Any
import flet as ft
import flet.canvas as cv
import numpy as np

from src.core.histogram import HistogramData, compute_histogram, compute_rgb_histogram
from src.ui import theme


class NativeHistogramChart(ft.Container):
    """
    Componente Flet para renderização nativa de histogramas via Flutter Canvas.
    """

    def __init__(
        self,
        title: str = "Histograma",
        image_or_data: np.ndarray | HistogramData | dict[str, HistogramData] | None = None,
        color: str = theme.PRIMARY_LIGHT,
        is_rgb: bool = False,
        is_quantized: bool = False,
        chart_height: int = 180,
        chart_width: int | None = None,
        on_zoom_fn: Any = None,
    ) -> None:
        self._title_str = title
        self._color = color
        self._is_rgb = is_rgb
        self._is_quantized = is_quantized
        self._chart_height = chart_height
        self._chart_width = chart_width
        self._on_zoom_fn = on_zoom_fn
        self._last_data = image_or_data

        # Controles visuais internos
        self._title_text = ft.Text(
            title,
            weight=ft.FontWeight.BOLD,
            size=theme.FONT_CAPTION,
            color=ft.Colors.ON_SURFACE,
        )
        self._peak_badge = ft.Text(
            "",
            size=11,
            color=ft.Colors.ON_SURFACE_VARIANT,
            weight=ft.FontWeight.BOLD,
        )
        self._legend_row = ft.Row(
            controls=[],
            spacing=8,
            visible=is_rgb,
        )
        self._canvas = cv.Canvas(
            shapes=[],
            height=chart_height,
            expand=True if chart_width is None else False,
            width=chart_width,
        )

        header_right_controls: list[ft.Control] = [self._legend_row, self._peak_badge]
        if on_zoom_fn:
            zoom_btn = ft.IconButton(
                icon=ft.Icons.ZOOM_IN,
                icon_size=18,
                tooltip="Ampliar Histograma",
                on_click=lambda _: on_zoom_fn(),
            )
            header_right_controls.append(zoom_btn)

        header = ft.Row(
            controls=[
                self._title_text,
                ft.Row(controls=header_right_controls, spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        content_col = ft.Column(
            controls=[
                header,
                self._canvas,
            ],
            spacing=6,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        super().__init__(
            content=content_col,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=10,
            expand=True if chart_width is None else False,
            width=chart_width,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
            on_click=(lambda _: on_zoom_fn()) if on_zoom_fn else None,
            ink=bool(on_zoom_fn),
            tooltip="Clique para ampliar o histograma" if on_zoom_fn else None,
        )

        if image_or_data is not None:
            self.set_data(image_or_data, title=title, color=color, is_rgb=is_rgb, is_quantized=is_quantized)
        else:
            self._render_empty()

    # -----------------------------------------------------------------------
    # Atualização de Dados
    # -----------------------------------------------------------------------

    def set_data(
        self,
        image_or_data: np.ndarray | HistogramData | dict[str, HistogramData],
        title: str | None = None,
        color: str | None = None,
        is_rgb: bool | None = None,
        is_quantized: bool | None = None,
    ) -> None:
        """Atualiza os dados do histograma e recalcula a renderização no Canvas."""
        self._last_data = image_or_data
        if title is not None:
            self._title_str = title
            self._title_text.value = title

        if color is not None:
            self._color = color

        if is_rgb is not None:
            self._is_rgb = is_rgb
            self._legend_row.visible = is_rgb

        if is_quantized is not None:
            self._is_quantized = is_quantized

        if self._is_rgb:
            if isinstance(image_or_data, dict):
                rgb_hists = image_or_data
            elif isinstance(image_or_data, np.ndarray):
                rgb_hists = compute_rgb_histogram(image_or_data)
            else:
                rgb_hists = {"R": image_or_data, "G": image_or_data, "B": image_or_data}
            self._render_rgb(rgb_hists)
        else:
            if isinstance(image_or_data, HistogramData):
                hist_data = image_or_data
            elif isinstance(image_or_data, np.ndarray):
                hist_data = compute_histogram(image_or_data)
            elif isinstance(image_or_data, dict):
                hist_data = next(iter(image_or_data.values()))
            else:
                hist_data = HistogramData(counts=np.zeros(256), bin_edges=np.arange(257))
            self._render_single(hist_data)

    # -----------------------------------------------------------------------
    # Renderização Interna no Canvas
    # -----------------------------------------------------------------------

    def _render_empty(self) -> None:
        """Renderiza estado vazio."""
        self._canvas.shapes = [
            cv.Text(20, self._chart_height // 2 - 10, "Sem dados de histograma", style=ft.TextStyle(size=12, color=theme.TEXT_SECONDARY))
        ]

    def _render_single(self, hist: HistogramData) -> None:
        """Renderiza o histograma monocromático no Canvas."""
        counts = hist.counts
        max_count = float(np.max(counts)) if len(counts) > 0 else 1.0
        if max_count == 0:
            max_count = 1.0

        self._peak_badge.value = f"Pico: {int(max_count):,} px"
        self._legend_row.controls.clear()
        self._legend_row.visible = False

        shapes: list[Any] = []
        pad_left = 32.0
        pad_right = 16.0
        pad_top = 10.0
        pad_bottom = 22.0

        c_w = float(self._chart_width or 360.0)
        c_h = float(self._chart_height)

        plot_w = max(10.0, c_w - pad_left - pad_right)
        plot_h = max(10.0, c_h - pad_top - pad_bottom)

        # 1. Linhas de Grade de Fundo e Eixos
        self._draw_grid_and_axes(shapes, pad_left, pad_top, plot_w, plot_h)

        # 2. Desenho do Histograma
        n_bins = len(counts)
        bar_w = plot_w / float(n_bins)

        if self._is_quantized:
            # Para quantizado: desenha barras verticais destacadas apenas onde há contagem
            spike_paint = ft.Paint(stroke_width=max(2.0, bar_w * 1.5), color=self._color, style=ft.PaintingStyle.STROKE)
            for i in range(n_bins):
                val = counts[i]
                if val > 0:
                    h_bar = (val / max_count) * plot_h
                    x_c = pad_left + (i / 255.0) * plot_w
                    shapes.append(cv.Line(x_c, pad_top + plot_h, x_c, pad_top + plot_h - h_bar, paint=spike_paint))
        else:
            # Para imagem contínua: gera polígono preenchido suave + linha de contorno
            shapes.extend(self._build_curve_paths(counts, max_count, pad_left, pad_top, plot_w, plot_h, self._color, opacity=0.35))

        self._canvas.shapes = shapes

    def _render_rgb(self, rgb_hists: dict[str, HistogramData]) -> None:
        """Renderiza os histogramas sobrepostos dos 3 canais RGB no Canvas."""
        counts_r = rgb_hists.get("R", HistogramData(np.zeros(256), np.arange(257))).counts
        counts_g = rgb_hists.get("G", HistogramData(np.zeros(256), np.arange(257))).counts
        counts_b = rgb_hists.get("B", HistogramData(np.zeros(256), np.arange(257))).counts

        max_count = max(float(np.max(counts_r)), float(np.max(counts_g)), float(np.max(counts_b)), 1.0)

        self._peak_badge.value = f"Pico: {int(max_count):,} px"

        # Monta legenda cromática RGB
        self._legend_row.controls = [
            ft.Row([ft.Container(width=10, height=10, bgcolor="#e53935", border_radius=2), ft.Text("R", size=10, weight=ft.FontWeight.BOLD)], spacing=2),
            ft.Row([ft.Container(width=10, height=10, bgcolor="#43a047", border_radius=2), ft.Text("G", size=10, weight=ft.FontWeight.BOLD)], spacing=2),
            ft.Row([ft.Container(width=10, height=10, bgcolor="#1e88e5", border_radius=2), ft.Text("B", size=10, weight=ft.FontWeight.BOLD)], spacing=2),
        ]
        self._legend_row.visible = True

        shapes: list[Any] = []
        pad_left = 32.0
        pad_right = 16.0
        pad_top = 10.0
        pad_bottom = 22.0

        c_w = float(self._chart_width or 360.0)
        c_h = float(self._chart_height)

        plot_w = max(10.0, c_w - pad_left - pad_right)
        plot_h = max(10.0, c_h - pad_top - pad_bottom)

        self._draw_grid_and_axes(shapes, pad_left, pad_top, plot_w, plot_h)

        # Desenha as 3 curvas / distribuições (R, G, B)
        channels = [
            (counts_r, "#e53935"),
            (counts_g, "#43a047"),
            (counts_b, "#1e88e5"),
        ]

        n_bins = 256
        bar_w = plot_w / float(n_bins)

        for counts, hex_color in channels:
            if self._is_quantized:
                spike_paint = ft.Paint(stroke_width=max(2.0, bar_w * 1.2), color=hex_color, style=ft.PaintingStyle.STROKE)
                for i in range(n_bins):
                    val = counts[i]
                    if val > 0:
                        h_bar = (val / max_count) * plot_h
                        x_c = pad_left + (i / 255.0) * plot_w
                        shapes.append(cv.Line(x_c, pad_top + plot_h, x_c, pad_top + plot_h - h_bar, paint=spike_paint))
            else:
                shapes.extend(self._build_curve_paths(counts, max_count, pad_left, pad_top, plot_w, plot_h, hex_color, opacity=0.20))

        self._canvas.shapes = shapes

    @staticmethod
    def _draw_grid_and_axes(
        shapes: list[Any],
        pad_left: float,
        pad_top: float,
        plot_w: float,
        plot_h: float,
    ) -> None:
        """Desenha as linhas de grade de fundo, marcadores numéricos e eixos cartesianos."""
        grid_paint = ft.Paint(stroke_width=1, color=ft.Colors.OUTLINE_VARIANT, style=ft.PaintingStyle.STROKE)
        axis_paint = ft.Paint(stroke_width=1.5, color=ft.Colors.ON_SURFACE_VARIANT, style=ft.PaintingStyle.STROKE)

        # Linhas horizontais (0%, 50%, 100%)
        for frac in (0.0, 0.5, 1.0):
            y_pos = pad_top + plot_h * (1.0 - frac)
            shapes.append(cv.Line(pad_left, y_pos, pad_left + plot_w, y_pos, paint=grid_paint))

        # Linhas verticais e rótulos de intensidade (0, 64, 128, 192, 255)
        text_style = ft.TextStyle(size=9, color=ft.Colors.ON_SURFACE_VARIANT)
        for intensity in (0, 64, 128, 192, 255):
            x_pos = pad_left + (intensity / 255.0) * plot_w
            shapes.append(cv.Line(x_pos, pad_top, x_pos, pad_top + plot_h, paint=grid_paint))
            shapes.append(cv.Text(x_pos - 6, pad_top + plot_h + 4, str(intensity), style=text_style))

        # Eixo inferior
        shapes.append(cv.Line(pad_left, pad_top + plot_h, pad_left + plot_w, pad_top + plot_h, paint=axis_paint))

    @staticmethod
    def _build_curve_paths(
        counts: np.ndarray,
        max_count: float,
        pad_left: float,
        pad_top: float,
        plot_w: float,
        plot_h: float,
        color: str,
        opacity: float = 0.35,
    ) -> list[Any]:
        """Gera os caminhos de preenchimento e contorno de uma distribuição de histograma."""
        fill_elements = [cv.Path.MoveTo(pad_left, pad_top + plot_h)]
        stroke_elements = []

        first = True
        n_bins = len(counts)
        for i in range(n_bins):
            x_val = pad_left + (i / 255.0) * plot_w
            h_val = (counts[i] / max_count) * plot_h
            y_val = pad_top + plot_h - h_val

            fill_elements.append(cv.Path.LineTo(x_val, y_val))
            if first:
                stroke_elements.append(cv.Path.MoveTo(x_val, y_val))
                first = False
            else:
                stroke_elements.append(cv.Path.LineTo(x_val, y_val))

        fill_elements.append(cv.Path.LineTo(pad_left + plot_w, pad_top + plot_h))
        fill_elements.append(cv.Path.Close())

        fill_color = ft.Colors.with_opacity(opacity, color) if hasattr(ft.Colors, "with_opacity") else color
        fill_paint = ft.Paint(color=fill_color, style=ft.PaintingStyle.FILL)
        stroke_paint = ft.Paint(stroke_width=1.5, color=color, style=ft.PaintingStyle.STROKE)

        return [
            cv.Path(elements=fill_elements, paint=fill_paint),
            cv.Path(elements=stroke_elements, paint=stroke_paint),
        ]


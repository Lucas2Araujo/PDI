"""
single_view.py — Aba de Processamento Individual de Imagens.

Permite ao usuário:
  - Selecionar uma imagem via FilePicker (com downscaling preventivo de até 800×800 px).
  - Escolher o método de conversão para tons de cinza / isolamento de canais RGB.
  - Ajustar o nível de bits (1–8) por um Slider interativo.
  - Selecionar a técnica de quantização (Uniforme, K-Means com Lazy Loading, Histograma ou Comparação).
  - Visualizar o resultado de forma 100% nativa sem Matplotlib:
      * 📊 Gráfico Cinza: Imagens lado a lado + Histogramas Nativos Flet (Canvas)
      * 🎨 Gráfico Colorido: Imagem RGB + Histograma RGB sobreposto + Quantizada
      * 🖼️ Apenas Imagem Processada (imagem pura em alta definição com zoom 10×)
      * 🌓 Lado a Lado: Cinza × Quantizada
      * 🌈 Lado a Lado: Colorida × Quantizada
      * 📑 Grade Tripla (Colorida × Cinza × Quantizada)
  - Inspecionar as Entranhas do Processamento didático (raio-x e mapa térmico leve).
  - Salvar/Baixar o resultado em disco ou navegador.
"""

import asyncio
import gc
from pathlib import Path
import threading
from typing import Any, Callable

import flet as ft
import numpy as np

from src.core.grayscale import (
    GrayscaleMethod,
    colorize_channel,
    get_channel_color_hex,
    get_channel_color_name,
    is_channel_isolation,
    isolate_channel_rgb,
    method_label,
    to_grayscale,
)
from src.core.histogram import (
    calculate_metrics,
    generate_color_comparison_figure,
    generate_comparison_figure,
    generate_dither_comparison_figure,
    generate_full_comparison_figure,
)
from src.core.image_io import (
    MAX_IMAGE_DIMENSION,
    MAX_THUMBNAIL_DIMENSION,
    make_thumbnail_png,
    open_and_downscale_image,
)
from src.core.quantization import (
    QuantizationTechnique,
    get_kmeans_class,
    is_kmeans_loaded,
    quantize,
    quantizacao_dithering_floyd_steinberg,
    technique_label,
)
from src.core.samples import (
    SAMPLE_AYLA_NAME,
    SAMPLE_BENCHMARK_NAME,
    SAMPLE_LENA_NAME,
    SAMPLE_PENTAGONO_NAME,
    SAMPLE_PORTRAIT_NAME,
    get_sample_path,
    load_sample_array,
)
from src.ui import theme
from src.ui.common import (
    _GRAYSCALE_DETAILS,
    _TECHNIQUE_OPTIONS,
    TRANSPARENT_PIXEL_PNG_URI,
    _bytes_to_data_uri,
    _ndarray_to_png_bytes,
    _register_file_pickers,
)
from src.ui.components.histogram_chart import NativeHistogramChart
from src.ui.dialogs import open_histogram_zoom_dialog, open_inspector_dialog, open_zoom_dialog


class SingleView(ft.Column):
    """
    View de processamento individual de imagens com suporte a preview instantâneo,
    gráficos nativos Flet, processamento assíncrono e zoom interativo.
    """

    def __init__(self, page: ft.Page) -> None:
        super().__init__(
            scroll=ft.ScrollMode.AUTO,
            spacing=16,
            expand=True,
        )
        self._page = page
        self._source_path: Path | None = None
        self._loaded_array: np.ndarray | None = None
        self._raw_image: np.ndarray | None = None
        self._gray_image: np.ndarray | None = None
        self._quantized_image: np.ndarray | None = None
        self._is_color = False

        # Buffers de bytes para exibição, preview e download
        self._input_image_bytes: bytes | None = None
        self._quantized_image_bytes: bytes | None = None
        self._gray_image_bytes: bytes | None = None
        self._color_image_bytes: bytes | None = None
        self._figure_bytes: bytes | None = None
        self._color_figure_bytes: bytes | None = None
        self._dither_figure_bytes: bytes | None = None

        # Buffers e métricas para Aprimoramento / Pós-Processamento (Dithering)
        self._dither_image: np.ndarray | None = None
        self._dither_image_bytes: bytes | None = None
        self._direct_quantized_image: np.ndarray | None = None
        self._direct_quantized_bytes: bytes | None = None
        self._gray_quantized_image: np.ndarray | None = None
        self._gray_quantized_bytes: bytes | None = None
        self._direct_metrics: Any | None = None
        self._kmeans_metrics: Any | None = None
        self._dither_metrics: Any | None = None
        self._enhancement_enabled: bool = False

        self._selected_technique_key: QuantizationTechnique | str = QuantizationTechnique.UNIFORM
        self._selected_gray_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE
        self._convert_to_gray: bool = True
        self._active_view_mode: str = "graph"
        self._active_single_algo: str = "kmeans"
        self._is_processing: bool = False
        self._background_tasks: set[asyncio.Task] = set()

        self._build_controls()
        self._assemble_layout()

    @property
    def _image_stem(self) -> str:
        """Retorna o stem (nome base sem extensão) da imagem selecionada ou fallback padrão."""
        return self._source_path.stem if self._source_path else "imagem"

    # -----------------------------------------------------------------------
    # Construção Modular dos Controles
    # -----------------------------------------------------------------------

    def _build_controls(self) -> None:
        """Inicializa todos os controles da view organizados por componentes."""
        # 1. Serviços e Pickers
        self._file_picker = ft.FilePicker()
        self._save_picker = ft.FilePicker()
        _register_file_pickers(self._page, self._file_picker, self._save_picker)

        self._path_label = ft.Text(
            "Nenhuma imagem selecionada",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
            italic=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        # 2. Sub-blocos de controle
        self._build_input_preview_controls()
        self._build_preprocess_controls()
        self._build_grayscale_controls()
        self._build_quantization_controls()
        self._build_action_buttons()
        self._build_metrics_controls()
        self._build_execution_summary_card()
        self._build_view_mode_controls()

    def _build_input_preview_controls(self) -> None:
        """Constrói o card de visualização instantânea da imagem carregada."""
        self._input_thumbnail = ft.Image(
            src=TRANSPARENT_PIXEL_PNG_URI,
            width=100,
            height=100,
            fit=getattr(ft.BoxFit, "COVER", None) if hasattr(ft, "BoxFit") else None,
            border_radius=8,
        )
        self._input_name_text = ft.Text(
            "",
            size=theme.FONT_BODY,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.ON_SURFACE,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._input_dim_badge = theme.metric_badge("Dimensões", "—")
        self._input_type_badge = theme.metric_badge("Espaço de Cores", "—")
        self._input_orig_badge = theme.metric_badge("Origem", "—")

        self._btn_zoom_input = ft.Button(
            content="Visualizar Original com Zoom",
            icon=ft.Icons.ZOOM_IN,
            on_click=lambda _: self._open_zoom_dialog(
                f"Imagem de Entrada Original — {self._input_name_text.value}",
                self._input_image_bytes,
                default_filename=f"{self._image_stem}_original.png",
            ),
            bgcolor=theme.PRIMARY_LIGHT,
            color="#FFFFFF",
        )
        self._btn_download_input = ft.OutlinedButton(
            content="💾 Baixar Original",
            icon=ft.Icons.DOWNLOAD,
            on_click=lambda _: self._trigger_download(
                self._input_image_bytes,
                f"{self._image_stem}_original.png",
            ),
        )

        self._input_preview_card = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        content=self._input_thumbnail,
                        border_radius=8,
                        border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
                        on_click=lambda _: self._open_zoom_dialog(
                            f"Imagem de Entrada Original — {self._input_name_text.value}",
                            self._input_image_bytes,
                            default_filename=f"{self._image_stem}_original.png",
                        ),
                        ink=True,
                        tooltip="Clique para abrir a foto original no pop-up de zoom",
                    ),
                    ft.Column(
                        controls=[
                            self._input_name_text,
                            ft.Row(
                                controls=[
                                    self._input_dim_badge,
                                    self._input_type_badge,
                                    self._input_orig_badge,
                                ],
                                spacing=8,
                                wrap=True,
                            ),
                            ft.Row(
                                controls=[self._btn_zoom_input, self._btn_download_input],
                                spacing=8,
                                wrap=True,
                            ),
                        ],
                        spacing=6,
                        expand=True,
                    ),
                ],
                spacing=14,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=10,
            padding=12,
            visible=False,
        )

    def _build_preprocess_controls(self) -> None:
        """Constrói o seletor explícito de pré-processamento (Tons de Cinza vs Colorido RGB)."""
        self._convert_grayscale_toggle = ft.SegmentedButton(
            segments=[
                ft.Segment(
                    value="yes",
                    label=ft.Text("Sim (Tons de Cinza)", size=theme.FONT_BODY),
                    icon=ft.Icon(ft.Icons.TONALITY),
                ),
                ft.Segment(
                    value="no",
                    label=ft.Text("Não (Preservar RGB)", size=theme.FONT_BODY),
                    icon=ft.Icon(ft.Icons.PALETTE),
                ),
            ],
            selected=["yes"],
            on_change=self._on_convert_grayscale_toggled,
            show_selected_icon=False,
        )
        self._preprocess_box = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.SETTINGS_SUGGEST, size=18, color=theme.PRIMARY_LIGHT),
                            ft.Text(
                                "Modo de Entrada / Pré-Processamento:",
                                weight=ft.FontWeight.BOLD,
                                size=theme.FONT_SUBTITLE,
                                color=ft.Colors.ON_SURFACE,
                            ),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Text(
                        "Converter para Tons de Cinza antes de processar?",
                        size=theme.FONT_BODY,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.ON_SURFACE,
                    ),
                    ft.Row(controls=[self._convert_grayscale_toggle], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Text(
                        "• Se MARCADO (Sim): converte a imagem para 1 canal (Grayscale) antes de entrar no quantizador.\n"
                        "• Se DESMARCADO (Não): preserva os 3 canais (RGB) e quantiza no espaço tridimensional de cores com histogramas sobrepostos.",
                        size=theme.FONT_CAPTION,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                ],
                spacing=6,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=10,
        )

    def _build_grayscale_controls(self) -> None:
        """Constrói os seletores de algoritmo e painel didático de tons de cinza."""
        self._gray_category_selector = ft.SegmentedButton(
            segments=[
                ft.Segment(value="weighted", label=ft.Text("Ponderação / Média"), icon=ft.Icon(ft.Icons.AUTO_AWESOME)),
                ft.Segment(value="channels", label=ft.Text("Isolamento de Canais (RGB)"), icon=ft.Icon(ft.Icons.PALETTE)),
            ],
            selected=["weighted"],
            on_change=self._on_gray_category_changed,
        )

        self._gray_options_selector = ft.SegmentedButton(
            segments=[
                ft.Segment(value=str(GrayscaleMethod.LUMINANCE.value), label=ft.Text("Luminância ITU-R BT.601"), icon=ft.Icon(ft.Icons.VISIBILITY)),
                ft.Segment(value=str(GrayscaleMethod.AVERAGE.value), label=ft.Text("Média Aritmética"), icon=ft.Icon(ft.Icons.CALCULATE)),
            ],
            selected=[str(GrayscaleMethod.LUMINANCE.value)],
            on_change=self._on_gray_method_segmented_changed,
        )

        details = _GRAYSCALE_DETAILS[GrayscaleMethod.LUMINANCE]
        self._gray_info_title = ft.Text(details["title"], size=theme.FONT_BODY, weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE)
        self._gray_info_formula = ft.Text(f"Fórmula: {details['formula']}", size=theme.FONT_CAPTION, weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT)
        self._gray_info_desc = ft.Text(details["desc"], size=theme.FONT_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT)

        self._gray_info_box = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(controls=[ft.Icon(ft.Icons.INFO_OUTLINE, size=18, color=theme.PRIMARY_LIGHT), self._gray_info_title], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    self._gray_info_formula,
                    self._gray_info_desc,
                ],
                spacing=4,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=10,
        )

        self._gray_section_container = ft.Column(
            controls=[
                ft.Divider(height=1),
                ft.Text("Método de Conversão para Tons de Cinza:", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE),
                ft.Row(controls=[self._gray_category_selector], scroll=ft.ScrollMode.AUTO, alignment=ft.MainAxisAlignment.CENTER),
                ft.Row(controls=[self._gray_options_selector], scroll=ft.ScrollMode.AUTO, alignment=ft.MainAxisAlignment.CENTER),
                self._gray_info_box,
            ],
            spacing=8,
            visible=True,
        )

    def _build_quantization_controls(self) -> None:
        """Constrói o dropdown de técnicas e o slider de bits."""
        self._technique_dropdown = ft.Dropdown(
            label="Técnica de Quantização",
            options=[
                ft.dropdown.Option(
                    key=str(t.value) if isinstance(t, QuantizationTechnique) else t,
                    text=label,
                )
                for t, label in _TECHNIQUE_OPTIONS
            ],
            value=str(QuantizationTechnique.UNIFORM.value),
            color=ft.Colors.ON_SURFACE,
            on_select=self._on_technique_changed,
        )

        self._bits_value = 4
        self._bits_label = ft.Text(
            f"{self._bits_value} bits  —  {2 ** self._bits_value} tons de cinza",
            size=theme.FONT_BODY,
            color=ft.Colors.ON_SURFACE,
            weight=ft.FontWeight.BOLD,
        )
        self._bits_slider_hint = ft.Text(
            "1=2 tons · 2=4 tons · 3=8 tons · 4=16 tons · 5=32 tons · 6=64 tons · 7=128 tons · 8=256 tons",
            color=ft.Colors.ON_SURFACE_VARIANT,
            size=theme.FONT_CAPTION,
            text_align=ft.TextAlign.CENTER,
        )
        self._bits_slider = ft.Slider(
            min=1,
            max=8,
            divisions=7,
            value=self._bits_value,
            label="{value} bits",
            active_color=theme.PRIMARY,
            thumb_color=theme.PRIMARY_LIGHT,
            expand=True,
            on_change=self._on_bits_changed,
        )
        self._update_bits_label()

        # ── Aprimoramento e Pós-Processamento (Dithering de Floyd-Steinberg) ──
        self._enhancement_switch = ft.Switch(
            label="Ativar Pós-Processamento: Dithering (Floyd-Steinberg)",
            value=False,
            active_color=theme.SUCCESS,
            on_change=self._on_enhancement_toggled,
        )
        self._enhancement_box = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.AUTO_FIX_HIGH, size=18, color=theme.PRIMARY_LIGHT),
                            ft.Text(
                                "Aprimoramento da Quantização (Difusão de Erro)",
                                weight=ft.FontWeight.BOLD,
                                size=theme.FONT_BODY,
                                color=ft.Colors.ON_SURFACE,
                            ),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self._enhancement_switch,
                    ft.Text(
                        "O algoritmo de Floyd-Steinberg difunde o erro residual de quantização entre os vizinhos (7/16, 3/16, 5/16, 1/16), eliminando faixas de falso contorno e gerando percepção visual superior em baixas profundidades de bits.",
                        size=theme.FONT_CAPTION,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                ],
                spacing=6,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=10,
        )

    def _build_action_buttons(self) -> None:
        """Constrói os botões de amostras, seleção de imagem e execução."""
        self._btn_select = ft.Button(
            content="Selecionar do Disco",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_select_image,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_sample_portrait = ft.OutlinedButton(
            content="🛰️ Imagem Aérea",
            tooltip="Imagem de satélite (512×512) • Foto colorida, texturas e detalhes de terreno.",
            on_click=lambda _: self._on_select_sample(SAMPLE_PORTRAIT_NAME, "Exemplo 1: Imagem Aérea"),
        )
        self._btn_sample_benchmark = ft.OutlinedButton(
            content="📊 Benchmark",
            tooltip="Benchmark Sintético (512×512) • Padrões geométricos e gradientes contínuos para análise.",
            on_click=lambda _: self._on_select_sample(SAMPLE_BENCHMARK_NAME, "Exemplo 2: Benchmark Sintético"),
        )
        self._btn_sample_lena = ft.OutlinedButton(
            content="👒 Lenna Clássica",
            tooltip="Lenna Clássica (512×512) • Imagem canônica padrão de processamento digital de imagens.",
            on_click=lambda _: self._on_select_sample(SAMPLE_LENA_NAME, "Exemplo 3: Lenna Clássica"),
        )
        self._btn_sample_ayla = ft.OutlinedButton(
            content="🐕 Ayla (HD)",
            tooltip="Ayla Foto HD (máx 800×800) • Texturas finas de pelos e iluminação natural de alta definição.",
            on_click=lambda _: self._on_select_sample(SAMPLE_AYLA_NAME, "Exemplo 4: Ayla (Foto HD)"),
        )
        self._btn_sample_pentagono = ft.OutlinedButton(
            content="🏛️ Pentágono (TIFF)",
            tooltip="Pentágono PDI (512×512) • Fotografia aérea monocromática de alta frequência espacial.",
            on_click=lambda _: self._on_select_sample(SAMPLE_PENTAGONO_NAME, "Exemplo 5: Pentágono PDI"),
        )

        self._btn_process = ft.Button(
            content="Quantizar Imagem",
            icon=ft.Icons.TUNE,
            on_click=self._on_process,
            disabled=True,
            bgcolor=theme.SUCCESS,
            color="#FFFFFF",
        )
        self._btn_convert_gray_only = ft.Button(
            content="Converter & Salvar em Tons de Cinza (8 bits)",
            icon=ft.Icons.DOWNLOAD_FOR_OFFLINE,
            on_click=self._on_convert_and_save_gray,
            disabled=True,
            bgcolor=theme.PRIMARY_DARK,
            color="#FFFFFF",
        )
        self._btn_save = ft.Button(
            content="Salvar Resultado",
            icon=ft.Icons.SAVE_ALT,
            on_click=self._on_save,
            disabled=True,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_download_all_zip = ft.Button(
            content="📦 Baixar Todas as Comparações (ZIP)",
            icon=ft.Icons.FOLDER_ZIP,
            on_click=self._on_download_all_zip,
            disabled=True,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_quick_zip = ft.OutlinedButton(
            content="📦 Baixar ZIP Completo",
            icon=ft.Icons.FOLDER_ZIP,
            on_click=self._on_download_all_zip,
            visible=False,
        )
        self._btn_inspect = ft.Button(
            content="🔬 Entranhas do Processo",
            icon=ft.Icons.ANALYTICS,
            on_click=lambda _: self._open_inspector_dialog(),
            bgcolor=theme.ACCENT,
            color="#FFFFFF",
            disabled=True,
        )

        # Indicador de progresso assíncrono
        self._progress_ring = ft.ProgressRing(visible=False, color=theme.PRIMARY, width=28, height=28)
        self._progress_label = ft.Text("", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION)

    def _build_metrics_controls(self) -> None:
        """Constrói os badges de métricas de qualidade."""
        self._badge_tech = theme.metric_badge("Técnica", "—", color=theme.SUCCESS)
        self._badge_gray_method = theme.metric_badge("Método", "—", color=theme.PRIMARY_LIGHT)
        self._badge_mse = theme.metric_badge("MSE", "—")
        self._badge_psnr = theme.metric_badge("PSNR", "—", color=theme.SUCCESS)
        self._badge_levels = theme.metric_badge("Níveis", "—")
        self._badge_time = theme.metric_badge("Tempo", "—", color=theme.WARNING)
        self._comp_metrics_box = ft.Container(visible=False)

    def _build_execution_summary_card(self) -> None:
        """Constrói o banner visual de resumo da execução ativa."""
        self._exec_badge_file = theme.metric_badge("Imagem", "—")
        self._exec_badge_mode = theme.metric_badge("Modo", "—")
        self._exec_badge_tech = theme.metric_badge("Técnica", "—", color=theme.SUCCESS)
        self._exec_badge_bits = theme.metric_badge("Resolução", "—", color=theme.PRIMARY_LIGHT)
        self._exec_badge_dither = theme.metric_badge("Dithering", "—")

        self._execution_summary_card = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.CHECK_CIRCLE, size=18, color=theme.SUCCESS),
                    ft.Text("Execução Concluída:", weight=ft.FontWeight.BOLD, size=theme.FONT_CAPTION, color=ft.Colors.ON_SURFACE),
                    self._exec_badge_file,
                    self._exec_badge_mode,
                    self._exec_badge_tech,
                    self._exec_badge_bits,
                    self._exec_badge_dither,
                ],
                spacing=8,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=ft.Padding.symmetric(horizontal=12, vertical=8) if hasattr(ft, "Padding") else 8,
            visible=False,
        )

    def _update_view_mode_buttons(self) -> None:
        """Atualiza dinamicamente as abas de visualização com os nomes reais dos algoritmos selecionados."""
        t_key = self._selected_technique_key
        bits = self._bits_value
        m_label = method_label(self._selected_gray_method) if self._convert_to_gray else "RGB"

        if t_key == "BOTH":
            # Modo Comparativo Completo: Uniforme vs K-Means (+ Floyd-Steinberg)
            segments = [
                ft.Segment(
                    value="graph",
                    label=ft.Text("Painel Geral (Uniforme × K-Means)", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.BAR_CHART),
                ),
                ft.Segment(
                    value="side_unif_kmeans",
                    label=ft.Text(f"Uniforme × K-Means ({bits}b)", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.COMPARE),
                ),
                ft.Segment(
                    value="image",
                    label=ft.Text("Apenas Quantizada (Alternar)", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.IMAGE),
                ),
                ft.Segment(
                    value="orig_uniform",
                    label=ft.Text(f"Original × Uniforme ({bits}b)", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.COMPARE_ARROWS),
                ),
                ft.Segment(
                    value="orig_kmeans",
                    label=ft.Text(f"Original × K-Means ({bits}b)", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.AUTO_AWESOME_MOTION),
                ),
                ft.Segment(
                    value="triple",
                    label=ft.Text("Grade (Orig × Unif × K-Means)", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.VIEW_COLUMN),
                ),
                ft.Segment(
                    value="dither_comp",
                    label=ft.Text(f"Uniforme × Floyd-Steinberg ({bits}b)", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.AUTO_FIX_HIGH),
                ),
            ]
        else:
            # Modo Técnica Específica
            t_short = (
                technique_label(t_key)
                if isinstance(t_key, QuantizationTechnique)
                else str(t_key)
            )
            if self._enhancement_enabled and t_key != QuantizationTechnique.FLOYD_STEINBERG:
                t_display = f"{t_short} + Dithering"
            else:
                t_display = t_short

            if self._convert_to_gray:
                segments = [
                    ft.Segment(
                        value="graph",
                        label=ft.Text(f"Painel Analítico ({t_short})", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.BAR_CHART),
                    ),
                    ft.Segment(
                        value="color_graph",
                        label=ft.Text(f"Comparação Colorida ({t_short})", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.INSERT_CHART_OUTLINED),
                    ),
                    ft.Segment(
                        value="image",
                        label=ft.Text(f"Apenas {t_display} ({bits}b)", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.IMAGE),
                    ),
                    ft.Segment(
                        value="color_side",
                        label=ft.Text(f"Original ({m_label}) × {t_short}", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.COMPARE),
                    ),
                    ft.Segment(
                        value="triple",
                        label=ft.Text(f"Grade Tripla (Orig × {m_label} × {t_short})", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.VIEW_COLUMN),
                    ),
                    ft.Segment(
                        value="dither_comp",
                        label=ft.Text(f"{t_short} × Floyd-Steinberg", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.AUTO_FIX_HIGH),
                    ),
                ]
            else:
                segments = [
                    ft.Segment(
                        value="graph",
                        label=ft.Text(f"Painel Analítico RGB ({t_short})", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.BAR_CHART),
                    ),
                    ft.Segment(
                        value="image",
                        label=ft.Text(f"Apenas {t_display} ({bits}b RGB)", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.IMAGE),
                    ),
                    ft.Segment(
                        value="color_side",
                        label=ft.Text(f"Original (RGB) × {t_short}", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.COMPARE),
                    ),
                    ft.Segment(
                        value="triple",
                        label=ft.Text("Grade Tripla (Original × RGB × Cinza)", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.VIEW_COLUMN),
                    ),
                    ft.Segment(
                        value="dither_comp",
                        label=ft.Text(f"{t_short} × Floyd-Steinberg", size=theme.FONT_CAPTION),
                        icon=ft.Icon(ft.Icons.AUTO_FIX_HIGH),
                    ),
                ]

        if hasattr(self, "_view_mode_buttons"):
            self._view_mode_buttons.segments = segments
            valid_values = [s.value for s in segments]
            if self._active_view_mode not in valid_values:
                self._active_view_mode = "graph"
                self._view_mode_buttons.selected = ["graph"]
            else:
                self._view_mode_buttons.selected = [self._active_view_mode]

    def _build_view_mode_controls(self) -> None:
        """Constrói as áreas de visualização modular do resultado com suporte a download em cada painel."""
        box_fit = getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None

        self._view_mode_buttons = ft.SegmentedButton(
            segments=[],
            selected=["graph"],
            on_change=self._on_view_mode_changed,
            visible=False,
            show_selected_icon=False,
        )
        self._update_view_mode_buttons()

        # ── Área 1: Figura Analítica Completa (Imagens + Histogramas Integrados) ──
        self._graph_display_image = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit, expand=True)
        self._btn_graph_zoom = ft.Button(
            content="🔍 Ampliar Figura / Zoom (até 10×)",
            icon=ft.Icons.ZOOM_IN,
            on_click=lambda _: self._open_graph_figure_zoom(),
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_graph_download = ft.Button(
            content="💾 Baixar Figura Analítica",
            icon=ft.Icons.DOWNLOAD,
            on_click=lambda _: self._download_active_figure(),
            bgcolor=theme.PRIMARY_LIGHT,
            color="#FFFFFF",
        )
        self._graph_image_box = ft.Container(
            content=self._graph_display_image,
            height=520,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
            on_click=lambda _: self._open_graph_figure_zoom(),
            ink=True,
            border_radius=8,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            padding=4,
            tooltip="Clique para abrir a figura completa com zoom interativo de alta definição",
        )
        self._graph_container = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Row(controls=[self._btn_graph_zoom, self._btn_graph_download], spacing=8, wrap=True),
                        ft.Row(
                            controls=[
                                ft.Icon(ft.Icons.TOUCH_APP, size=16, color=theme.PRIMARY_LIGHT),
                                ft.Text(
                                    "Clique na figura para zoom interativo (0.25× a 10×) ou use os botões de download",
                                    size=theme.FONT_CAPTION,
                                    color=ft.Colors.ON_SURFACE_VARIANT,
                                ),
                            ],
                            spacing=6,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    wrap=True,
                ),
                self._graph_image_box,
            ],
            spacing=10,
            visible=False,
        )
        self._native_graph_view = self._graph_container

        # ── Área 2: Apenas Imagem Quantizada (com Seletor Rápido de Algoritmo) ──
        self._single_display_image = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit, expand=True)
        self._btn_single_zoom = ft.Button(
            content="🔍 Ampliar em Tela Cheia / Zoom",
            icon=ft.Icons.ZOOM_IN,
            on_click=lambda _: self._open_active_single_zoom(),
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_single_download = ft.Button(
            content="💾 Baixar Esta Imagem",
            icon=ft.Icons.DOWNLOAD,
            on_click=lambda _: self._download_active_single(),
            bgcolor=theme.PRIMARY_LIGHT,
            color="#FFFFFF",
        )
        self._single_algo_switcher = ft.SegmentedButton(
            segments=[
                ft.Segment(value="kmeans", label=ft.Text("K-Means (Adaptativo)")),
                ft.Segment(value="uniform", label=ft.Text("Uniforme (Escalar)")),
                ft.Segment(value="dither", label=ft.Text("Floyd-Steinberg (Dither)")),
            ],
            selected=["kmeans"],
            on_change=self._on_single_algo_switch_changed,
            visible=False,
            show_selected_icon=False,
        )
        self._single_display_container = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Row(controls=[self._btn_single_zoom, self._btn_single_download], spacing=8, wrap=True),
                        self._single_algo_switcher,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    wrap=True,
                ),
                ft.Container(
                    content=self._single_display_image,
                    height=440,
                    alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                    on_click=lambda _: self._open_active_single_zoom(),
                    ink=True,
                    tooltip="Clique para abrir o visualizador de zoom interativo",
                    border_radius=8,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                    padding=8,
                ),
            ],
            spacing=8,
            expand=True,
            visible=False,
        )

        # ── Área 3: Comparação lado a lado (Original vs Quantizada ou Uniforme vs K-Means) ──
        self._orig_side_label = ft.Text("Imagem 1", weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT)
        self._quant_side_label = ft.Text("Imagem 2", weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT)
        self._orig_side_img = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit)
        self._quant_side_img = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit)

        self._orig_side_col, self._orig_side_img_box = self._create_card_column(
            label_ctrl=self._orig_side_label,
            image_ctrl=self._orig_side_img,
            on_zoom_fn=self._open_side_orig_zoom,
            height=380,
            on_download_fn=self._download_side_orig_slot,
        )
        self._quant_side_col, self._quant_side_img_box = self._create_card_column(
            label_ctrl=self._quant_side_label,
            image_ctrl=self._quant_side_img,
            on_zoom_fn=self._open_side_quant_zoom,
            height=380,
            on_download_fn=self._download_side_quant_slot,
        )
        self._side_by_side_container = ft.Row(
            controls=[self._orig_side_col, ft.VerticalDivider(width=1), self._quant_side_col],
            spacing=16,
            expand=True,
            visible=False,
        )

        # ── Área 4: Grade Tripla ──
        self._triple_color_label = ft.Text("1. Painel A", weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT, size=theme.FONT_CAPTION)
        self._triple_gray_label = ft.Text("2. Painel B", weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION)
        self._triple_quant_label = ft.Text("3. Painel C", weight=ft.FontWeight.BOLD, color=theme.SUCCESS, size=theme.FONT_CAPTION)

        self._triple_color_img = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit)
        self._triple_gray_img = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit)
        self._triple_quant_img = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit)

        self._triple_color_col, self._triple_color_img_box = self._create_card_column(
            label_ctrl=self._triple_color_label,
            image_ctrl=self._triple_color_img,
            on_zoom_fn=self._open_triple_col1_zoom,
            height=340,
            icon_button=True,
            on_download_fn=self._download_triple_col1,
        )
        self._triple_gray_col, self._triple_gray_img_box = self._create_card_column(
            label_ctrl=self._triple_gray_label,
            image_ctrl=self._triple_gray_img,
            on_zoom_fn=self._open_triple_col2_zoom,
            height=340,
            icon_button=True,
            on_download_fn=self._download_triple_col2,
        )
        self._triple_quant_col, self._triple_quant_img_box = self._create_card_column(
            label_ctrl=self._triple_quant_label,
            image_ctrl=self._triple_quant_img,
            on_zoom_fn=self._open_triple_col3_zoom,
            height=340,
            icon_button=True,
            on_download_fn=self._download_triple_col3,
        )

        self._triple_container = ft.Row(
            controls=[self._triple_color_col, ft.VerticalDivider(width=1), self._triple_gray_col, ft.VerticalDivider(width=1), self._triple_quant_col],
            spacing=10,
            expand=True,
            visible=False,
        )

        # ── Área 5: Comparação Antes × Depois do Pós-Processamento (Direta × Dithering) ──
        self._direct_comp_label = ft.Text("Quantização Direta", weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT)
        self._dither_comp_label = ft.Text("Com Dithering Floyd-Steinberg", weight=ft.FontWeight.BOLD, color=theme.SUCCESS)
        self._direct_comp_img = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit)
        self._dither_comp_img = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=box_fit)

        self._direct_comp_col, self._direct_comp_img_box = self._create_card_column(
            label_ctrl=self._direct_comp_label,
            image_ctrl=self._direct_comp_img,
            on_zoom_fn=self._open_dither_direct_zoom,
            height=380,
            on_download_fn=self._download_dither_direct,
        )
        self._dither_comp_col, self._dither_comp_img_box = self._create_card_column(
            label_ctrl=self._dither_comp_label,
            image_ctrl=self._dither_comp_img,
            on_zoom_fn=self._open_dither_fs_zoom,
            height=380,
            on_download_fn=self._download_dither_fs,
        )
        self._dither_comp_container = ft.Row(
            controls=[self._direct_comp_col, ft.VerticalDivider(width=1), self._dither_comp_col],
            spacing=16,
            expand=True,
            visible=False,
        )

        # Placeholder & Toolbar
        self._zoom_toolbar = ft.Row(
            controls=[
                ft.Icon(ft.Icons.PHOTO_SIZE_SELECT_ACTUAL, size=16, color=theme.PRIMARY_LIGHT),
                ft.Text(" Visualizador Interativo: Clique em qualquer imagem para abrir o pop-up com zoom até 10× e pan, ou use os botões de download individuais.", size=theme.FONT_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            visible=False,
        )
        self._result_placeholder = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(ft.Icons.IMAGE_SEARCH, size=64, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("Selecione uma imagem e clique em 'Quantizar Imagem' para ver o resultado", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_BODY),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            height=360,
            expand=True,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
        )

    # -----------------------------------------------------------------------
    # Construtores Utilitários de UI (DRY)
    # -----------------------------------------------------------------------

    def _create_preview_column(
        self,
        label_ctrl: ft.Text,
        image_ctrl: ft.Image,
        on_zoom_fn: Callable[[], None],
        bottom_ctrl: ft.Control,
        height: int,
    ) -> ft.Column:
        """Cria uma coluna padronizada de exibição com cabeçalho, imagem clicável e controle inferior."""
        zoom_btn = ft.IconButton(icon=ft.Icons.ZOOM_IN, icon_size=18, tooltip="Ampliar", on_click=lambda _: on_zoom_fn())
        header = ft.Row(controls=[label_ctrl, zoom_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        img_container = ft.Container(
            content=image_ctrl,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=4,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
            on_click=lambda _: on_zoom_fn(),
            ink=True,
        )
        return ft.Column(
            controls=[header, img_container, bottom_ctrl],
            spacing=8,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.START,
        )

    def _create_card_column(
        self,
        label_ctrl: ft.Text,
        image_ctrl: ft.Image,
        on_zoom_fn: Callable[[], None],
        height: int,
        icon_button: bool = False,
        on_download_fn: Callable[[], None] | None = None,
    ) -> tuple[ft.Column, ft.Container]:
        """Cria uma coluna de imagem para comparações com botões de zoom e download individuais."""
        action_btns: list[ft.Control] = []
        if icon_button:
            action_btns.append(
                ft.IconButton(
                    icon=ft.Icons.ZOOM_IN,
                    icon_size=18,
                    tooltip="Ampliar",
                    on_click=lambda _: on_zoom_fn(),
                )
            )
            if on_download_fn is not None:
                action_btns.append(
                    ft.IconButton(
                        icon=ft.Icons.DOWNLOAD,
                        icon_size=18,
                        tooltip="Baixar esta imagem em alta resolução",
                        on_click=lambda _: on_download_fn(),
                    )
                )
        else:
            action_btns.append(
                ft.OutlinedButton(
                    content="🔍 Ampliar",
                    on_click=lambda _: on_zoom_fn(),
                )
            )
            if on_download_fn is not None:
                action_btns.append(
                    ft.Button(
                        content="💾 Baixar",
                        icon=ft.Icons.DOWNLOAD,
                        bgcolor=theme.PRIMARY_LIGHT,
                        color="#FFFFFF",
                        on_click=lambda _: on_download_fn(),
                    )
                )

        header = ft.Row(
            controls=[label_ctrl, ft.Row(controls=action_btns, spacing=4)],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            wrap=True,
        )

        img_box = ft.Container(
            content=image_ctrl,
            height=height,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
            on_click=lambda _: on_zoom_fn(),
            ink=True,
            tooltip="Clique para ampliar",
            border_radius=8,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            padding=6,
        )
        col = ft.Column(controls=[header, img_box], horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)
        return col, img_box

    def _assemble_layout(self) -> None:
        """Monta o layout da view organizando os controles em cards."""
        self.controls = [
            # Card 1: Configurações & Imagem
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("⚙️  Configurações & Imagem"),
                        ft.Divider(height=1),
                        ft.Row(controls=[self._btn_select, self._path_label], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True),
                        ft.Column(
                            controls=[
                                ft.Text("Imagens de Teste Embutidas (passe o mouse para detalhes):", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION, weight=ft.FontWeight.BOLD),
                                ft.Row(
                                    controls=[self._btn_sample_portrait, self._btn_sample_benchmark, self._btn_sample_lena, self._btn_sample_ayla, self._btn_sample_pentagono],
                                    spacing=8,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                    wrap=True,
                                    scroll=ft.ScrollMode.AUTO,
                                ),
                            ],
                            spacing=6,
                        ),
                        self._input_preview_card,
                        ft.Divider(height=1),
                        self._preprocess_box,
                        self._gray_section_container,
                        ft.Divider(height=1),
                        ft.Text("Técnica de Quantização:", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE),
                        self._technique_dropdown,
                        ft.Divider(height=1),
                        ft.Column(
                            controls=[
                                ft.Row(controls=[ft.Text("Nível de Bits:", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_BODY), self._bits_label], spacing=8),
                                ft.Row(controls=[ft.Text("1 bit", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION), self._bits_slider, ft.Text("8 bits", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION)], spacing=4),
                                self._bits_slider_hint,
                            ],
                            spacing=6,
                        ),
                        ft.Divider(height=1),
                        self._enhancement_box,
                        ft.Divider(height=1),
                        ft.Row(controls=[self._btn_process, self._btn_convert_gray_only, self._btn_save, self._btn_download_all_zip, self._btn_inspect], spacing=12, wrap=True),
                    ],
                    spacing=12,
                )
            ),
            # Card 2: Progresso
            ft.Container(
                content=ft.Row(controls=[self._progress_ring, self._progress_label], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                visible=False,
            ),
            # Card 3: Métricas
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("📊  Métricas de Qualidade & Identificação"),
                        ft.Divider(height=1),
                        ft.Row(controls=[self._badge_tech, self._badge_gray_method, self._badge_mse, self._badge_psnr, self._badge_levels, self._badge_time], spacing=10, wrap=True),
                        self._comp_metrics_box,
                        ft.Text("MSE: erro quadrático médio por pixel (menor = melhor) · PSNR: relação sinal-ruído de pico em dB (maior = melhor)", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION),
                    ],
                    spacing=10,
                )
            ),
            # Card 4: Área de resultado com seletor de visualização
            theme.card(
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                theme.section_title("🖼️  Visualização do Resultado"),
                                ft.Row(controls=[self._btn_quick_zip, self._view_mode_buttons], scroll=ft.ScrollMode.AUTO, spacing=8),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            wrap=True,
                            spacing=10,
                        ),
                        self._execution_summary_card,
                        self._zoom_toolbar,
                        ft.Divider(height=1),
                        self._result_placeholder,
                        self._graph_container,
                        self._single_display_container,
                        self._side_by_side_container,
                        self._triple_container,
                        self._dither_comp_container,
                    ],
                    spacing=12,
                )
            ),
        ]

    # -----------------------------------------------------------------------
    # Visualizador de Zoom Modal & Ações de Download
    # -----------------------------------------------------------------------

    def _open_zoom_dialog(self, title: str, image_bytes: bytes | str | None, default_filename: str | None = None) -> None:
        """Abre o visualizador modal com zoom interativo e botão de download integrado."""
        on_download = None
        if image_bytes and isinstance(image_bytes, bytes):
            filename = default_filename or "imagem.png"
            on_download = lambda: self._trigger_download(image_bytes, filename)
        open_zoom_dialog(self._page, title, image_bytes, on_download=on_download)

    async def _save_image_data(self, data: bytes | None, default_filename: str) -> None:
        """Abre o FilePicker e salva os bytes de imagem no disco ou navegador."""
        if not data:
            self._show_message("Nenhum dado de imagem para salvar.", theme.WARNING)
            return
        try:
            save_path = await self._save_picker.save_file(
                dialog_title=f"Salvar Imagem — {default_filename}",
                file_name=default_filename,
                allowed_extensions=["png"],
                src_bytes=data,
            )
            if save_path and not getattr(self._page, "web", False):
                Path(save_path).write_bytes(data)
            self._show_message(f"✅ Imagem '{default_filename}' salva com sucesso!", theme.SUCCESS)
        except Exception as exc:
            self._show_message(f"Erro ao salvar arquivo: {exc}", theme.ACCENT)

    def _trigger_download(self, data: bytes | None, default_filename: str) -> None:
        """Dispara de forma não-bloqueante a rotina assíncrona de download/salvamento."""
        if not data:
            return

        async def _runner():
            await self._save_image_data(data, default_filename)

        if hasattr(self._page, "run_task"):
            self._page.run_task(_runner)
        else:
            task = asyncio.create_task(_runner())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def _get_active_figure_bytes_and_name(self) -> tuple[bytes | None, str]:
        """Retorna os bytes e nome de arquivo para a figura analítica exibida."""
        stem = self._image_stem
        if self._active_view_mode == "color_graph" and self._color_figure_bytes:
            return self._color_figure_bytes, f"{stem}_painel_colorido_{self._bits_value}bits.png"
        if self._active_view_mode == "dither_comp" and self._dither_figure_bytes:
            return self._dither_figure_bytes, f"{stem}_painel_dithering_{self._bits_value}bits.png"
        if self._figure_bytes:
            return self._figure_bytes, f"{stem}_painel_analitico_{self._bits_value}bits.png"
        return self._quantized_image_bytes, f"{stem}_quantizada_{self._bits_value}bits.png"

    def _download_active_figure(self) -> None:
        """Dispara o download da figura analítica ativa."""
        data, name = self._get_active_figure_bytes_and_name()
        self._trigger_download(data, name)

    def _get_active_single_bytes_and_name(self) -> tuple[bytes | None, str]:
        """Retorna os bytes e nome de arquivo para o modo 'Apenas Quantizada'."""
        stem = self._image_stem
        if self._selected_technique_key == "BOTH":
            if self._active_single_algo == "kmeans" and self._quantized_image_bytes:
                return self._quantized_image_bytes, f"{stem}_kmeans_{self._bits_value}bits.png"
            if self._active_single_algo == "uniform" and self._direct_quantized_bytes:
                return self._direct_quantized_bytes, f"{stem}_uniforme_{self._bits_value}bits.png"
            if self._active_single_algo == "dither" and self._dither_image_bytes:
                return self._dither_image_bytes, f"{stem}_floyd_steinberg_{self._bits_value}bits.png"

        t_slug = (
            self._selected_technique_key.name.lower()
            if isinstance(self._selected_technique_key, QuantizationTechnique)
            else str(self._selected_technique_key).lower()
        )
        if self._enhancement_enabled and self._selected_technique_key != QuantizationTechnique.FLOYD_STEINBERG:
            t_slug = f"{t_slug}_dither"
        return self._quantized_image_bytes, f"{stem}_{t_slug}_{self._bits_value}bits.png"

    def _download_active_single(self) -> None:
        """Dispara o download da imagem quantizada ativa do modo único."""
        data, name = self._get_active_single_bytes_and_name()
        self._trigger_download(data, name)

    def _on_single_algo_switch_changed(self, event: ft.ControlEvent) -> None:
        """Alterna a imagem quantizada em exibição no modo 'Apenas Quantizada' quando em técnica BOTH."""
        selected = getattr(event.control, "selected", None)
        if not selected:
            return
        self._active_single_algo = next(iter(selected))
        data, _ = self._get_active_single_bytes_and_name()
        if data:
            self._single_display_image.src = _bytes_to_data_uri(data)
            self._page.update()

    def _open_active_single_zoom(self) -> None:
        """Abre o visualizador de zoom para a imagem quantizada ativa com download integrado."""
        data, name = self._get_active_single_bytes_and_name()
        if data:
            self._open_zoom_dialog(f"Imagem Quantizada ({name})", data, default_filename=name)

    def _get_side_orig_bytes_and_name(self) -> tuple[bytes | None, str]:
        """Retorna os bytes e nome de arquivo para o slot 1 da comparação lado a lado."""
        stem = self._image_stem
        if self._selected_technique_key == "BOTH" and self._active_view_mode == "side_unif_kmeans":
            return self._direct_quantized_bytes, f"{stem}_uniforme_{self._bits_value}bits.png"
        if self._active_view_mode in ("color_side", "color_graph") and self._color_image_bytes:
            return self._color_image_bytes, f"{stem}_original_rgb.png"
        if self._gray_image_bytes:
            return self._gray_image_bytes, f"{stem}_pre_processamento.png"
        return self._input_image_bytes, f"{stem}_original.png"

    def _open_side_orig_zoom(self) -> None:
        """Abre zoom para a imagem do slot esquerdo da comparação."""
        data, name = self._get_side_orig_bytes_and_name()
        label = self._orig_side_label.value or "Imagem 1"
        self._open_zoom_dialog(label, data, default_filename=name)

    def _download_side_orig_slot(self) -> None:
        """Dispara o download do slot esquerdo da comparação lado a lado."""
        data, name = self._get_side_orig_bytes_and_name()
        self._trigger_download(data, name)

    def _get_side_quant_bytes_and_name(self) -> tuple[bytes | None, str]:
        """Retorna os bytes e nome de arquivo para o slot 2 da comparação lado a lado."""
        stem = self._image_stem
        if self._selected_technique_key == "BOTH":
            if self._active_view_mode == "orig_uniform":
                return self._direct_quantized_bytes, f"{stem}_uniforme_{self._bits_value}bits.png"
            return self._quantized_image_bytes, f"{stem}_kmeans_{self._bits_value}bits.png"
        t_slug = (
            self._selected_technique_key.name.lower()
            if isinstance(self._selected_technique_key, QuantizationTechnique)
            else str(self._selected_technique_key).lower()
        )
        return self._quantized_image_bytes, f"{stem}_{t_slug}_{self._bits_value}bits.png"

    def _open_side_quant_zoom(self) -> None:
        """Abre zoom para a imagem do slot direito da comparação."""
        data, name = self._get_side_quant_bytes_and_name()
        label = self._quant_side_label.value or "Imagem 2"
        self._open_zoom_dialog(label, data, default_filename=name)

    def _download_side_quant_slot(self) -> None:
        """Dispara o download do slot direito da comparação lado a lado."""
        data, name = self._get_side_quant_bytes_and_name()
        self._trigger_download(data, name)

    def _get_triple_col1_bytes_and_name(self) -> tuple[bytes | None, str]:
        stem = self._image_stem
        data = self._color_image_bytes or self._input_image_bytes or self._gray_image_bytes
        return data, f"{stem}_painel1_original.png"

    def _open_triple_col1_zoom(self) -> None:
        data, name = self._get_triple_col1_bytes_and_name()
        label = self._triple_color_label.value or "Painel 1"
        self._open_zoom_dialog(label, data, default_filename=name)

    def _download_triple_col1(self) -> None:
        data, name = self._get_triple_col1_bytes_and_name()
        self._trigger_download(data, name)

    def _get_triple_col2_bytes_and_name(self) -> tuple[bytes | None, str]:
        stem = self._image_stem
        if self._selected_technique_key == "BOTH":
            return self._direct_quantized_bytes, f"{stem}_painel2_uniforme_{self._bits_value}bits.png"
        if not self._convert_to_gray:
            return self._quantized_image_bytes, f"{stem}_painel2_quantizada_rgb_{self._bits_value}bits.png"
        return self._gray_image_bytes, f"{stem}_painel2_cinza_8bits.png"

    def _open_triple_col2_zoom(self) -> None:
        data, name = self._get_triple_col2_bytes_and_name()
        label = self._triple_gray_label.value or "Painel 2"
        self._open_zoom_dialog(label, data, default_filename=name)

    def _download_triple_col2(self) -> None:
        data, name = self._get_triple_col2_bytes_and_name()
        self._trigger_download(data, name)

    def _get_triple_col3_bytes_and_name(self) -> tuple[bytes | None, str]:
        stem = self._image_stem
        if self._selected_technique_key == "BOTH":
            return self._quantized_image_bytes, f"{stem}_painel3_kmeans_{self._bits_value}bits.png"
        if not self._convert_to_gray:
            return self._gray_quantized_bytes, f"{stem}_painel3_quantizada_cinza_{self._bits_value}bits.png"
        return self._quantized_image_bytes, f"{stem}_painel3_quantizada_{self._bits_value}bits.png"

    def _open_triple_col3_zoom(self) -> None:
        data, name = self._get_triple_col3_bytes_and_name()
        label = self._triple_quant_label.value or "Painel 3"
        self._open_zoom_dialog(label, data, default_filename=name)

    def _download_triple_col3(self) -> None:
        data, name = self._get_triple_col3_bytes_and_name()
        self._trigger_download(data, name)

    def _open_dither_direct_zoom(self) -> None:
        stem = self._image_stem
        self._open_zoom_dialog(
            self._direct_comp_label.value or "Quantização Direta",
            self._direct_quantized_bytes,
            default_filename=f"{stem}_quantizada_direta_{self._bits_value}bits.png",
        )

    def _download_dither_direct(self) -> None:
        stem = self._image_stem
        self._trigger_download(self._direct_quantized_bytes, f"{stem}_quantizada_direta_{self._bits_value}bits.png")

    def _open_dither_fs_zoom(self) -> None:
        stem = self._image_stem
        self._open_zoom_dialog(
            self._dither_comp_label.value or "Com Dithering Floyd-Steinberg",
            self._dither_image_bytes,
            default_filename=f"{stem}_floyd_steinberg_{self._bits_value}bits.png",
        )

    def _download_dither_fs(self) -> None:
        stem = self._image_stem
        self._trigger_download(self._dither_image_bytes, f"{stem}_floyd_steinberg_{self._bits_value}bits.png")

    def _open_orig_chart_zoom(self) -> None:
        """Abre zoom dedicado com estatísticas para o histograma da imagem de entrada."""
        is_rgb = bool(self._active_view_mode == "color_graph" and self._is_color)
        data = self._raw_image if is_rgb else self._gray_image
        if data is not None:
            m_label = self._selected_gray_method.name.capitalize()
            title = "Histograma — Entrada Colorida (RGB)" if is_rgb else f"Histograma — Entrada ({m_label})"
            open_histogram_zoom_dialog(
                page=self._page,
                title=title,
                data=data,
                color=theme.PRIMARY_LIGHT,
                is_rgb=is_rgb,
                is_quantized=False,
            )

    def _open_graph_figure_zoom(self) -> None:
        """Abre o visualizador modal com zoom interativo (até 10×) para a figura analítica completa."""
        data, name = self._get_active_figure_bytes_and_name()
        if data:
            title = "Figura Analítica Completa"
            if self._active_view_mode == "color_graph":
                title = "Figura Analítica Completa (Colorida RGB + Quantizada)"
            elif self._active_view_mode == "dither_comp":
                title = "Figura Comparativa: Direta vs Floyd-Steinberg"
            self._open_zoom_dialog(title, data, default_filename=name)

    def _open_quant_chart_zoom(self) -> None:
        """Abre zoom dedicado com estatísticas para o histograma da imagem quantizada."""
        if self._quantized_image is not None:
            t_name = (
                technique_label(self._selected_technique_key)
                if isinstance(self._selected_technique_key, QuantizationTechnique)
                else str(self._selected_technique_key)
            )
            title = f"Histograma — Quantizada ({t_name} • {self._bits_value} bits)"
            open_histogram_zoom_dialog(
                page=self._page,
                title=title,
                data=self._quantized_image,
                color=theme.SUCCESS,
                is_rgb=False,
                is_quantized=True,
            )

    def _update_execution_summary(self) -> None:
        """Atualiza o banner visual superior com os parâmetros da execução recém-finalizada."""
        fname = self._source_path.name if self._source_path else "Imagem de Teste"
        self._exec_badge_file.content.controls[1].value = fname

        mode_str = "Tons de Cinza" if self._convert_to_gray else "Colorido RGB"
        if self._convert_to_gray:
            mode_str += f" ({method_label(self._selected_gray_method)})"
        self._exec_badge_mode.content.controls[1].value = mode_str

        if self._selected_technique_key == "BOTH":
            t_str = "Uniforme vs K-Means"
        else:
            t_str = (
                technique_label(self._selected_technique_key)
                if isinstance(self._selected_technique_key, QuantizationTechnique)
                else str(self._selected_technique_key)
            )
        self._exec_badge_tech.content.controls[1].value = t_str

        if self._convert_to_gray:
            self._exec_badge_bits.content.controls[1].value = f"{self._bits_value} bits ({2**self._bits_value} tons)"
        else:
            if self._selected_technique_key == QuantizationTechnique.KMEANS:
                self._exec_badge_bits.content.controls[1].value = f"{2**self._bits_value} cores"
            else:
                self._exec_badge_bits.content.controls[1].value = f"{self._bits_value} b/canal ({(2**self._bits_value)**3} cores)"

        dit_str = "Ativo (Floyd-Steinberg)" if self._enhancement_enabled else "Inativo"
        self._exec_badge_dither.content.controls[1].value = dit_str
        self._execution_summary_card.visible = True
        self._btn_download_all_zip.disabled = False
        self._btn_quick_zip.visible = True

    def _generate_comparison_zip_bytes(self) -> tuple[bytes, str]:
        """Compacta todos os resultados de imagem, figuras analíticas e relatório em um arquivo ZIP."""
        import io
        import zipfile
        buf = io.BytesIO()
        stem = self._image_stem
        zip_filename = f"{stem}_comparações_{self._bits_value}bits.zip"

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Imagem Original de Entrada
            if self._color_image_bytes:
                zf.writestr(f"1_original_{stem}_colorida_rgb.png", self._color_image_bytes)
            elif self._input_image_bytes:
                zf.writestr(f"1_original_{stem}.png", self._input_image_bytes)

            # 2. Imagem Pré-Processada (Cinza ou Canal Isolado)
            if self._convert_to_gray and self._gray_image_bytes:
                if is_channel_isolation(self._selected_gray_method):
                    ch_name = get_channel_color_name(self._selected_gray_method).lower()
                    zf.writestr(f"2_pre_processamento_canal_{ch_name}.png", self._gray_image_bytes)
                else:
                    m_slug = self._selected_gray_method.name.lower()
                    zf.writestr(f"2_pre_processamento_cinza_{m_slug}.png", self._gray_image_bytes)

            # 3. Imagens Quantizadas
            if self._selected_technique_key == "BOTH":
                if self._direct_quantized_bytes:
                    zf.writestr(f"3_quantizada_uniforme_{self._bits_value}bits.png", self._direct_quantized_bytes)
                if self._quantized_image_bytes:
                    zf.writestr(f"4_quantizada_kmeans_{self._bits_value}bits.png", self._quantized_image_bytes)
                if self._dither_image_bytes:
                    zf.writestr(f"5_quantizada_floyd_steinberg_{self._bits_value}bits.png", self._dither_image_bytes)
            else:
                t_slug = (
                    self._selected_technique_key.name.lower()
                    if isinstance(self._selected_technique_key, QuantizationTechnique)
                    else str(self._selected_technique_key).lower()
                )
                if self._direct_quantized_bytes:
                    zf.writestr(f"3_quantizada_{t_slug}_{self._bits_value}bits.png", self._direct_quantized_bytes)
                if self._dither_image_bytes and self._selected_technique_key != QuantizationTechnique.FLOYD_STEINBERG:
                    zf.writestr(f"4_quantizada_floyd_steinberg_{self._bits_value}bits.png", self._dither_image_bytes)

            # 4. Figuras Analíticas e Gráficos
            if self._figure_bytes:
                zf.writestr(f"painel_analitico_histogramas_{self._bits_value}bits.png", self._figure_bytes)
            if self._color_figure_bytes:
                zf.writestr(f"painel_analitico_colorido_{self._bits_value}bits.png", self._color_figure_bytes)
            if self._dither_figure_bytes:
                zf.writestr(f"painel_comparativo_dithering_{self._bits_value}bits.png", self._dither_figure_bytes)

            # 5. Relatório Técnico Didático em TXT
            report_text = self._build_zip_report_text()
            zf.writestr("relatorio_metricas.txt", report_text.encode("utf-8"))

        return buf.getvalue(), zip_filename

    def _build_zip_report_text(self) -> str:
        """Gera o relatório descritivo em texto para inclusão no arquivo ZIP."""
        fname = self._source_path.name if self._source_path else "Imagem de Teste"
        mode_str = "Tons de Cinza" if self._convert_to_gray else "Colorido RGB"
        m_label = method_label(self._selected_gray_method) if self._convert_to_gray else "RGB"
        t_name = (
            "Comparação Uniforme vs K-Means"
            if self._selected_technique_key == "BOTH"
            else (technique_label(self._selected_technique_key) if isinstance(self._selected_technique_key, QuantizationTechnique) else str(self._selected_technique_key))
        )

        lines = [
            "=" * 64,
            "  RELATÓRIO TÉCNICO DE PROCESSAMENTO E QUANTIZAÇÃO DIGITAL",
            "=" * 64,
            f"Arquivo de Entrada : {fname}",
            f"Modo Selecionado   : {mode_str}",
            f"Método/Espaço      : {m_label}",
            f"Técnica Aplicada   : {t_name}",
            f"Resolução de Bits  : {self._bits_value} bits",
            f"Dithering Ativo    : {'Sim (Floyd-Steinberg)' if self._enhancement_enabled else 'Não'}",
            "-" * 64,
            "MÉTRICAS QUANTITATIVAS DE QUALIDADE (FIDELIDADE):",
            "-" * 64,
        ]

        if self._direct_metrics is not None:
            name_dir = "Quantização Uniforme" if self._selected_technique_key == "BOTH" else t_name
            lines.append(f"[{name_dir}]")
            lines.append(f"  • MSE  (Erro Quadrático Médio) : {self._direct_metrics.mse:.4f}")
            lines.append(f"  • PSNR (Relação Sinal-Ruído)  : {self._direct_metrics.psnr:.2f} dB")
            n_levels = getattr(self._direct_metrics, "unique_levels", getattr(self._direct_metrics, "num_levels", "—"))
            lines.append(f"  • Níveis Efetivos de Cores/Tons: {n_levels}")
            proc_time = getattr(self._direct_metrics, "processing_time_ms", None)
            if proc_time is not None:
                lines.append(f"  • Tempo de Processamento       : {proc_time:.1f} ms")
            elif hasattr(self, "_badge_time") and self._badge_time.content.controls[1].value not in ("—", ""):
                lines.append(f"  • Tempo de Processamento       : {self._badge_time.content.controls[1].value}")
            lines.append("")

        if self._kmeans_metrics is not None:
            lines.append("[Quantização Adaptativa K-Means]")
            lines.append(f"  • MSE  (Erro Quadrático Médio) : {self._kmeans_metrics.mse:.4f}")
            lines.append(f"  • PSNR (Relação Sinal-Ruído)  : {self._kmeans_metrics.psnr:.2f} dB")
            n_levels_km = getattr(self._kmeans_metrics, "unique_levels", getattr(self._kmeans_metrics, "num_levels", "—"))
            lines.append(f"  • Níveis Efetivos de Cores/Tons: {n_levels_km}")
            proc_time_km = getattr(self._kmeans_metrics, "processing_time_ms", None)
            if proc_time_km is not None:
                lines.append(f"  • Tempo de Processamento       : {proc_time_km:.1f} ms")
            elif hasattr(self, "_badge_time") and self._badge_time.content.controls[1].value not in ("—", ""):
                lines.append(f"  • Tempo de Processamento       : {self._badge_time.content.controls[1].value}")
            lines.append("")

        if self._dither_metrics is not None:
            lines.append("[Aprimoramento com Floyd-Steinberg (Dithering)]")
            lines.append(f"  • MSE  (Erro Quadrático Médio) : {self._dither_metrics.mse:.4f}")
            lines.append(f"  • PSNR (Relação Sinal-Ruído)  : {self._dither_metrics.psnr:.2f} dB")
            n_levels_dit = getattr(self._dither_metrics, "unique_levels", getattr(self._dither_metrics, "num_levels", "—"))
            lines.append(f"  • Níveis Efetivos de Cores/Tons: {n_levels_dit}")
            proc_time_dit = getattr(self._dither_metrics, "processing_time_ms", None)
            if proc_time_dit is not None:
                lines.append(f"  • Tempo de Processamento       : {proc_time_dit:.1f} ms")
            elif hasattr(self, "_badge_time") and self._badge_time.content.controls[1].value not in ("—", ""):
                lines.append(f"  • Tempo de Processamento       : {self._badge_time.content.controls[1].value}")
            lines.append("")

        lines.extend([
            "=" * 64,
            "INTERPRETAÇÃO DIDÁTICA:",
            "- MSE (Mean Squared Error): Mede a divergência média quadrática de pixel a pixel.",
            "  Valores menores indicam maior fidelidade matemática à imagem original.",
            "- PSNR (Peak Signal-to-Noise Ratio): Mede a qualidade percebida em escala logarítmica (dB).",
            "  Valores acima de 30 dB geralmente indicam excelente preservação visual.",
            "- K-Means vs Uniforme: O K-Means adapta a paleta às cores mais frequentes da imagem,",
            "  resultando habitualmente em menor MSE e maior PSNR com a mesma quantidade de bits.",
            "=" * 64,
        ])
        return "\n".join(lines)

    async def _on_download_all_zip(self, _: ft.ControlEvent | None = None) -> None:
        """Abre o FilePicker para salvar o pacote ZIP com todas as comparações geradas."""
        if self._raw_image is None and self._loaded_array is None and not self._input_image_bytes:
            self._show_message("Execute o processamento de uma imagem antes de baixar o pacote.", theme.WARNING)
            return

        try:
            self._show_message("📦 Preparando arquivo ZIP com todas as comparações...", theme.INFO)
            zip_bytes, default_name = self._generate_comparison_zip_bytes()

            save_path = await self._save_picker.save_file(
                dialog_title="Salvar Todas as Comparações (ZIP)",
                file_name=default_name,
                allowed_extensions=["zip"],
                src_bytes=zip_bytes,
            )
            if save_path and not getattr(self._page, "web", False):
                Path(save_path).write_bytes(zip_bytes)
            self._show_message("✅ Arquivo ZIP com todas as comparações salvo com sucesso!", theme.SUCCESS)
        except Exception as exc:
            self._show_message(f"Erro ao gerar pacote ZIP: {exc}", theme.ACCENT)

    def update_responsive_layout(self, width: float | None = None, height: float | None = None) -> None:
        """Adapta o layout da view conforme as dimensões da viewport de forma simplificada."""
        w = width if width is not None else getattr(self._page, "width", None)
        is_mob = theme.is_mobile(w)
        self._side_by_side_container.controls = [
            self._orig_side_col,
            ft.Divider(height=1) if is_mob else ft.VerticalDivider(width=1),
            self._quant_side_col,
        ]
        self._triple_container.controls = [
            self._triple_color_col,
            ft.Divider(height=1) if is_mob else ft.VerticalDivider(width=1),
            self._triple_gray_col,
            ft.Divider(height=1) if is_mob else ft.VerticalDivider(width=1),
            self._triple_quant_col,
        ]
        self._dither_comp_container.controls = [
            self._direct_comp_col,
            ft.Divider(height=1) if is_mob else ft.VerticalDivider(width=1),
            self._dither_comp_col,
        ]

        side_h = 280 if is_mob else 380
        triple_h = 260 if is_mob else 340
        expand_cols = not is_mob

        self._orig_side_img_box.height = side_h
        self._quant_side_img_box.height = side_h
        self._triple_color_img_box.height = triple_h
        self._triple_gray_img_box.height = triple_h
        self._triple_quant_img_box.height = triple_h
        self._direct_comp_img_box.height = side_h
        self._dither_comp_img_box.height = side_h

        for col in (
            self._orig_side_col,
            self._quant_side_col,
            self._triple_color_col,
            self._triple_gray_col,
            self._triple_quant_col,
            self._direct_comp_col,
            self._dither_comp_col,
        ):
            col.expand = expand_cols

    # -----------------------------------------------------------------------
    # Atualização do Preview Imediato da Imagem de Entrada
    # -----------------------------------------------------------------------

    def _update_input_preview(self, source_title: str, array: np.ndarray, is_sample: bool) -> None:
        """Atualiza o card de preview com thumbnail e metadados da imagem selecionada."""
        h, w = array.shape[:2]
        is_color = bool(array.ndim == 3 and array.shape[2] >= 3)

        thumb_bytes = make_thumbnail_png(array, max_size=MAX_THUMBNAIL_DIMENSION)
        type_str = f"Colorida RGB ({array.shape[2]} canais)" if is_color else "Monocromática (1 canal)"

        self._input_image_bytes = _ndarray_to_png_bytes(array)
        self._input_thumbnail.src = _bytes_to_data_uri(thumb_bytes)
        self._input_name_text.value = source_title
        self._input_dim_badge.content.controls[1].value = f"{w}×{h} px"
        self._input_type_badge.content.controls[1].value = type_str
        self._input_orig_badge.content.controls[1].value = "Amostra PDI" if is_sample else "Arquivo Local"
        self._input_preview_card.visible = True

    # -----------------------------------------------------------------------
    # Handlers de Eventos
    # -----------------------------------------------------------------------

    def _on_gray_category_changed(self, event: ft.ControlEvent) -> None:
        selected = event.control.selected
        if not selected:
            return
        category = next(iter(selected))
        if category == "weighted":
            self._gray_options_selector.segments = [
                ft.Segment(value=str(GrayscaleMethod.LUMINANCE.value), label=ft.Text("Luminância ITU-R BT.601"), icon=ft.Icon(ft.Icons.VISIBILITY)),
                ft.Segment(value=str(GrayscaleMethod.AVERAGE.value), label=ft.Text("Média Aritmética"), icon=ft.Icon(ft.Icons.CALCULATE)),
            ]
            self._selected_gray_method = GrayscaleMethod.LUMINANCE
            self._gray_options_selector.selected = [str(GrayscaleMethod.LUMINANCE.value)]
        else:
            self._gray_options_selector.segments = [
                ft.Segment(value=str(GrayscaleMethod.CHANNEL_R.value), label=ft.Text("Canal R"), icon=ft.Icon(ft.Icons.LOOKS_ONE)),
                ft.Segment(value=str(GrayscaleMethod.CHANNEL_G.value), label=ft.Text("Canal G"), icon=ft.Icon(ft.Icons.LOOKS_TWO)),
                ft.Segment(value=str(GrayscaleMethod.CHANNEL_B.value), label=ft.Text("Canal B"), icon=ft.Icon(ft.Icons.LOOKS_3)),
            ]
            self._selected_gray_method = GrayscaleMethod.CHANNEL_R
            self._gray_options_selector.selected = [str(GrayscaleMethod.CHANNEL_R.value)]

        self._update_gray_info_box()
        self._page.update()

    def _on_gray_method_segmented_changed(self, event: ft.ControlEvent) -> None:
        selected = event.control.selected
        if not selected:
            return
        val_int = int(next(iter(selected)))
        for method in GrayscaleMethod:
            if method.value == val_int:
                self._selected_gray_method = method
                break
        self._update_gray_info_box()
        self._page.update()

    def _update_gray_info_box(self) -> None:
        details = _GRAYSCALE_DETAILS.get(self._selected_gray_method, {})
        if details:
            self._gray_info_title.value = details["title"]
            self._gray_info_formula.value = f"Fórmula: {details['formula']}"
            self._gray_info_desc.value = details["desc"]

        if is_channel_isolation(self._selected_gray_method):
            ch_name = get_channel_color_name(self._selected_gray_method)
            self._btn_convert_gray_only.content = f"Salvar Canal {ch_name} Isolado (8 bits)"
        else:
            self._btn_convert_gray_only.content = "Converter & Salvar em Tons de Cinza (8 bits)"

    def _on_select_sample(self, sample_name: str, display_title: str) -> None:
        """Carrega uma das imagens de teste embutidas com downscaling preventivo."""
        try:
            self._source_path = get_sample_path(sample_name)
            self._loaded_array = load_sample_array(sample_name, max_dim=MAX_IMAGE_DIMENSION)

            h, w = self._loaded_array.shape[:2]
            self._path_label.value = f"📦 {display_title} ({w}×{h})"
            self._path_label.italic = False
            self._path_label.color = ft.Colors.ON_SURFACE
            self._btn_process.disabled = False
            self._btn_convert_gray_only.disabled = False

            self._update_input_preview(display_title, self._loaded_array, is_sample=True)
            self._reset_view_state()
            self._page.update()
            self._show_message(f"'{display_title}' carregada! Clique em 'Quantizar Imagem'.", theme.PRIMARY)
        except Exception as exc:
            self._show_message(f"Erro ao carregar imagem de exemplo: {exc}", theme.ACCENT)

    async def _on_select_image(self, _: ft.ControlEvent) -> None:
        """Abre o FilePicker para seleção de imagem e aplica downscaling preventivo."""
        try:
            files = await self._file_picker.pick_files(
                dialog_title="Selecionar Imagem",
                allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                allow_multiple=False,
                with_data=True,
            )
        except Exception as exc:
            self._show_message(f"Erro ao abrir seletor: {exc}", theme.ACCENT)
            return

        if not files or len(files) == 0:
            return

        file_obj = files[0]
        file_name = getattr(file_obj, "name", "imagem.png") or "imagem.png"
        file_path = getattr(file_obj, "path", None)
        file_bytes = getattr(file_obj, "bytes", None)

        try:
            source = file_bytes if file_bytes is not None else file_path
            if source is None:
                return
            img_arr = open_and_downscale_image(source, max_dim=MAX_IMAGE_DIMENSION)
            self._source_path = Path(file_path) if file_path else None
        except Exception as exc:
            self._show_message(f"Erro ao decodificar imagem: {exc}", theme.ACCENT)
            return

        self._loaded_array = img_arr
        h, w = img_arr.shape[:2]
        self._path_label.value = f"{self._source_path.name} ({w}×{h} px)" if self._source_path else f"🌐 Arquivo Web: {file_name} ({w}×{h} px)"
        self._path_label.italic = False
        self._path_label.color = ft.Colors.ON_SURFACE
        self._btn_process.disabled = False
        self._btn_convert_gray_only.disabled = False

        self._reset_view_state()
        self._update_input_preview(file_name, img_arr, is_sample=False)
        self._page.update()
        self._show_message(f"'{file_name}' carregada ({w}×{h} px)! Clique em 'Quantizar Imagem'.", theme.PRIMARY)

    def _reset_view_state(self) -> None:
        """Reseta o estado visual das áreas de resultado e métricas ao trocar de imagem (DRY)."""
        self._cleanup_previous_run()
        self._btn_save.disabled = True
        self._btn_download_all_zip.disabled = True
        self._btn_quick_zip.visible = False
        self._execution_summary_card.visible = False
        self._btn_inspect.disabled = True
        self._view_mode_buttons.visible = False
        self._native_graph_view.visible = False
        self._single_display_container.visible = False
        self._side_by_side_container.visible = False
        self._triple_container.visible = False
        self._dither_comp_container.visible = False
        self._zoom_toolbar.visible = False
        self._result_placeholder.visible = True
        self._reset_metrics()

    def _on_convert_grayscale_toggled(self, event: ft.ControlEvent) -> None:
        """Alterna entre modo Grayscale (1 canal) e modo Colorido Direto (3 canais RGB)."""
        selected = getattr(event.control, "selected", None)
        if not selected:
            return
        mode_str = next(iter(selected))
        self._convert_to_gray = (mode_str == "yes")
        self._gray_section_container.visible = self._convert_to_gray
        self._btn_convert_gray_only.visible = self._convert_to_gray
        self._update_bits_label()
        self._update_view_mode_buttons()
        self._page.update()

    def _update_bits_label(self) -> None:
        """Atualiza dinamicamente o texto explicativo do slider conforme o modo e a técnica."""
        if self._convert_to_gray:
            n_tons = 2 ** self._bits_value
            self._bits_label.value = f"{self._bits_value} bits  —  {n_tons} tons de cinza"
            self._bits_slider_hint.value = "1=2 tons · 2=4 tons · 3=8 tons · 4=16 tons · 5=32 tons · 6=64 tons · 7=128 tons · 8=256 tons"
        else:
            if self._selected_technique_key == QuantizationTechnique.KMEANS:
                n_cores = 2 ** self._bits_value
                self._bits_label.value = f"Paleta de {n_cores} cores  —  K-Means 3D ({self._bits_value} bits)"
                self._bits_slider_hint.value = f"Quantização vetorial 3D agrupando em {n_cores} centróides de cores RGB"
            else:
                total_cores = (2 ** self._bits_value) ** 3
                self._bits_label.value = f"{self._bits_value} bits/canal  —  {total_cores} cores no total ((2^{self._bits_value})^3)"
                self._bits_slider_hint.value = f"Mapeamento escalar por canal: R, G, B em {2**self._bits_value} níveis cada -> {total_cores} cores"

    def _on_technique_changed(self, event: ft.ControlEvent) -> None:
        """Atualiza a técnica de quantização selecionada com lazy loading se for K-Means."""
        value = getattr(event.control, "value", None) or self._technique_dropdown.value
        if value is None:
            return
        if str(value) == "BOTH":
            self._selected_technique_key = "BOTH"
            self._check_and_notify_kmeans_lazy()
            self._update_bits_label()
            self._update_view_mode_buttons()
            self._page.update()
            return

        for technique, _ in _TECHNIQUE_OPTIONS:
            if isinstance(technique, QuantizationTechnique) and str(technique.value) == str(value):
                self._selected_technique_key = technique
                if technique == QuantizationTechnique.KMEANS:
                    self._check_and_notify_kmeans_lazy()
                break
        self._update_bits_label()
        self._update_view_mode_buttons()
        self._page.update()

    def _check_and_notify_kmeans_lazy(self) -> None:
        """Garante a inicialização do módulo K-Means de forma silenciosa e não-bloqueante."""
        if not is_kmeans_loaded():
            get_kmeans_class()

    def _on_bits_changed(self, event: ft.ControlEvent) -> None:
        self._bits_value = int(event.control.value)
        self._update_bits_label()
        self._update_view_mode_buttons()
        self._page.update()

    def _on_view_mode_changed(self, event: ft.ControlEvent) -> None:
        selected_set = getattr(event.control, "selected", None)
        if selected_set:
            self._active_view_mode = next(iter(selected_set))
        if self._active_view_mode == "dither_comp" and self._direct_metrics is not None and self._dither_metrics is not None:
            self._update_metrics_dither_comparison(self._direct_metrics, self._dither_metrics, 0.0)
        elif self._selected_technique_key == "BOTH":
            if self._active_view_mode == "side_unif_kmeans" and self._direct_metrics and self._kmeans_metrics:
                self._update_metrics_comparison(self._direct_metrics, self._kmeans_metrics, 0.0)
            elif self._active_view_mode == "orig_uniform" and self._direct_metrics:
                m_label = method_label(self._selected_gray_method) if self._convert_to_gray else "RGB"
                self._update_metrics(self._direct_metrics, 0.0, "Uniforme", m_label)
            elif self._active_view_mode == "orig_kmeans" and self._kmeans_metrics:
                m_label = method_label(self._selected_gray_method) if self._convert_to_gray else "RGB"
                self._update_metrics(self._kmeans_metrics, 0.0, "K-Means", m_label)
        self._render_active_view_mode()

    def _on_enhancement_toggled(self, event: ft.ControlEvent) -> None:
        """Alterna a aplicação do aprimoramento de dithering pós-quantização."""
        self._enhancement_enabled = bool(getattr(event.control, "value", False))
        if self._direct_quantized_image is not None and self._dither_image is not None:
            active_img = self._dither_image if self._enhancement_enabled else self._direct_quantized_image
            self._quantized_image = active_img
            self._quantized_image_bytes = _ndarray_to_png_bytes(active_img)

            t_name = technique_label(self._selected_technique_key) if isinstance(self._selected_technique_key, QuantizationTechnique) else str(self._selected_technique_key)
            if self._enhancement_enabled and self._selected_technique_key != QuantizationTechnique.FLOYD_STEINBERG:
                t_name = f"{t_name} + Dithering"
            m_label = method_label(self._selected_gray_method) if self._convert_to_gray else "Modo RGB"
            metrics = self._dither_metrics if self._enhancement_enabled else self._direct_metrics
            if metrics is not None:
                self._update_metrics(metrics, 0.0, t_name, m_label)

            self._render_active_view_mode()

    def _on_process(self, _: ft.ControlEvent) -> None:
        """Inicia o processamento assíncrono não-bloqueante."""
        if self._source_path is None and self._loaded_array is None:
            self._show_message("Selecione ou carregue uma imagem primeiro.", theme.WARNING)
            return

        self._set_processing_state(True)
        if hasattr(self._page, "run_thread"):
            self._page.run_thread(self._run_processing)
        else:
            threading.Thread(target=self._run_processing, daemon=True).start()

    # -----------------------------------------------------------------------
    # Lógica de Processamento Assíncrono Modularizada
    # -----------------------------------------------------------------------

    def _run_processing(self) -> None:
        """Executa o pipeline completo de quantização em background isolado."""
        import time
        start_time = time.perf_counter()

        try:
            # 1. Carregamento e preparação da imagem de entrada
            image_array = (
                self._loaded_array.copy()
                if self._loaded_array is not None
                else open_and_downscale_image(self._source_path, max_dim=MAX_IMAGE_DIMENSION)
            )
            self._raw_image = image_array
            self._is_color = bool(image_array.ndim == 3 and image_array.shape[2] >= 3)
            self._color_image_bytes = _ndarray_to_png_bytes(image_array)

            if self._convert_to_gray:
                # Fluxo 1: Conversão para Tons de Cinza (1 canal)
                gray = to_grayscale(image_array, method=self._selected_gray_method)
                display_source, hist_color = self._prepare_display_source(image_array, gray)
                self._gray_image = display_source
                self._gray_image_bytes = _ndarray_to_png_bytes(display_source)
                target_input = gray
                display_input = display_source
            else:
                # Fluxo 2: Preservação dos 3 canais (RGB)
                rgb_input = image_array[:, :, :3] if self._is_color else image_array
                self._gray_image = rgb_input
                self._gray_image_bytes = _ndarray_to_png_bytes(rgb_input)
                target_input = rgb_input
                display_input = rgb_input

            # 3. Execução da Quantização (Comparação ou Técnica Única)
            if self._selected_technique_key == "BOTH":
                self._execute_quantization_comparison(target_input, start_time)
            else:
                self._execute_quantization_single(target_input, start_time)

            # 4. Geração em segundo plano da versão quantizada em escala de cinza para a Grade Tripla
            if self._convert_to_gray:
                gray_quant = self._quantized_image
            else:
                gray_source = to_grayscale(image_array, method=GrayscaleMethod.LUMINANCE)
                if self._selected_technique_key == "BOTH":
                    gray_quant = quantize(gray_source, bits=self._bits_value, technique=QuantizationTechnique.KMEANS)
                elif self._enhancement_enabled and self._selected_technique_key != QuantizationTechnique.FLOYD_STEINBERG:
                    gray_quant = quantizacao_dithering_floyd_steinberg(gray_source, self._bits_value)
                else:
                    gray_quant = quantize(gray_source, bits=self._bits_value, technique=self._selected_technique_key)
            self._gray_quantized_image = gray_quant
            self._gray_quantized_bytes = _ndarray_to_png_bytes(gray_quant)

            # 5. Geração em segundo plano das figuras analíticas consolidadas
            m_label = method_label(self._selected_gray_method) if self._convert_to_gray else "Modo RGB"
            if self._selected_technique_key == "BOTH":
                self._figure_bytes = generate_full_comparison_figure(
                    original=display_input,
                    uniform=self._direct_quantized_image,
                    kmeans=self._quantized_image,
                    bits=self._bits_value,
                    gray_method_name=m_label,
                )
            else:
                t_name = (
                    technique_label(self._selected_technique_key)
                    if isinstance(self._selected_technique_key, QuantizationTechnique)
                    else str(self._selected_technique_key)
                )
                self._figure_bytes = generate_comparison_figure(
                    original=display_input,
                    quantized=self._quantized_image,
                    bits=self._bits_value,
                    technique_name=t_name,
                    gray_method_name=m_label,
                )

            if self._is_color:
                t_name_color = "K-Means" if self._selected_technique_key == "BOTH" else (
                    technique_label(self._selected_technique_key)
                    if isinstance(self._selected_technique_key, QuantizationTechnique)
                    else str(self._selected_technique_key)
                )
                self._color_figure_bytes = generate_color_comparison_figure(
                    color_image=self._raw_image,
                    quantized=self._quantized_image,
                    bits=self._bits_value,
                    technique_name=t_name_color,
                    gray_image=display_input if self._convert_to_gray else None,
                    gray_method_name=m_label if self._convert_to_gray else None,
                )

            if self._direct_quantized_image is not None and self._dither_image is not None:
                self._dither_figure_bytes = generate_dither_comparison_figure(
                    original_gray=display_input,
                    direct_quantized=self._direct_quantized_image,
                    dither_quantized=self._dither_image,
                    bits=self._bits_value,
                    gray_method_name=m_label,
                    mse_direct=self._direct_metrics.mse if self._direct_metrics else None,
                    psnr_direct=self._direct_metrics.psnr if self._direct_metrics else None,
                    mse_dither=self._dither_metrics.mse if self._dither_metrics else None,
                    psnr_dither=self._dither_metrics.psnr if self._dither_metrics else None,
                )
            else:
                self._dither_figure_bytes = None

            self._update_execution_summary()
            self._update_view_mode_buttons()
            self._view_mode_buttons.visible = True
            self._render_active_view_mode()

        except Exception as error:
            self._show_message(f"Erro no processamento: {error}", theme.ACCENT)
        finally:
            gc.collect()
            self._set_processing_state(False)

    def _prepare_display_source(self, raw_img: np.ndarray, gray_img: np.ndarray) -> tuple[np.ndarray, str]:
        """Prepara a matriz de visualização da imagem de entrada e a cor temático do canal (DRY)."""
        if is_channel_isolation(self._selected_gray_method):
            display = isolate_channel_rgb(raw_img, self._selected_gray_method)
            color = get_channel_color_hex(self._selected_gray_method)
        else:
            display = gray_img
            color = "#4a90d9"
        return display, color

    def _execute_quantization_comparison(self, target_input: np.ndarray, start_time: float) -> None:
        """Executa o modo de comparação completa entre Uniforme e K-Means."""
        import time
        uniform = quantize(target_input, bits=self._bits_value, technique=QuantizationTechnique.UNIFORM)
        kmeans = quantize(target_input, bits=self._bits_value, technique=QuantizationTechnique.KMEANS)

        needs_dither = self._enhancement_enabled or (self._active_view_mode == "dither_comp")
        dither_unif = (
            quantizacao_dithering_floyd_steinberg(target_input, self._bits_value)
            if needs_dither
            else None
        )

        if self._convert_to_gray and is_channel_isolation(self._selected_gray_method):
            display_kmeans = colorize_channel(kmeans, self._selected_gray_method)
            display_uniform = colorize_channel(uniform, self._selected_gray_method)
            display_dither = colorize_channel(dither_unif, self._selected_gray_method) if dither_unif is not None else None
        else:
            display_kmeans = kmeans
            display_uniform = uniform
            display_dither = dither_unif

        self._direct_quantized_image = display_uniform
        self._direct_quantized_bytes = _ndarray_to_png_bytes(display_uniform)
        self._dither_image = display_dither
        self._dither_image_bytes = _ndarray_to_png_bytes(display_dither) if display_dither is not None else None

        self._quantized_image = display_kmeans
        self._quantized_image_bytes = _ndarray_to_png_bytes(display_kmeans)

        elapsed = time.perf_counter() - start_time
        m_unif = calculate_metrics(target_input, uniform, self._bits_value)
        m_km = calculate_metrics(target_input, kmeans, self._bits_value)
        m_dit = calculate_metrics(target_input, dither_unif, self._bits_value) if dither_unif is not None else None

        self._direct_metrics = m_unif
        self._kmeans_metrics = m_km
        self._dither_metrics = m_dit

        self._update_metrics_comparison(m_unif, m_km, elapsed)

    def _execute_quantization_single(self, target_input: np.ndarray, start_time: float) -> None:
        """Executa quantização direta e aprimorada por Dithering com cálculo comparativo."""
        import time
        technique = self._selected_technique_key

        # 1. Quantização direta da técnica selecionada
        direct_quantized = quantize(target_input, bits=self._bits_value, technique=technique)

        # 2. Quantização com difusão de erro (Floyd-Steinberg) calculada sob demanda
        needs_dither = self._enhancement_enabled or (technique == QuantizationTechnique.FLOYD_STEINBERG) or (self._active_view_mode == "dither_comp")
        if needs_dither:
            dither_quantized = (
                direct_quantized
                if technique == QuantizationTechnique.FLOYD_STEINBERG
                else quantizacao_dithering_floyd_steinberg(target_input, self._bits_value)
            )
        else:
            dither_quantized = None

        if self._convert_to_gray and is_channel_isolation(self._selected_gray_method):
            display_direct = colorize_channel(direct_quantized, self._selected_gray_method)
            display_dither = colorize_channel(dither_quantized, self._selected_gray_method) if dither_quantized is not None else None
        else:
            display_direct = direct_quantized
            display_dither = dither_quantized

        self._direct_quantized_image = display_direct
        self._direct_quantized_bytes = _ndarray_to_png_bytes(display_direct)
        self._dither_image = display_dither
        self._dither_image_bytes = _ndarray_to_png_bytes(display_dither) if display_dither is not None else None

        # Determina qual imagem será exibida como ativa principal
        use_dither = self._enhancement_enabled or (technique == QuantizationTechnique.FLOYD_STEINBERG)
        active_img = display_dither if (use_dither and display_dither is not None) else display_direct
        self._quantized_image = active_img
        self._quantized_image_bytes = _ndarray_to_png_bytes(active_img)

        t_name = technique_label(technique)
        if use_dither and technique != QuantizationTechnique.FLOYD_STEINBERG:
            t_name = f"{t_name} + Floyd-Steinberg"
        m_label = method_label(self._selected_gray_method) if self._convert_to_gray else "Colorido (RGB)"
        elapsed = time.perf_counter() - start_time

        m_direct = calculate_metrics(target_input, direct_quantized, self._bits_value)
        self._direct_metrics = m_direct
        if dither_quantized is not None:
            m_dither = calculate_metrics(target_input, dither_quantized, self._bits_value)
            self._dither_metrics = m_dither
        else:
            m_dither = None
            self._dither_metrics = None

        metrics = m_dither if (use_dither and m_dither is not None) else m_direct
        self._update_metrics(metrics, elapsed, t_name, m_label)

        if self._active_view_mode == "dither_comp" and m_direct is not None and m_dither is not None:
            self._update_metrics_dither_comparison(m_direct, m_dither, elapsed)

    async def _on_save(self, _: ft.ControlEvent) -> None:
        """Salva a imagem quantizada ativa selecionada."""
        stem = self._image_stem
        data_to_save, default_name = self._get_active_single_bytes_and_name()
        if data_to_save is None:
            data_to_save = self._quantized_image_bytes or self._gray_image_bytes
            default_name = f"{stem}_quantizada_{self._bits_value}bits.png"

        try:
            save_path = await self._save_picker.save_file(
                dialog_title="Salvar Imagem Quantizada",
                file_name=default_name,
                allowed_extensions=["png"],
                src_bytes=data_to_save,
            )
            if save_path and not getattr(self._page, "web", False):
                Path(save_path).write_bytes(data_to_save)
            self._show_message(f"✅ Arquivo '{default_name}' salvo com sucesso!", theme.SUCCESS)
        except Exception as exc:
            self._show_message(f"Erro ao salvar arquivo: {exc}", theme.ACCENT)

    def _on_convert_and_save_gray(self, _: ft.ControlEvent) -> None:
        """Converte a imagem diretamente para tons de cinza ou canal isolado (8 bits) e salva sem quantização."""
        if self._source_path is None and self._loaded_array is None:
            return

        async def _save_gray_task():
            await self._execute_convert_and_save_gray()

        if hasattr(self._page, "run_task"):
            self._page.run_task(_save_gray_task)
        else:
            task = asyncio.create_task(_save_gray_task())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _execute_convert_and_save_gray(self) -> None:
        """Rotina assíncrona para geração e salvamento da imagem pré-processada em 8 bits."""
        self._set_processing_state(True)
        self._progress_label.value = "Convertendo imagem para 8 bits..."
        self._page.update()

        try:
            image_array = (
                self._loaded_array.copy()
                if self._loaded_array is not None
                else open_and_downscale_image(self._source_path, max_dim=MAX_IMAGE_DIMENSION)
            )
            self._raw_image = image_array
            self._is_color = bool(image_array.ndim == 3 and image_array.shape[2] >= 3)
            gray = to_grayscale(image_array, method=self._selected_gray_method)

            stem = self._image_stem
            if is_channel_isolation(self._selected_gray_method):
                display_gray = isolate_channel_rgb(image_array, self._selected_gray_method)
                ch_name = get_channel_color_name(self._selected_gray_method).lower()
                default_name = f"{stem}_canal_{ch_name}_8bits.png"
                dialog_title = f"Salvar Canal {ch_name.capitalize()} Isolado"
                success_msg = f"Canal {ch_name.capitalize()} isolado salvo com sucesso!"
            else:
                display_gray = gray
                method_str = self._selected_gray_method.name.lower()
                default_name = f"{stem}_cinza_{method_str}_8bits.png"
                dialog_title = "Salvar Imagem em Tons de Cinza"
                success_msg = "Imagem em tons de cinza salva com sucesso!"

            self._gray_image = display_gray
            self._gray_image_bytes = _ndarray_to_png_bytes(display_gray)
            self._quantized_image = display_gray
            self._quantized_image_bytes = self._gray_image_bytes

            self._render_image_view_mode(self._gray_image_bytes)

            try:
                save_path = await self._save_picker.save_file(
                    dialog_title=dialog_title,
                    file_name=default_name,
                    allowed_extensions=["png"],
                    src_bytes=self._gray_image_bytes,
                )
                if save_path and not getattr(self._page, "web", False):
                    Path(save_path).write_bytes(self._gray_image_bytes)
                self._show_message(success_msg, theme.SUCCESS)
            except Exception as exc:
                self._show_message(f"Erro ao salvar arquivo: {exc}", theme.ACCENT)
        except Exception as exc:
            self._show_message(f"Erro ao converter canal/tons de cinza: {exc}", theme.ACCENT)

    def _render_image_view_mode(self, image_bytes: bytes) -> None:
        """Configura a interface diretamente para o modo 'Apenas Imagem'."""
        self._result_placeholder.visible = False
        self._single_display_image.src = _bytes_to_data_uri(image_bytes)
        self._single_display_container.visible = True
        self._native_graph_view.visible = False
        self._side_by_side_container.visible = False
        self._triple_container.visible = False
        self._dither_comp_container.visible = False
        self._zoom_toolbar.visible = True
        self._view_mode_buttons.visible = True
        self._view_mode_buttons.selected = ["image"]
        self._active_view_mode = "image"

    # -----------------------------------------------------------------------
    # Renderização Modular dos Modos de Visualização
    # -----------------------------------------------------------------------

    def _render_active_view_mode(self) -> None:
        """Renderiza a visualização atual com base no modo ativo usando dispatchers limpos."""
        self._result_placeholder.visible = False
        self._btn_save.disabled = False
        self._zoom_toolbar.visible = True

        # Oculta todos os containers antes de ativar o selecionado
        self._native_graph_view.visible = False
        self._single_display_container.visible = False
        self._side_by_side_container.visible = False
        self._triple_container.visible = False
        self._dither_comp_container.visible = False

        t_name = (
            "K-Means (Comparação)"
            if self._selected_technique_key == "BOTH"
            else (technique_label(self._selected_technique_key) if isinstance(self._selected_technique_key, QuantizationTechnique) else str(self._selected_technique_key))
        )
        if self._enhancement_enabled and self._selected_technique_key != QuantizationTechnique.FLOYD_STEINBERG:
            t_name = f"{t_name} + Floyd-Steinberg"
        m_name = method_label(self._selected_gray_method)

        mode_map = {
            "graph": lambda: self._display_graph_mode(t_name, m_name),
            "color_graph": lambda: self._display_color_graph_mode(t_name),
            "image": self._display_image_mode,
            "color_side": lambda: self._display_color_side_mode(t_name),
            "triple": lambda: self._display_triple_mode(t_name, m_name),
            "dither_comp": lambda: self._display_dither_comp_mode(t_name, m_name),
            "side_unif_kmeans": self._display_side_unif_kmeans,
            "orig_uniform": self._display_orig_uniform,
            "orig_kmeans": self._display_orig_kmeans,
        }

        render_fn = mode_map.get(self._active_view_mode)
        if render_fn:
            render_fn()

        self._page.update()

    def _display_graph_mode(self, t_name: str, m_name: str) -> None:
        """Renderiza a Figura Analítica Consolidada (Imagens + Histogramas Integrados)."""
        if not self._figure_bytes:
            self._display_image_mode()
            return
        self._graph_display_image.src = _bytes_to_data_uri(self._figure_bytes)
        self._graph_container.visible = True

    def _display_color_graph_mode(self, t_name: str) -> None:
        """Renderiza a Figura Analítica Consolidada com destaque Colorido (RGB)."""
        fig_bytes = self._color_figure_bytes or self._figure_bytes
        if not fig_bytes:
            self._display_image_mode()
            return
        self._graph_display_image.src = _bytes_to_data_uri(fig_bytes)
        self._graph_container.visible = True

    def _display_image_mode(self) -> None:
        """Renderiza o modo de imagem pura em alta definição com switcher de algoritmo se BOTH."""
        self._single_algo_switcher.visible = bool(self._selected_technique_key == "BOTH")
        data, _ = self._get_active_single_bytes_and_name()
        if not data:
            data = self._quantized_image_bytes
        if data:
            self._single_display_image.src = _bytes_to_data_uri(data)
            self._single_display_container.visible = True

    def _display_side_unif_kmeans(self) -> None:
        """Renderiza comparação direta Uniforme vs K-Means com métricas por card."""
        if not self._direct_quantized_bytes or not self._quantized_image_bytes:
            return
        m_u_str = f" • MSE: {self._direct_metrics.mse:.1f} | PSNR: {self._direct_metrics.psnr:.1f} dB" if self._direct_metrics else ""
        m_k_str = f" • MSE: {self._kmeans_metrics.mse:.1f} | PSNR: {self._kmeans_metrics.psnr:.1f} dB" if self._kmeans_metrics else ""

        self._orig_side_label.value = f"1. Quantização Uniforme ({self._bits_value}b){m_u_str}"
        self._quant_side_label.value = f"2. Quantização K-Means ({self._bits_value}b){m_k_str}"
        self._orig_side_img.src = _bytes_to_data_uri(self._direct_quantized_bytes)
        self._quant_side_img.src = _bytes_to_data_uri(self._quantized_image_bytes)
        self._side_by_side_container.visible = True

    def _display_orig_uniform(self) -> None:
        """Renderiza comparação Original vs Uniforme."""
        orig_bytes = self._color_image_bytes or self._gray_image_bytes or self._input_image_bytes
        if not orig_bytes or not self._direct_quantized_bytes:
            return
        m_u_str = f" • MSE: {self._direct_metrics.mse:.1f} | PSNR: {self._direct_metrics.psnr:.1f} dB" if self._direct_metrics else ""
        self._orig_side_label.value = "1. Imagem Original" if not self._is_color else "1. Original Colorida (RGB)"
        self._quant_side_label.value = f"2. Quantizada Uniforme ({self._bits_value}b){m_u_str}"
        self._orig_side_img.src = _bytes_to_data_uri(orig_bytes)
        self._quant_side_img.src = _bytes_to_data_uri(self._direct_quantized_bytes)
        self._side_by_side_container.visible = True

    def _display_orig_kmeans(self) -> None:
        """Renderiza comparação Original vs K-Means."""
        orig_bytes = self._color_image_bytes or self._gray_image_bytes or self._input_image_bytes
        if not orig_bytes or not self._quantized_image_bytes:
            return
        m_k_str = f" • MSE: {self._kmeans_metrics.mse:.1f} | PSNR: {self._kmeans_metrics.psnr:.1f} dB" if self._kmeans_metrics else ""
        self._orig_side_label.value = "1. Imagem Original" if not self._is_color else "1. Original Colorida (RGB)"
        self._quant_side_label.value = f"2. Quantizada K-Means ({self._bits_value}b){m_k_str}"
        self._orig_side_img.src = _bytes_to_data_uri(orig_bytes)
        self._quant_side_img.src = _bytes_to_data_uri(self._quantized_image_bytes)
        self._side_by_side_container.visible = True

    def _display_color_side_mode(self, t_name: str) -> None:
        """Renderiza comparação lado a lado (Original × Quantizada)."""
        orig_bytes = self._color_image_bytes or self._gray_image_bytes or self._input_image_bytes
        quant_bytes = self._quantized_image_bytes
        if not orig_bytes or not quant_bytes:
            return
        m_info = f" • MSE: {self._direct_metrics.mse:.1f} | PSNR: {self._direct_metrics.psnr:.1f} dB" if self._direct_metrics else ""
        if not self._convert_to_gray:
            self._orig_side_label.value = "1. Imagem Original Colorida (RGB)"
            self._quant_side_label.value = f"2. Quantizada Colorida ({t_name}) — {self._bits_value}b/canal{m_info}"
        else:
            self._orig_side_label.value = "1. Imagem Original" if not self._is_color else "1. Original Colorida (RGB)"
            self._quant_side_label.value = f"2. Quantizada ({t_name}) — {self._bits_value}b{m_info}"
        self._orig_side_img.src = _bytes_to_data_uri(orig_bytes)
        self._quant_side_img.src = _bytes_to_data_uri(quant_bytes)
        self._side_by_side_container.visible = True

    def _display_triple_mode(self, t_name: str, m_name: str) -> None:
        """Renderiza a grade tripla comparativa."""
        if self._selected_technique_key == "BOTH":
            orig_bytes = self._color_image_bytes or self._gray_image_bytes or self._input_image_bytes
            unif_bytes = self._direct_quantized_bytes
            km_bytes = self._quantized_image_bytes
            if not orig_bytes or not unif_bytes or not km_bytes:
                return

            m_u = f" ({self._direct_metrics.psnr:.1f} dB)" if self._direct_metrics else ""
            m_k = f" ({self._kmeans_metrics.psnr:.1f} dB)" if self._kmeans_metrics else ""

            self._triple_color_label.value = "1. Original" if not self._is_color else "1. Original RGB"
            self._triple_gray_label.value = f"2. Uniforme ({self._bits_value}b){m_u}"
            self._triple_quant_label.value = f"3. K-Means ({self._bits_value}b){m_k}"

            self._triple_color_img.src = _bytes_to_data_uri(orig_bytes)
            self._triple_gray_img.src = _bytes_to_data_uri(unif_bytes)
            self._triple_quant_img.src = _bytes_to_data_uri(km_bytes)
            self._triple_container.visible = True
            return

        if not self._convert_to_gray:
            # Modo Colorido: Imagem Original, Quantizada com Cores, Quantizada em Tons de Cinza
            color_bytes = self._color_image_bytes or self._input_image_bytes
            rgb_quant_bytes = self._quantized_image_bytes
            gray_quant_bytes = self._gray_quantized_bytes

            if not color_bytes or not rgb_quant_bytes or not gray_quant_bytes:
                return

            self._triple_color_label.value = "1. Original Colorida (RGB)"
            self._triple_gray_label.value = f"2. Quantizada com Cores ({t_name} • {self._bits_value}b)"
            self._triple_quant_label.value = f"3. Quantizada em Tons de Cinza ({t_name} • {self._bits_value}b)"
            self._triple_color_img.src = _bytes_to_data_uri(color_bytes)
            self._triple_gray_img.src = _bytes_to_data_uri(rgb_quant_bytes)
            self._triple_quant_img.src = _bytes_to_data_uri(gray_quant_bytes)
            self._triple_container.visible = True
        else:
            # Modo Tons de Cinza: Imagem Original, Entrada em Tons de Cinza, Quantizada em Tons de Cinza
            color_bytes = self._color_image_bytes or self._gray_image_bytes
            gray_bytes = self._gray_image_bytes or color_bytes
            quant_bytes = self._quantized_image_bytes or gray_bytes

            if not gray_bytes or not quant_bytes:
                return

            self._triple_color_label.value = "1. Original (RGB)" if self._is_color else "1. Imagem Original"
            self._triple_gray_label.value = f"2. Tons de Cinza ({m_name})"
            self._triple_quant_label.value = f"3. Quantizada ({t_name} • {self._bits_value}b)"
            self._triple_color_img.src = _bytes_to_data_uri(color_bytes)
            self._triple_gray_img.src = _bytes_to_data_uri(gray_bytes)
            self._triple_quant_img.src = _bytes_to_data_uri(quant_bytes)
            self._triple_container.visible = True

    def _display_dither_comp_mode(self, t_name: str, m_name: str) -> None:
        """Renderiza a comparação direta Antes × Depois do Pós-Processamento (Direta × Floyd-Steinberg)."""
        if not self._direct_quantized_bytes or not self._dither_image_bytes:
            return

        m_dir_info = f" • MSE: {self._direct_metrics.mse:.1f} | PSNR: {self._direct_metrics.psnr:.1f} dB" if self._direct_metrics else ""
        m_dit_info = f" • MSE: {self._dither_metrics.mse:.1f} | PSNR: {self._dither_metrics.psnr:.1f} dB" if self._dither_metrics else ""

        self._direct_comp_label.value = f"1. Quantização Direta ({self._bits_value}b){m_dir_info}"
        self._dither_comp_label.value = f"2. Com Floyd-Steinberg ({self._bits_value}b){m_dit_info}"

        self._direct_comp_img.src = _bytes_to_data_uri(self._direct_quantized_bytes)
        self._dither_comp_img.src = _bytes_to_data_uri(self._dither_image_bytes)
        self._dither_comp_container.visible = True

    # -----------------------------------------------------------------------
    # Gestão de Estado da UI e Métricas
    # -----------------------------------------------------------------------

    def _set_processing_state(self, is_processing: bool) -> None:
        """Alterna o estado da UI entre processando e disponível."""
        self._is_processing = is_processing
        self._progress_ring.visible = is_processing
        self._progress_label.value = "Processando quantização de forma assíncrona, aguarde..." if is_processing else ""

        buttons = [
            self._btn_process,
            self._btn_convert_gray_only,
            self._btn_select,
            self._btn_sample_portrait,
            self._btn_sample_benchmark,
            self._btn_sample_lena,
            self._btn_sample_ayla,
            self._btn_sample_pentagono,
            self._bits_slider,
            self._technique_dropdown,
            self._enhancement_switch,
        ]
        for btn in buttons:
            btn.disabled = is_processing

        if not is_processing and (self._quantized_image is not None or self._gray_image is not None):
            self._btn_save.disabled = False
        else:
            self._btn_save.disabled = is_processing

        self.controls[1].visible = is_processing
        self._page.update()

    def _update_metrics(self, metrics, elapsed: float, tech_name: str, method_name: str) -> None:
        """Atualiza os badges de métricas com os valores calculados."""
        self._badge_tech.content.controls[1].value = tech_name
        if self._convert_to_gray:
            self._badge_gray_method.content.controls[0].value = "Método"
            self._badge_gray_method.content.controls[1].value = method_name
        else:
            self._badge_gray_method.content.controls[0].value = "Espaço"
            self._badge_gray_method.content.controls[1].value = "Colorido (RGB)"
        self._badge_mse.content.controls[1].value = f"{metrics.mse:.2f}"
        psnr_str = f"{metrics.psnr:.2f} dB" if metrics.psnr != float("inf") else "∞ dB"
        self._badge_psnr.content.controls[1].value = psnr_str
        self._badge_levels.content.controls[1].value = str(metrics.unique_levels)
        self._badge_time.content.controls[1].value = f"{elapsed:.2f}s" if elapsed > 0 else "—"
        self._comp_metrics_box.visible = False
        self._btn_inspect.disabled = False
        self._page.update()

    def _update_metrics_comparison(self, m_unif, m_km, elapsed: float) -> None:
        """Atualiza os badges de métricas com o comparativo direto entre Uniforme e K-Means."""
        self._badge_tech.content.controls[1].value = "Uniforme × K-Means"
        if self._convert_to_gray:
            self._badge_gray_method.content.controls[0].value = "Método"
            self._badge_gray_method.content.controls[1].value = method_label(self._selected_gray_method)
        else:
            self._badge_gray_method.content.controls[0].value = "Espaço"
            self._badge_gray_method.content.controls[1].value = "Colorido (RGB)"
        self._badge_mse.content.controls[1].value = f"U:{m_unif.mse:.1f} | K:{m_km.mse:.1f}"
        self._badge_psnr.content.controls[1].value = f"U:{m_unif.psnr:.1f} | K:{m_km.psnr:.1f} dB"
        self._badge_levels.content.controls[1].value = f"{2 ** self._bits_value}"
        self._badge_time.content.controls[1].value = f"{elapsed:.2f}s"

        best_tech = "K-Means" if m_km.psnr > m_unif.psnr else "Uniforme"
        gain_psnr = abs(m_km.psnr - m_unif.psnr)
        self._comp_metrics_box.content = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.AUTO_AWESOME, size=18, color=theme.PRIMARY_LIGHT),
                    ft.Text(f"Comparativo: {best_tech} obteve maior fidelidade (+{gain_psnr:.2f} dB de PSNR).", size=theme.FONT_CAPTION, weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=8,
        )
        self._comp_metrics_box.visible = True
        self._btn_inspect.disabled = False
        self._page.update()

    def _update_metrics_dither_comparison(self, m_dir, m_dit, elapsed: float) -> None:
        """Atualiza os badges com comparativo detalhado entre Quantização Direta e Floyd-Steinberg."""
        self._badge_tech.content.controls[1].value = "Direta × Floyd-Steinberg"
        if self._convert_to_gray:
            self._badge_gray_method.content.controls[0].value = "Método"
            self._badge_gray_method.content.controls[1].value = method_label(self._selected_gray_method)
        else:
            self._badge_gray_method.content.controls[0].value = "Espaço"
            self._badge_gray_method.content.controls[1].value = "Colorido (RGB)"
        self._badge_mse.content.controls[1].value = f"Dir:{m_dir.mse:.1f} | Dit:{m_dit.mse:.1f}"
        self._badge_psnr.content.controls[1].value = f"Dir:{m_dir.psnr:.1f} | Dit:{m_dit.psnr:.1f} dB"
        self._badge_levels.content.controls[1].value = f"{m_dir.unique_levels} / {m_dit.unique_levels}"
        self._badge_time.content.controls[1].value = f"{elapsed:.2f}s" if elapsed > 0 else "—"

        self._comp_metrics_box.content = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.AUTO_FIX_HIGH, size=18, color=theme.PRIMARY_LIGHT),
                    ft.Text(
                        "✨ Pós-Processamento PDI: O Dithering redistribui o erro espacialmente, quebrando bandas de falso contorno e preservando gradientes contínuos para o sistema visual.",
                        size=theme.FONT_CAPTION,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.ON_SURFACE,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=8,
        )
        self._comp_metrics_box.visible = True
        self._btn_inspect.disabled = False
        self._page.update()

    def _cleanup_previous_run(self) -> None:
        """Descarta buffers e arrays de execuções anteriores para liberar RAM."""
        self._raw_image = None
        self._gray_image = None
        self._quantized_image = None
        self._quantized_image_bytes = None
        self._gray_image_bytes = None
        self._color_image_bytes = None
        self._gray_quantized_image = None
        self._gray_quantized_bytes = None
        self._figure_bytes = None
        self._color_figure_bytes = None
        self._dither_figure_bytes = None
        self._direct_quantized_image = None
        self._direct_quantized_bytes = None
        self._dither_image = None
        self._dither_image_bytes = None
        self._direct_metrics = None
        self._kmeans_metrics = None
        self._dither_metrics = None
        gc.collect()

    def _reset_metrics(self) -> None:
        """Restaura os badges de métricas para o estado inicial."""
        self._badge_tech.content.controls[1].value = "—"
        self._badge_gray_method.content.controls[1].value = "—"
        self._badge_mse.content.controls[1].value = "—"
        self._badge_psnr.content.controls[1].value = "—"
        self._badge_levels.content.controls[1].value = "—"
        self._badge_time.content.controls[1].value = "—"
        self._comp_metrics_box.visible = False
        self._btn_inspect.disabled = True

    def _show_message(self, message: str, color: str = theme.SUCCESS) -> None:
        """Exibe uma notificação ou SnackBar na tela."""
        snack = ft.SnackBar(ft.Text(message), bgcolor=color)
        if hasattr(self._page, "show_dialog"):
            self._page.show_dialog(snack)
        elif hasattr(self._page, "show_snack_bar"):
            self._page.show_snack_bar(snack)
        self._page.update()

    def _open_inspector_dialog(self) -> None:
        """Abre o modal didático 'Entranhas do Processo' com auditoria e raio-x do pipeline."""
        if self._raw_image is None or self._gray_image is None or self._quantized_image is None:
            self._show_message("Execute o processamento de uma imagem antes de inspecionar.", theme.WARNING)
            return

        gray_for_inspector = (
            self._gray_image
            if self._gray_image.ndim == 2
            else to_grayscale(self._raw_image, self._selected_gray_method)
        )
        open_inspector_dialog(
            page=self._page,
            raw_image=self._raw_image,
            gray_image=gray_for_inspector,
            quantized_image=self._quantized_image,
            bits=self._bits_value,
            technique=self._selected_technique_key,
            method=self._selected_gray_method,
        )

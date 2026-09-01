"""
single_view.py — Aba de Processamento Individual de Imagens.

Permite ao usuário:
  - Selecionar um arquivo de imagem via FilePicker.
  - Escolher o método de conversão para tons de cinza em um menu amplo e organizado.
  - Ajustar o nível de bits (1–8) por um Slider interativo.
  - Selecionar a técnica de quantização (Uniforme, K-Means ou Comparação Completa).
  - Escolher o modo de visualização do resultado:
      * 📊 Gráficos & Histogramas (visão analítica em escala de cinza)
      * 🎨 Gráfico com Cores (comparação analítica completa com histograma RGB)
      * 🖼️ Apenas Imagem Processada (imagem pura em alta definição)
      * 🌓 Lado a Lado: Cinza × Quantizada
      * 🌈 Lado a Lado: Colorida × Quantizada
      * 📑 Grade Tripla (Colorida × Cinza × Quantizada)
  - Checar as métricas MSE e PSNR calculadas.
  - Salvar o resultado no disco no formato correspondente à visualização ativa.
"""

import base64
import io
from pathlib import Path

import flet as ft
import numpy as np
from PIL import Image
from skimage import io as skio

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
    generate_full_comparison_figure,
)
from src.core.quantization import (
    QuantizationTechnique,
    quantize,
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


# ---------------------------------------------------------------------------
# Informações Didáticas dos Métodos de Conversão
# ---------------------------------------------------------------------------

_GRAYSCALE_DETAILS = {
    GrayscaleMethod.LUMINANCE: {
        "title": "Luminância ITU-R BT.601 (Padrão Perceptual)",
        "formula": "Y = 0.2989·R + 0.5870·G + 0.1140·B",
        "desc": "Ponderação perceptual padrão da visão humana (58.7% verde, 29.9% vermelho, 11.4% azul). Gera imagem monocromática.",
        "icon": ft.Icons.VISIBILITY,
    },
    GrayscaleMethod.AVERAGE: {
        "title": "Média Aritmética Simples",
        "formula": "Y = (R + G + B) / 3",
        "desc": "Média uniforme dos três canais RGB sem compensação fisiológica. Gera imagem monocromática.",
        "icon": ft.Icons.CALCULATE,
    },
    GrayscaleMethod.CHANNEL_R: {
        "title": "Isolamento do Canal Vermelho (R)",
        "formula": "Matriz RGB pura: [R, 0, 0] (Tons de Vermelho)",
        "desc": "Isola e exibe exclusivamente o canal vermelho em cores reais; quantização em níveis da cor vermelha.",
        "icon": ft.Icons.LOOKS_ONE,
    },
    GrayscaleMethod.CHANNEL_G: {
        "title": "Isolamento do Canal Verde (G)",
        "formula": "Matriz RGB pura: [0, G, 0] (Tons de Verde)",
        "desc": "Isola e exibe exclusivamente o canal verde em cores reais; quantização em níveis da cor verde.",
        "icon": ft.Icons.LOOKS_TWO,
    },
    GrayscaleMethod.CHANNEL_B: {
        "title": "Isolamento do Canal Azul (B)",
        "formula": "Matriz RGB pura: [0, 0, B] (Tons de Azul)",
        "desc": "Isola e exibe exclusivamente o canal azul em cores reais; quantização em níveis da cor azul.",
        "icon": ft.Icons.LOOKS_3,
    },
}


_TECHNIQUE_OPTIONS = [
    (QuantizationTechnique.UNIFORM, "Modo 1: Quantização Uniforme (Intervalos Iguais)"),
    (QuantizationTechnique.KMEANS, "Modo 2: Quantização Não-Uniforme (K-Means Adaptativo)"),
    (QuantizationTechnique.HISTOGRAM, "Modo 3: Quantização por Histograma (Frequência/Quantis)"),
    ("BOTH", "Modo 4: Comparação Completa (Script Histograma Comparativo 2×3)"),
]


# ---------------------------------------------------------------------------
# Helpers de Compatibilidade e Utilitários de Imagem
# ---------------------------------------------------------------------------


def _register_file_pickers(page: ft.Page, *pickers: ft.FilePicker) -> None:
    """Registra os FilePickers como serviços na página (Flet 0.86+)."""
    if hasattr(page, "services") and hasattr(page.services, "register_service"):
        for picker in pickers:
            page.services.register_service(picker)


def _ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    """Converte um array NumPy uint8 em bytes PNG em memória."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        # Se for RGBA converte
        pil_img = Image.fromarray(arr, mode="RGBA")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil_img = Image.fromarray(arr, mode="RGB")
    else:
        pil_img = Image.fromarray(arr, mode="L")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def _bytes_to_data_uri(image_bytes: bytes) -> str:
    """Converte bytes de imagem PNG em Data URI Base64 para exibição no Flet."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# View Principal
# ---------------------------------------------------------------------------


class SingleView(ft.Column):
    """
    View de processamento individual de imagens com suporte a preview instantâneo,
    alternância de modos de visualização e pop-up modal isolado com zoom interativo.
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

        # Buffers de bytes para exibição, preview e salvamento
        self._input_image_bytes: bytes | None = None
        self._graph_bytes: bytes | None = None
        self._color_graph_bytes: bytes | None = None
        self._quantized_image_bytes: bytes | None = None
        self._gray_image_bytes: bytes | None = None
        self._color_image_bytes: bytes | None = None

        self._selected_technique_key: QuantizationTechnique | str = QuantizationTechnique.UNIFORM
        self._selected_gray_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE
        self._active_view_mode: str = "graph"

        self._build_controls()
        self._assemble_layout()

    # -----------------------------------------------------------------------
    # Construção dos Controles
    # -----------------------------------------------------------------------

    def _build_controls(self) -> None:
        """Inicializa todos os controles da view."""
        box_fit = getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None

        # FilePickers
        self._file_picker = ft.FilePicker()
        self._save_picker = ft.FilePicker()
        _register_file_pickers(self._page, self._file_picker, self._save_picker)

        # Caminho da imagem selecionada
        self._path_label = ft.Text(
            "Nenhuma imagem selecionada",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
            italic=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        # --- Card de Preview Imediato da Imagem de Entrada ---
        self._input_thumbnail = ft.Image(
            src="",
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
            ),
            bgcolor=theme.PRIMARY_LIGHT,
            color="#FFFFFF",
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
                            self._btn_zoom_input,
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

        # --- Menu Organizado de Conversão para Escala de Cinza ---
        self._gray_category_selector = ft.SegmentedButton(
            segments=[
                ft.Segment(
                    value="weighted",
                    label=ft.Text("Ponderação / Média"),
                    icon=ft.Icon(ft.Icons.AUTO_AWESOME),
                ),
                ft.Segment(
                    value="channels",
                    label=ft.Text("Isolamento de Canais (RGB)"),
                    icon=ft.Icon(ft.Icons.PALETTE),
                ),
            ],
            selected=["weighted"],
            on_change=self._on_gray_category_changed,
        )

        self._gray_options_selector = ft.SegmentedButton(
            segments=[
                ft.Segment(
                    value=str(GrayscaleMethod.LUMINANCE.value),
                    label=ft.Text("Luminância ITU-R BT.601"),
                    icon=ft.Icon(ft.Icons.VISIBILITY),
                ),
                ft.Segment(
                    value=str(GrayscaleMethod.AVERAGE.value),
                    label=ft.Text("Média Aritmética"),
                    icon=ft.Icon(ft.Icons.CALCULATE),
                ),
            ],
            selected=[str(GrayscaleMethod.LUMINANCE.value)],
            on_change=self._on_gray_method_segmented_changed,
        )

        # Caixa de informações didáticas sobre o método selecionado
        details = _GRAYSCALE_DETAILS[GrayscaleMethod.LUMINANCE]
        self._gray_info_title = ft.Text(
            details["title"],
            size=theme.FONT_BODY,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.ON_SURFACE,
        )
        self._gray_info_formula = ft.Text(
            f"Fórmula: {details['formula']}",
            size=theme.FONT_CAPTION,
            weight=ft.FontWeight.BOLD,
            color=theme.PRIMARY_LIGHT,
        )
        self._gray_info_desc = ft.Text(
            details["desc"],
            size=theme.FONT_CAPTION,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )

        self._gray_info_box = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.INFO_OUTLINE, size=18, color=theme.PRIMARY_LIGHT),
                            self._gray_info_title,
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self._gray_info_formula,
                    self._gray_info_desc,
                ],
                spacing=4,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=10,
        )

        # Dropdown: técnica de quantização
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

        # Slider de bits
        self._bits_value = 4
        self._bits_label = ft.Text(
            f"{self._bits_value} bits  —  {2 ** self._bits_value} tons de cinza",
            size=theme.FONT_BODY,
            color=ft.Colors.ON_SURFACE,
            weight=ft.FontWeight.BOLD,
        )
        self._bits_slider = ft.Slider(
            min=1,
            max=8,
            divisions=7,
            value=self._bits_value,
            label="{value} bits",
            active_color=theme.PRIMARY,
            thumb_color=theme.PRIMARY_LIGHT,
            on_change=self._on_bits_changed,
        )

        # Badges de métricas
        self._badge_mse = theme.metric_badge("MSE", "—")
        self._badge_psnr = theme.metric_badge("PSNR", "—", color=theme.SUCCESS)
        self._badge_levels = theme.metric_badge("Níveis", "—")
        self._badge_time = theme.metric_badge("Tempo", "—", color=theme.WARNING)

        # Seletor de Modo de Visualização do Resultado
        self._view_mode_buttons = ft.SegmentedButton(
            segments=[
                ft.Segment(
                    value="graph",
                    label=ft.Text("Gráfico Cinza"),
                    icon=ft.Icon(ft.Icons.BAR_CHART),
                ),
                ft.Segment(
                    value="color_graph",
                    label=ft.Text("Gráfico Colorido"),
                    icon=ft.Icon(ft.Icons.INSERT_CHART_OUTLINED),
                ),
                ft.Segment(
                    value="image",
                    label=ft.Text("Apenas Quantizada"),
                    icon=ft.Icon(ft.Icons.IMAGE),
                ),
                ft.Segment(
                    value="side_by_side",
                    label=ft.Text("Cinza × Quantizada"),
                    icon=ft.Icon(ft.Icons.COMPARE),
                ),
                ft.Segment(
                    value="color_side",
                    label=ft.Text("Colorida × Quantizada"),
                    icon=ft.Icon(ft.Icons.COLOR_LENS),
                ),
                ft.Segment(
                    value="triple",
                    label=ft.Text("Grade Tripla"),
                    icon=ft.Icon(ft.Icons.VIEW_COLUMN),
                ),
            ],
            selected=["graph"],
            on_change=self._on_view_mode_changed,
            visible=False,
            show_selected_icon=False,
        )

        # --- Áreas de exibição dos resultados (Sem interferência de scroll na página) ---

        # Área 1: Imagem única / Gráfico com suporte a clique para pop-up modal
        self._single_display_image = ft.Image(
            src="",
            fit=box_fit,
            expand=True,
        )
        self._btn_single_zoom = ft.Button(
            content="🔍 Ampliar em Tela Cheia / Zoom",
            icon=ft.Icons.ZOOM_IN,
            on_click=lambda _: self._open_active_single_zoom(),
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._single_display_container = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        self._btn_single_zoom,
                        ft.Row(
                            controls=[
                                ft.Icon(ft.Icons.TOUCH_APP, size=16, color=theme.PRIMARY_LIGHT),
                                ft.Text(
                                    "Clique na imagem abaixo para abrir o pop-up com zoom",
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
        # Compatibilidade com referências existentes
        self._single_interactive = self._single_display_container

        # Área 2: Comparação lado a lado (Original vs Quantizada) com botões e clique
        self._orig_side_label = ft.Text("Imagem Original", weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT)
        self._orig_side_img = ft.Image(src="", fit=box_fit, expand=True)
        self._quant_side_img = ft.Image(src="", fit=box_fit, expand=True)

        self._btn_orig_side_zoom = ft.OutlinedButton(
            content="🔍 Ampliar",
            on_click=lambda _: self._open_side_orig_zoom(),
        )
        self._btn_quant_side_zoom = ft.OutlinedButton(
            content="🔍 Ampliar",
            on_click=lambda _: self._open_side_quant_zoom(),
        )

        self._side_by_side_container = ft.Row(
            controls=[
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[self._orig_side_label, self._btn_orig_side_zoom],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Container(
                            content=self._orig_side_img,
                            height=380,
                            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                            on_click=lambda _: self._open_side_orig_zoom(),
                            ink=True,
                            tooltip="Clique para ampliar a imagem original",
                            border_radius=8,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                            padding=6,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                ),
                ft.VerticalDivider(width=1),
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("Imagem Quantizada", weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT),
                                self._btn_quant_side_zoom,
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Container(
                            content=self._quant_side_img,
                            height=380,
                            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                            on_click=lambda _: self._open_side_quant_zoom(),
                            ink=True,
                            tooltip="Clique para ampliar a imagem quantizada",
                            border_radius=8,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                            padding=6,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                ),
            ],
            spacing=16,
            expand=True,
            visible=False,
        )

        # Área 3: Grade Tripla (Colorida × Cinza × Quantizada)
        self._triple_color_img = ft.Image(src="", fit=box_fit, expand=True)
        self._triple_gray_img = ft.Image(src="", fit=box_fit, expand=True)
        self._triple_quant_img = ft.Image(src="", fit=box_fit, expand=True)

        self._triple_container = ft.Row(
            controls=[
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("1. Original RGB", weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT, size=theme.FONT_CAPTION),
                                ft.IconButton(
                                    icon=ft.Icons.ZOOM_IN,
                                    icon_size=18,
                                    tooltip="Ampliar Imagem Colorida",
                                    on_click=lambda _: self._open_zoom_dialog("1. Original Colorida (RGB)", self._color_image_bytes),
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Container(
                            content=self._triple_color_img,
                            height=340,
                            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                            on_click=lambda _: self._open_zoom_dialog("1. Original Colorida (RGB)", self._color_image_bytes),
                            ink=True,
                            tooltip="Clique para ampliar a imagem colorida",
                            border_radius=8,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                            padding=4,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                ),
                ft.VerticalDivider(width=1),
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("2. Tons de Cinza", weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION),
                                ft.IconButton(
                                    icon=ft.Icons.ZOOM_IN,
                                    icon_size=18,
                                    tooltip="Ampliar Imagem Tons de Cinza",
                                    on_click=lambda _: self._open_zoom_dialog("2. Tons de Cinza (8 bits)", self._gray_image_bytes),
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Container(
                            content=self._triple_gray_img,
                            height=340,
                            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                            on_click=lambda _: self._open_zoom_dialog("2. Tons de Cinza (8 bits)", self._gray_image_bytes),
                            ink=True,
                            tooltip="Clique para ampliar a imagem em tons de cinza",
                            border_radius=8,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                            padding=4,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                ),
                ft.VerticalDivider(width=1),
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("3. Quantizada", weight=ft.FontWeight.BOLD, color=theme.SUCCESS, size=theme.FONT_CAPTION),
                                ft.IconButton(
                                    icon=ft.Icons.ZOOM_IN,
                                    icon_size=18,
                                    tooltip="Ampliar Imagem Quantizada",
                                    on_click=lambda _: self._open_zoom_dialog(f"3. Quantizada ({self._bits_value} bits)", self._quantized_image_bytes),
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Container(
                            content=self._triple_quant_img,
                            height=340,
                            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                            on_click=lambda _: self._open_zoom_dialog(f"3. Quantizada ({self._bits_value} bits)", self._quantized_image_bytes),
                            ink=True,
                            tooltip="Clique para ampliar a imagem quantizada",
                            border_radius=8,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                            padding=4,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                ),
            ],
            spacing=10,
            expand=True,
            visible=False,
        )

        # Barra de instrução do novo visualizador
        self._zoom_toolbar = ft.Row(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.PHOTO_SIZE_SELECT_ACTUAL, size=16, color=theme.PRIMARY_LIGHT),
                        ft.Text(
                            " Visualizador com Zoom: Clique em qualquer imagem ou gráfico para abrir o pop-up com zoom interativo (até 10×) e pan.",
                            size=theme.FONT_CAPTION,
                            color=ft.Colors.ON_SURFACE_VARIANT,
                        ),
                    ],
                    spacing=6,
                ),
            ],
            visible=False,
        )

        # Placeholder quando nenhuma imagem está carregada
        self._result_placeholder = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(ft.Icons.IMAGE_SEARCH, size=64, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text(
                        "Selecione uma imagem e clique em 'Quantizar' para ver o resultado",
                        color=ft.Colors.ON_SURFACE_VARIANT,
                        size=theme.FONT_BODY,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            height=360,
            expand=True,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
        )

        # Indicador de progresso
        self._progress_ring = ft.ProgressRing(
            visible=False,
            color=theme.PRIMARY,
            width=32,
            height=32,
        )
        self._progress_label = ft.Text("", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION)

        # Botões de ação e seleção de amostras com Tooltips Descritivos
        self._btn_select = ft.Button(
            content="Selecionar do Disco",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_select_image,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_sample_portrait = ft.OutlinedButton(
            content="👤 Retrato RGB",
            tooltip="Retrato RGB (512×512) • Foto colorida com tons de pele, texturas e detalhes faciais.",
            on_click=lambda _: self._on_select_sample(SAMPLE_PORTRAIT_NAME, "Exemplo 1: Retrato RGB"),
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
            tooltip="Ayla Foto HD (512×512) • Texturas finas de pelos e iluminação natural de alta definição.",
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

    def _assemble_layout(self) -> None:
        """Monta o layout da view organizando os controles em cards."""
        self.controls = [
            # Card 1: Seleção de imagem e Configurações
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("⚙️  Configurações & Imagem"),
                        ft.Divider(height=1),
                        ft.Row(
                            controls=[self._btn_select, self._path_label],
                            spacing=12,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            wrap=True,
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    "Imagens de Teste Embutidas (passe o mouse para detalhes):",
                                    color=ft.Colors.ON_SURFACE_VARIANT,
                                    size=theme.FONT_CAPTION,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.Row(
                                    controls=[
                                        self._btn_sample_portrait,
                                        self._btn_sample_benchmark,
                                        self._btn_sample_lena,
                                        self._btn_sample_ayla,
                                        self._btn_sample_pentagono,
                                    ],
                                    spacing=8,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                    wrap=True,
                                ),
                            ],
                            spacing=6,
                        ),
                        # Preview Imediato da Imagem de Entrada
                        self._input_preview_card,
                        ft.Divider(height=1),
                        # Seção de seleção do algoritmo de tons de cinza
                        ft.Text("Método de Conversão para Tons de Cinza:", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE),
                        self._gray_category_selector,
                        self._gray_options_selector,
                        self._gray_info_box,
                        ft.Divider(height=1),
                        ft.Text("Técnica de Quantização:", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE),
                        self._technique_dropdown,
                        ft.Divider(height=1),
                        ft.Column(
                            controls=[
                                ft.Row(
                                    controls=[
                                        ft.Text("Nível de Bits:", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_BODY),
                                        self._bits_label,
                                    ],
                                    spacing=8,
                                ),
                                ft.Row(
                                    controls=[
                                        ft.Text("1 bit", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION),
                                        self._bits_slider,
                                        ft.Text("8 bits", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION),
                                    ],
                                    spacing=4,
                                ),
                                ft.Text(
                                    "1=2 tons · 2=4 tons · 3=8 tons · 4=16 tons · 5=32 tons · 6=64 tons · 7=128 tons · 8=256 tons",
                                    color=ft.Colors.ON_SURFACE_VARIANT,
                                    size=theme.FONT_CAPTION,
                                    weight=ft.FontWeight.NORMAL,
                                ),
                            ],
                            spacing=6,
                        ),
                        ft.Divider(height=1),
                        ft.Row(
                            controls=[
                                self._btn_process,
                                self._btn_convert_gray_only,
                                self._btn_save,
                            ],
                            spacing=12,
                            wrap=True,
                        ),
                    ],
                    spacing=12,
                )
            ),
            # Card 2: Progresso
            ft.Container(
                content=ft.Row(
                    controls=[self._progress_ring, self._progress_label],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                visible=False,
            ),
            # Card 3: Métricas
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("📊  Métricas de Qualidade"),
                        ft.Divider(height=1),
                        ft.Row(
                            controls=[
                                self._badge_mse,
                                self._badge_psnr,
                                self._badge_levels,
                                self._badge_time,
                            ],
                            spacing=12,
                            wrap=True,
                        ),
                        ft.Text(
                            "MSE: erro quadrático médio por pixel (menor = melhor) · "
                            "PSNR: relação sinal-ruído de pico em dB (maior = melhor)",
                            color=ft.Colors.ON_SURFACE_VARIANT,
                            size=theme.FONT_CAPTION,
                        ),
                    ],
                    spacing=10,
                )
            ),
            # Card 4: Área de resultado com seletor de visualização e Zoom Modal
            theme.card(
                ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                theme.section_title("🖼️  Visualização do Resultado"),
                                self._view_mode_buttons,
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            wrap=True,
                        ),
                        self._zoom_toolbar,
                        ft.Divider(height=1),
                        self._result_placeholder,
                        self._single_display_container,
                        self._side_by_side_container,
                        self._triple_container,
                    ],
                    spacing=12,
                )
            ),
        ]

    # -----------------------------------------------------------------------
    # Visualizador de Zoom Modal (AlertDialog Isolado)
    # -----------------------------------------------------------------------

    def _open_zoom_dialog(self, title: str, image_bytes: bytes | str | None) -> None:
        """
        Abre um pop-up modal (AlertDialog) dedicado e isolado para visualização
        de imagem em alta resolução com ferramentas completas de zoom e pan.
        """
        if image_bytes is None:
            return

        data_uri = image_bytes if isinstance(image_bytes, str) else _bytes_to_data_uri(image_bytes)

        scale_val = [1.0]

        zoom_label = ft.Text(
            "100%",
            weight=ft.FontWeight.BOLD,
            size=theme.FONT_BODY,
            color=ft.Colors.ON_SURFACE,
        )

        img_control = ft.Image(
            src=data_uri,
            fit=getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None,
        )

        interactive_viewer = ft.InteractiveViewer(
            content=img_control,
            pan_enabled=True,
            scale_enabled=True,
            min_scale=0.2,
            max_scale=10.0,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            expand=True,
        )

        dialog = ft.AlertDialog(
            modal=True,
            content_padding=12,
            title_padding=ft.Padding.only(left=20, top=16, right=16, bottom=8) if hasattr(ft, "Padding") else 16,
            actions_padding=ft.Padding.only(left=20, right=20, bottom=16) if hasattr(ft, "Padding") else 16,
        )

        def _close_dialog(e: ft.ControlEvent = None) -> None:
            dialog.open = False
            self._page.pop_dialog()
            self._page.update()

        def _update_zoom_ui() -> None:
            zoom_label.value = f"{int(scale_val[0] * 100)}%"
            dialog.update()

        def _on_zoom_in(e: ft.ControlEvent) -> None:
            scale_val[0] = round(min(10.0, scale_val[0] + 0.25), 2)
            img_control.scale = ft.Scale(scale_val[0])
            _update_zoom_ui()

        def _on_zoom_out(e: ft.ControlEvent) -> None:
            scale_val[0] = round(max(0.25, scale_val[0] - 0.25), 2)
            img_control.scale = ft.Scale(scale_val[0])
            _update_zoom_ui()

        def _on_zoom_reset(e: ft.ControlEvent) -> None:
            scale_val[0] = 1.0
            img_control.scale = ft.Scale(1.0)
            _update_zoom_ui()

        dialog.title = ft.Row(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.ZOOM_IN, size=24, color=theme.PRIMARY_LIGHT),
                        ft.Text(
                            title,
                            weight=ft.FontWeight.BOLD,
                            size=theme.FONT_TITLE,
                            color=ft.Colors.ON_SURFACE,
                        ),
                    ],
                    spacing=8,
                ),
                ft.Row(
                    controls=[
                        ft.IconButton(
                            icon=ft.Icons.ZOOM_OUT,
                            tooltip="Diminuir Zoom (-25%)",
                            on_click=_on_zoom_out,
                        ),
                        ft.Container(
                            content=zoom_label,
                            padding=ft.Padding.symmetric(horizontal=6) if hasattr(ft, "Padding") else 6,
                            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                        ),
                        ft.IconButton(
                            icon=ft.Icons.ZOOM_IN,
                            tooltip="Aumentar Zoom (+25%)",
                            on_click=_on_zoom_in,
                        ),
                        ft.IconButton(
                            icon=ft.Icons.RESTART_ALT,
                            tooltip="Resetar Zoom (100%)",
                            on_click=_on_zoom_reset,
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            tooltip="Fechar",
                            on_click=_close_dialog,
                        ),
                    ],
                    spacing=4,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        dialog.content = ft.Container(
            content=interactive_viewer,
            width=960,
            height=580,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        )

        dialog.actions = [
            ft.Row(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.MOUSE, size=16, color=theme.PRIMARY_LIGHT),
                            ft.Text(
                                "💡 Dica: Use a roda do mouse ou os botões de zoom acima. Arraste com o cursor para mover a imagem (Pan).",
                                size=theme.FONT_CAPTION,
                                color=ft.Colors.ON_SURFACE_VARIANT,
                            ),
                        ],
                        spacing=6,
                    ),
                    ft.Button(
                        content="Fechar",
                        icon=ft.Icons.CHECK,
                        on_click=_close_dialog,
                        bgcolor=theme.PRIMARY,
                        color="#FFFFFF",
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                expand=True,
            )
        ]

        self._page.show_dialog(dialog)
        self._page.update()

    def _open_active_single_zoom(self) -> None:
        """Abre o visualizador de zoom para o modo ativo de imagem única / gráfico."""
        if self._active_view_mode == "graph" and self._graph_bytes:
            self._open_zoom_dialog("Gráfico Analítico em Tons de Cinza", self._graph_bytes)
        elif self._active_view_mode == "color_graph" and self._color_graph_bytes:
            self._open_zoom_dialog("Gráfico Analítico Colorido com Histograma RGB", self._color_graph_bytes)
        elif self._active_view_mode == "image" and self._quantized_image_bytes:
            self._open_zoom_dialog(f"Imagem Quantizada ({self._bits_value} bits)", self._quantized_image_bytes)

    def _open_side_orig_zoom(self) -> None:
        """Abre zoom para o lado original no modo lado a lado."""
        if self._active_view_mode == "color_side" and self._color_image_bytes:
            self._open_zoom_dialog("Imagem Original Colorida (RGB)", self._color_image_bytes)
        elif self._gray_image_bytes:
            self._open_zoom_dialog("Imagem Original em Tons de Cinza", self._gray_image_bytes)

    def _open_side_quant_zoom(self) -> None:
        """Abre zoom para o lado quantizado no modo lado a lado."""
        if self._quantized_image_bytes:
            self._open_zoom_dialog(f"Imagem Quantizada ({self._bits_value} bits)", self._quantized_image_bytes)

    # -----------------------------------------------------------------------
    # Atualização do Preview Imediato da Imagem de Entrada
    # -----------------------------------------------------------------------

    def _update_input_preview(self, source_title: str, array: np.ndarray, is_sample: bool) -> None:
        """Atualiza o card de preview com thumbnail e metadados da imagem selecionada."""
        h, w = array.shape[:2]
        is_color = bool(array.ndim == 3 and array.shape[2] >= 3)

        if is_color:
            rgb_disp = array[:, :, :3]
            if rgb_disp.dtype != np.uint8 and np.issubdtype(rgb_disp.dtype, np.floating):
                rgb_disp = (np.clip(rgb_disp, 0.0, 1.0) * 255).astype(np.uint8)
            thumb_bytes = _ndarray_to_png_bytes(rgb_disp)
            type_str = f"Colorida RGB ({array.shape[2]} canais)"
        else:
            thumb_bytes = _ndarray_to_png_bytes(array)
            type_str = "Monocromática (1 canal)"

        self._input_image_bytes = thumb_bytes
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
        """Alterna os botões de opção conforme a categoria de método de cinza selecionada."""
        selected = event.control.selected
        if not selected:
            return
        category = next(iter(selected))
        if category == "weighted":
            self._gray_options_selector.segments = [
                ft.Segment(
                    value=str(GrayscaleMethod.LUMINANCE.value),
                    label=ft.Text("Luminância ITU-R BT.601"),
                    icon=ft.Icon(ft.Icons.VISIBILITY),
                ),
                ft.Segment(
                    value=str(GrayscaleMethod.AVERAGE.value),
                    label=ft.Text("Média Aritmética"),
                    icon=ft.Icon(ft.Icons.CALCULATE),
                ),
            ]
            self._selected_gray_method = GrayscaleMethod.LUMINANCE
            self._gray_options_selector.selected = [str(GrayscaleMethod.LUMINANCE.value)]
        else:
            self._gray_options_selector.segments = [
                ft.Segment(
                    value=str(GrayscaleMethod.CHANNEL_R.value),
                    label=ft.Text("Canal R"),
                    icon=ft.Icon(ft.Icons.LOOKS_ONE),
                ),
                ft.Segment(
                    value=str(GrayscaleMethod.CHANNEL_G.value),
                    label=ft.Text("Canal G"),
                    icon=ft.Icon(ft.Icons.LOOKS_TWO),
                ),
                ft.Segment(
                    value=str(GrayscaleMethod.CHANNEL_B.value),
                    label=ft.Text("Canal B"),
                    icon=ft.Icon(ft.Icons.LOOKS_3),
                ),
            ]
            self._selected_gray_method = GrayscaleMethod.CHANNEL_R
            self._gray_options_selector.selected = [str(GrayscaleMethod.CHANNEL_R.value)]

        self._update_gray_info_box()
        self._page.update()

    def _on_gray_method_segmented_changed(self, event: ft.ControlEvent) -> None:
        """Atualiza o método de escala de cinza quando uma opção específica é clicada."""
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
        """Atualiza a caixa didática com a fórmula e descrição do método ativo."""
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
        """Carrega uma das imagens de teste embutidas no aplicativo."""
        try:
            sample_path = get_sample_path(sample_name)
            self._source_path = sample_path
            self._loaded_array = load_sample_array(sample_name)

            self._path_label.value = f"📦 {display_title} (512×512)"
            self._path_label.italic = False
            self._path_label.color = ft.Colors.ON_SURFACE
            self._btn_process.disabled = False
            self._btn_convert_gray_only.disabled = False

            # Atualiza o preview imediato da imagem de entrada
            self._update_input_preview(display_title, self._loaded_array, is_sample=True)

            # Reseta estado anterior
            self._raw_image = None
            self._gray_image = None
            self._quantized_image = None
            self._graph_bytes = None
            self._color_graph_bytes = None
            self._quantized_image_bytes = None
            self._gray_image_bytes = None
            self._color_image_bytes = None

            self._btn_save.disabled = True
            self._view_mode_buttons.visible = False
            self._single_display_container.visible = False
            self._side_by_side_container.visible = False
            self._triple_container.visible = False
            self._zoom_toolbar.visible = False
            self._result_placeholder.visible = True
            self._reset_metrics()
            self._page.update()
            self._show_message(f"'{display_title}' carregada! Clique em 'Quantizar Imagem' ou no preview para ampliar.", theme.PRIMARY)
        except Exception as exc:
            self._show_message(f"Erro ao carregar imagem de exemplo: {exc}", theme.ACCENT)

    async def _on_select_image(self, _: ft.ControlEvent) -> None:
        """Abre o FilePicker para seleção de imagem."""
        files = await self._file_picker.pick_files(
            dialog_title="Selecionar Imagem",
            allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            allow_multiple=False,
        )
        if files and len(files) > 0:
            file_obj = files[0]
            file_path = getattr(file_obj, "path", None)
            if file_path:
                self._source_path = Path(file_path)
                self._loaded_array = None
                self._path_label.value = str(self._source_path)
                self._path_label.italic = False
                self._path_label.color = ft.Colors.ON_SURFACE
                self._btn_process.disabled = False
                self._btn_convert_gray_only.disabled = False

                # Carrega o array para o preview imediato
                try:
                    img_arr = skio.imread(str(self._source_path))
                    self._update_input_preview(self._source_path.name, img_arr, is_sample=False)
                except Exception:
                    pass

                # Reseta estado anterior
                self._raw_image = None
                self._gray_image = None
                self._quantized_image = None
                self._graph_bytes = None
                self._color_graph_bytes = None
                self._quantized_image_bytes = None
                self._gray_image_bytes = None
                self._color_image_bytes = None

                self._btn_save.disabled = True
                self._view_mode_buttons.visible = False
                self._single_display_container.visible = False
                self._side_by_side_container.visible = False
                self._triple_container.visible = False
                self._zoom_toolbar.visible = False
                self._result_placeholder.visible = True
                self._reset_metrics()
                self._page.update()

    def _on_technique_changed(self, event: ft.ControlEvent) -> None:
        """Atualiza a técnica de quantização selecionada."""
        value = getattr(event.control, "value", None) or self._technique_dropdown.value
        if value is None:
            return
        if str(value) == "BOTH":
            self._selected_technique_key = "BOTH"
        else:
            for technique, _ in _TECHNIQUE_OPTIONS:
                if isinstance(technique, QuantizationTechnique) and str(technique.value) == str(value):
                    self._selected_technique_key = technique
                    break

    def _on_bits_changed(self, event: ft.ControlEvent) -> None:
        """Atualiza o rótulo exibindo o nível de bits e tons atual."""
        self._bits_value = int(event.control.value)
        n_tons = 2 ** self._bits_value
        self._bits_label.value = f"{self._bits_value} bits  —  {n_tons} tons de cinza"
        self._page.update()

    def _on_view_mode_changed(self, event: ft.ControlEvent) -> None:
        """Alterna a exibição conforme o modo selecionado."""
        selected_set = getattr(event.control, "selected", None)
        if selected_set:
            self._active_view_mode = next(iter(selected_set))
        self._render_active_view_mode()

    def _on_process(self, _: ft.ControlEvent) -> None:
        """Inicia o processamento."""
        if self._source_path is None:
            return

        self._set_processing_state(True)
        if hasattr(self._page, "run_thread"):
            self._page.run_thread(self._run_processing)
        else:
            import threading
            threading.Thread(target=self._run_processing, daemon=True).start()

    async def _on_save(self, _: ft.ControlEvent) -> None:
        """Abre o FilePicker para salvar o resultado ativo no disco."""
        if self._source_path is None:
            return

        ch_name = get_channel_color_name(self._selected_gray_method).lower() if is_channel_isolation(self._selected_gray_method) else "cinza"

        # Determina o buffer e sufixo com base no modo de visualização ativo
        if self._active_view_mode == "image" and self._quantized_image_bytes:
            data_to_save = self._quantized_image_bytes
            default_name = f"{self._source_path.stem}_quantizada_{ch_name}_{self._bits_value}bits.png"
        elif self._active_view_mode == "color_graph" and self._color_graph_bytes:
            data_to_save = self._color_graph_bytes
            default_name = f"{self._source_path.stem}_grafico_colorido_{self._bits_value}bits.png"
        else:
            data_to_save = self._graph_bytes
            default_name = f"{self._source_path.stem}_grafico_{ch_name}_{self._bits_value}bits.png"

        if data_to_save is None:
            return

        save_path = await self._save_picker.save_file(
            dialog_title="Salvar Resultado",
            file_name=default_name,
            allowed_extensions=["png"],
        )
        if save_path:
            Path(save_path).write_bytes(data_to_save)
            self._show_message("Arquivo salvo com sucesso!", theme.SUCCESS)

    # -----------------------------------------------------------------------
    # Lógica de Processamento
    # -----------------------------------------------------------------------

    def _run_processing(self) -> None:
        """Executa o pipeline completo de quantização em background."""
        import time
        start_time = time.perf_counter()

        try:
            if self._loaded_array is not None:
                image_array = self._loaded_array.copy()
            else:
                image_array = skio.imread(str(self._source_path))

            self._raw_image = image_array
            self._is_color = bool(image_array.ndim == 3 and image_array.shape[2] >= 3)

            # Prepara imagem colorida original para exibição
            if self._is_color:
                rgb_display = image_array[:, :, :3]
                if rgb_display.dtype != np.uint8 and np.issubdtype(rgb_display.dtype, np.floating):
                    rgb_display = (np.clip(rgb_display, 0.0, 1.0) * 255).astype(np.uint8)
                self._color_image_bytes = _ndarray_to_png_bytes(rgb_display)
            else:
                self._color_image_bytes = _ndarray_to_png_bytes(image_array)

            gray = to_grayscale(image_array, method=self._selected_gray_method)

            # Se for isolamento de canal (R, G ou B), gera representação visual cromática pura
            if is_channel_isolation(self._selected_gray_method):
                display_source = isolate_channel_rgb(image_array, self._selected_gray_method)
                hist_color = get_channel_color_hex(self._selected_gray_method)
            else:
                display_source = gray
                hist_color = "#4a90d9"

            self._gray_image = display_source
            self._gray_image_bytes = _ndarray_to_png_bytes(display_source)

            if self._selected_technique_key == "BOTH":
                uniform = quantize(gray, bits=self._bits_value, technique=QuantizationTechnique.UNIFORM)
                kmeans = quantize(gray, bits=self._bits_value, technique=QuantizationTechnique.KMEANS)

                if is_channel_isolation(self._selected_gray_method):
                    display_uniform = colorize_channel(uniform, self._selected_gray_method)
                    display_kmeans = colorize_channel(kmeans, self._selected_gray_method)
                else:
                    display_uniform = uniform
                    display_kmeans = kmeans

                self._quantized_image = display_kmeans
                self._quantized_image_bytes = _ndarray_to_png_bytes(display_kmeans)
                self._graph_bytes = generate_full_comparison_figure(
                    original=display_source,
                    uniform=display_uniform,
                    kmeans=display_kmeans,
                    bits=self._bits_value,
                    hist_color_unif=hist_color,
                    hist_color_km="#e8624a" if not is_channel_isolation(self._selected_gray_method) else hist_color,
                )
                self._color_graph_bytes = generate_color_comparison_figure(
                    color_image=image_array,
                    quantized=display_kmeans,
                    bits=self._bits_value,
                    technique_name="K-Means (Comparação)",
                    gray_image=display_source,
                )
                elapsed = time.perf_counter() - start_time
                self._update_metrics_comparison(elapsed)
            else:
                technique = self._selected_technique_key
                quantized = quantize(gray, bits=self._bits_value, technique=technique)

                if is_channel_isolation(self._selected_gray_method):
                    display_quantized = colorize_channel(quantized, self._selected_gray_method)
                else:
                    display_quantized = quantized

                self._quantized_image = display_quantized
                self._quantized_image_bytes = _ndarray_to_png_bytes(display_quantized)
                t_name = technique_label(technique)
                self._graph_bytes = generate_comparison_figure(
                    original=display_source,
                    quantized=display_quantized,
                    bits=self._bits_value,
                    technique_name=t_name,
                    hist_color=hist_color,
                )
                self._color_graph_bytes = generate_color_comparison_figure(
                    color_image=image_array,
                    quantized=display_quantized,
                    bits=self._bits_value,
                    technique_name=t_name,
                    gray_image=display_source,
                )
                elapsed = time.perf_counter() - start_time
                metrics = calculate_metrics(gray, quantized, self._bits_value)
                self._update_metrics(metrics, elapsed)

            self._view_mode_buttons.visible = True
            self._render_active_view_mode()

        except Exception as error:
            self._show_message(f"Erro no processamento: {error}", theme.ACCENT)
        finally:
            self._set_processing_state(False)

    async def _on_convert_and_save_gray(self, _: ft.ControlEvent) -> None:
        """Converte a imagem atual diretamente para tons de cinza ou canal isolado (8 bits) e permite o download imediato."""
        if self._source_path is None and self._loaded_array is None:
            self._show_message("Selecione ou clique em uma imagem de teste primeiro.", theme.WARNING)
            return

        try:
            if self._loaded_array is not None:
                image_array = self._loaded_array.copy()
            else:
                image_array = skio.imread(str(self._source_path))

            self._raw_image = image_array
            self._is_color = bool(image_array.ndim == 3 and image_array.shape[2] >= 3)
            gray = to_grayscale(image_array, method=self._selected_gray_method)

            if is_channel_isolation(self._selected_gray_method):
                display_gray = isolate_channel_rgb(image_array, self._selected_gray_method)
                ch_name = get_channel_color_name(self._selected_gray_method).lower()
                stem = self._source_path.stem if self._source_path else "imagem"
                default_name = f"{stem}_canal_{ch_name}_8bits.png"
                dialog_title = f"Salvar Canal {ch_name.capitalize()} Isolado"
                success_msg = f"Canal {ch_name.capitalize()} isolado salvo com sucesso!"
            else:
                display_gray = gray
                stem = self._source_path.stem if self._source_path else "imagem"
                method_str = self._selected_gray_method.name.lower()
                default_name = f"{stem}_cinza_{method_str}_8bits.png"
                dialog_title = "Salvar Imagem em Tons de Cinza"
                success_msg = "Imagem em tons de cinza salva com sucesso!"

            self._gray_image = display_gray
            self._gray_image_bytes = _ndarray_to_png_bytes(display_gray)
            self._quantized_image = display_gray
            self._quantized_image_bytes = self._gray_image_bytes

            # Atualiza a área de visualização com a imagem gerada
            self._result_placeholder.visible = False
            self._single_display_image.src = _bytes_to_data_uri(self._gray_image_bytes)
            self._single_display_container.visible = True
            self._side_by_side_container.visible = False
            self._triple_container.visible = False
            self._zoom_toolbar.visible = True
            self._view_mode_buttons.visible = True
            self._view_mode_buttons.selected = ["image"]
            self._active_view_mode = "image"
            self._btn_save.disabled = False
            self._page.update()

            save_path = await self._save_picker.save_file(
                dialog_title=dialog_title,
                file_name=default_name,
                allowed_extensions=["png"],
            )
            if save_path:
                Path(save_path).write_bytes(self._gray_image_bytes)
                self._show_message(success_msg, theme.SUCCESS)
        except Exception as exc:
            self._show_message(f"Erro ao salvar canal/tons de cinza: {exc}", theme.ACCENT)

    # -----------------------------------------------------------------------
    # Atualização de UI e Modos de Visualização
    # -----------------------------------------------------------------------

    def _render_active_view_mode(self) -> None:
        """Renderiza a visualização atual com base no modo ativo."""
        self._result_placeholder.visible = False
        self._btn_save.disabled = False
        self._zoom_toolbar.visible = True

        self._single_display_container.visible = False
        self._side_by_side_container.visible = False
        self._triple_container.visible = False

        if self._active_view_mode == "graph":
            # Modo 1: Gráficos e Histogramas Cinza / Canal
            if self._graph_bytes:
                self._single_display_image.src = _bytes_to_data_uri(self._graph_bytes)
                self._single_display_container.visible = True

        elif self._active_view_mode == "color_graph":
            # Modo 2: Gráfico com Comparação Colorida e Histograma RGB
            if self._color_graph_bytes:
                self._single_display_image.src = _bytes_to_data_uri(self._color_graph_bytes)
                self._single_display_container.visible = True

        elif self._active_view_mode == "image":
            # Modo 3: Apenas a Imagem Quantizada em alta definição
            if self._quantized_image_bytes:
                self._single_display_image.src = _bytes_to_data_uri(self._quantized_image_bytes)
                self._single_display_container.visible = True

        elif self._active_view_mode == "side_by_side":
            # Modo 4: Comparação lado a lado (Original × Quantizada)
            if self._gray_image_bytes and self._quantized_image_bytes:
                if is_channel_isolation(self._selected_gray_method):
                    ch_name = get_channel_color_name(self._selected_gray_method)
                    self._orig_side_label.value = f"Canal {ch_name} (Original)"
                else:
                    self._orig_side_label.value = "Imagem em Tons de Cinza (Original)"
                self._orig_side_img.src = _bytes_to_data_uri(self._gray_image_bytes)
                self._quant_side_img.src = _bytes_to_data_uri(self._quantized_image_bytes)
                self._side_by_side_container.visible = True

        elif self._active_view_mode == "color_side":
            # Modo 5: Comparação lado a lado (Original Colorida × Quantizada)
            if self._color_image_bytes and self._quantized_image_bytes:
                self._orig_side_label.value = "Imagem Original Colorida (RGB)"
                self._orig_side_img.src = _bytes_to_data_uri(self._color_image_bytes)
                self._quant_side_img.src = _bytes_to_data_uri(self._quantized_image_bytes)
                self._side_by_side_container.visible = True

        elif self._active_view_mode == "triple":
            # Modo 6: Grade Tripla (Colorida × Cinza/Canal × Quantizada)
            if self._color_image_bytes and self._gray_image_bytes and self._quantized_image_bytes:
                self._triple_color_img.src = _bytes_to_data_uri(self._color_image_bytes)
                self._triple_gray_img.src = _bytes_to_data_uri(self._gray_image_bytes)
                self._triple_quant_img.src = _bytes_to_data_uri(self._quantized_image_bytes)
                self._triple_container.visible = True

        self._page.update()


    def _set_processing_state(self, is_processing: bool) -> None:
        """Alterna o estado da UI entre processando e disponível."""
        self._progress_ring.visible = is_processing
        self._progress_label.value = "Processando quantização, aguarde..." if is_processing else ""
        self._btn_process.disabled = is_processing
        self._btn_convert_gray_only.disabled = is_processing
        self._btn_select.disabled = is_processing
        self.controls[1].visible = is_processing
        self._page.update()

    def _update_metrics(self, metrics, elapsed: float) -> None:
        """Atualiza os badges de métricas com os valores calculados."""
        self._badge_mse.content.controls[1].value = f"{metrics.mse:.2f}"
        psnr_str = f"{metrics.psnr:.2f} dB" if metrics.psnr != float("inf") else "∞ dB"
        self._badge_psnr.content.controls[1].value = psnr_str
        self._badge_levels.content.controls[1].value = str(metrics.unique_levels)
        self._badge_time.content.controls[1].value = f"{elapsed:.2f}s"
        self._page.update()

    def _update_metrics_comparison(self, elapsed: float) -> None:
        """Reseta badges de métricas para o modo de comparação completa."""
        self._badge_mse.content.controls[1].value = "ver gráfico"
        self._badge_psnr.content.controls[1].value = "ver gráfico"
        self._badge_levels.content.controls[1].value = f"{2 ** self._bits_value}"
        self._badge_time.content.controls[1].value = f"{elapsed:.2f}s"
        self._page.update()

    def _reset_metrics(self) -> None:
        """Restaura os badges de métricas para o estado inicial."""
        for badge in [self._badge_mse, self._badge_psnr, self._badge_levels, self._badge_time]:
            badge.content.controls[1].value = "—"

    def _show_message(self, message: str, color: str = theme.SUCCESS) -> None:
        """Exibe uma notificação ou SnackBar na tela."""
        snack = ft.SnackBar(ft.Text(message), bgcolor=color)
        if hasattr(self._page, "show_dialog"):
            self._page.show_dialog(snack)
        elif hasattr(self._page, "show_snack_bar"):
            self._page.show_snack_bar(snack)
        self._page.update()


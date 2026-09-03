"""
batch_view.py — Aba de Processamento de Imagens em Lote (Batch).

Compatível com Desktop (nativo) e Web (browser).

Recursos:
  - Fila de Imagens com Feedback Imediato: exibe miniaturas leves, resoluções e status
    das imagens de amostra ou arquivos selecionados antes do processamento.
  - Configurações Didáticas: menu de métodos de conversão em escala de cinza,
    técnicas de quantização (Uniforme, K-Means, Histograma) e resolução em bits.
  - Telemetria e Progresso em Tempo Real: acompanhamento detalhado com barra de progresso,
    tempo de execução e atualização de status por imagem.
  - Galeria Interativa de Resultados: comparativo visual (Original × Quantizada)
    com métricas individuais (MSE, PSNR, Economia), Visualizador de Zoom (10×),
    modal das "🔬 Entranhas do Processo" para cada imagem e download individual/ZIP.
"""

import io
from pathlib import Path
from typing import Any
import zipfile
import gc

import flet as ft
import numpy as np
from PIL import Image

from src.core.batch import (
    SUPPORTED_EXTENSIONS,
    BatchItemResult,
    BatchResult,
    discover_images,
    make_thumbnail_png,
    process_batch,
    process_bytes_batch,
    process_file_list,
)
from src.core.grayscale import (
    GrayscaleMethod,
    colorize_channel,
    is_channel_isolation,
    method_label,
    to_grayscale,
)
from src.core.image_io import (
    MAX_IMAGE_DIMENSION,
    MAX_THUMBNAIL_DIMENSION,
    array_to_png_bytes,
    open_and_downscale_image,
)
from src.core.quantization import (
    QuantizationTechnique,
    get_kmeans_class,
    is_kmeans_loaded,
    technique_label,
)
from src.core.samples import (
    ASSETS_DIR,
    SAMPLE_OPTIONS,
    get_sample_path,
    load_sample_bytes,
)
from src.ui import theme
from src.ui.common import (
    _GRAYSCALE_DETAILS,
    _TECHNIQUE_OPTIONS,
    _bytes_to_data_uri,
    _ndarray_to_png_bytes,
    _read_image_file,
    _register_file_pickers,
)
from src.ui.dialogs import open_inspector_dialog, open_zoom_dialog


# ---------------------------------------------------------------------------
# Estrutura do Item em Fila
# ---------------------------------------------------------------------------


class _BatchQueueItem:
    """Estrutura auxiliar para representar um item carregado na fila."""

    def __init__(
        self,
        name: str,
        path: Path | None = None,
        raw_bytes: bytes | None = None,
        array: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self.raw_bytes = raw_bytes
        self.array = array
        self.dimensions = "—"
        self.color_type = "—"
        self.status = "⏳ Pronto"
        self.status_color = theme.INFO
        self.thumb_bytes: bytes | None = None
        self.card_ref: ft.Container | None = None
        self.status_badge_ref: ft.Text | None = None

        self._inspect_item()

    def _inspect_item(self) -> None:
        try:
            if self.array is None:
                if self.raw_bytes is not None:
                    self.array = open_and_downscale_image(self.raw_bytes, max_dim=MAX_IMAGE_DIMENSION)
                elif self.path is not None and self.path.exists():
                    self.array = open_and_downscale_image(self.path, max_dim=MAX_IMAGE_DIMENSION)

            if self.array is not None:
                h, w = self.array.shape[:2]
                self.dimensions = f"{w}×{h} px"
                is_color = bool(self.array.ndim == 3 and self.array.shape[2] >= 3)
                self.color_type = "RGB (24 bpp)" if is_color else "Cinza (8 bpp)"
                self.thumb_bytes = make_thumbnail_png(self.array, max_size=MAX_THUMBNAIL_DIMENSION)
        except Exception:
            self.dimensions = "Erro"
            self.color_type = "Desconhecido"

    def get_full_png_bytes(self) -> bytes | None:
        """Gera os bytes PNG em resolução total sob demanda (ex.: para Zoom)."""
        if self.array is not None:
            return _ndarray_to_png_bytes(self.array)
        if self.raw_bytes is not None:
            return self.raw_bytes
        if self.path is not None and self.path.exists():
            return self.path.read_bytes()
        return None


# ---------------------------------------------------------------------------
# View Principal
# ---------------------------------------------------------------------------


class BatchView(ft.Column):
    """
    View de processamento em lote adaptativa para Desktop e Web.
    """

    def __init__(self, page: ft.Page) -> None:
        super().__init__(
            scroll=ft.ScrollMode.AUTO,
            spacing=16,
            expand=True,
        )
        self._page = page
        self._is_web: bool = bool(getattr(page, "web", False))

        # Estado da fila e seleção
        self._input_dir: Path | None = None
        self._output_dir: Path | None = None
        self._selected_images: list[Path] = []
        self._web_file_data: list[tuple[str, bytes]] = []
        self._queue_items: list[_BatchQueueItem] = []
        self._batch_result: BatchResult | None = None

        self._selected_technique: QuantizationTechnique | str = QuantizationTechnique.UNIFORM
        self._selected_gray_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE
        self._bits_value: int = 4
        self._is_processing: bool = False

        self._build_controls()
        self._assemble_layout()

    # -----------------------------------------------------------------------
    # Construção dos Controles
    # -----------------------------------------------------------------------

    def _build_controls(self) -> None:
        """Inicializa todos os controles da view de processamento em lote."""
        is_web = self._is_web

        # FilePickers
        self._input_picker = ft.FilePicker()
        self._output_picker = ft.FilePicker()
        self._files_picker = ft.FilePicker()
        self._save_picker = ft.FilePicker()
        _register_file_pickers(
            self._page,
            self._input_picker,
            self._output_picker,
            self._files_picker,
            self._save_picker,
        )

        # ── Rótulos informativos ──────────────────────────────────────────
        self._input_label = ft.Text(
            "Nenhuma pasta ou imagens selecionadas",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
            italic=True,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        self._output_label = ft.Text(
            "Download automático após processamento" if is_web else "Destino padrão: assets/lote_resultado",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
            italic=True,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        self._image_count_text = ft.Text(
            "",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
        )

        # ── Botões de Início (Disponíveis tanto no Card 1 quanto no Card 2) ──
        self._btn_start_top = ft.Button(
            content="🚀 Iniciar Processamento em Lote Agora",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_start,
            disabled=True,
            bgcolor=theme.SUCCESS,
            color="#FFFFFF",
            height=44,
        )
        self._btn_start = ft.Button(
            content="🚀 Iniciar Processamento em Lote",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_start,
            disabled=True,
            bgcolor=theme.SUCCESS,
            color="#FFFFFF",
            height=44,
        )

        # ── Seção de Fila de Imagens com Feedback Imediato ────────────────
        self._queue_grid = ft.Row(
            controls=[],
            wrap=True,
            spacing=12,
        )
        self._queue_container = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.PHOTO_LIBRARY, size=20, color=theme.PRIMARY_LIGHT),
                            ft.Text(
                                "Fila de Imagens Carregadas (Preview Imediato)",
                                weight=ft.FontWeight.BOLD,
                                size=theme.FONT_SUBTITLE,
                                color=ft.Colors.ON_SURFACE,
                            ),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Divider(height=1),
                    self._queue_grid,
                    ft.Divider(height=1),
                    ft.Row(
                        controls=[self._btn_start_top],
                        alignment=ft.MainAxisAlignment.END,
                    ),
                ],
                spacing=10,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=10,
            padding=12,
            visible=False,
        )

        # ── Menu de Configurações ─────────────────────────────────────────
        self._convert_to_gray = True
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
                                "Modo de Entrada / Pré-Processamento do Lote:",
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
                        "• Se MARCADO (Sim): converte todas as imagens para 1 canal (Grayscale) antes de entrar no quantizador.\n"
                        "• Se DESMARCADO (Não): preserva os 3 canais (RGB) e quantiza o lote no espaço de cores tridimensional.",
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

        # Caixa didática de informações da conversão para cinza
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

        self._gray_section_container = ft.Column(
            controls=[
                ft.Text("2. Método de Conversão para Escala de Cinza / Canais:", weight=ft.FontWeight.BOLD, size=theme.FONT_BODY),
                self._gray_category_selector,
                self._gray_options_selector,
                self._gray_info_box,
            ],
            spacing=8,
            visible=True,
        )

        # Dropdown: Técnica de Quantização
        self._technique_dropdown = ft.Dropdown(
            label="Técnica de Quantização",
            options=[
                ft.dropdown.Option(
                    key=str(t.value) if isinstance(t, QuantizationTechnique) else str(t),
                    text=label,
                )
                for t, label in _TECHNIQUE_OPTIONS
                if t != "BOTH"
            ],
            value=str(QuantizationTechnique.UNIFORM.value),
            color=ft.Colors.ON_SURFACE,
            on_select=self._on_technique_changed,
        )

        # Compatibilidade com referências legadas
        self._gray_dropdown = ft.Dropdown(
            label="Método de Conversão para Cinza (Legado)",
            options=[
                ft.dropdown.Option(key=str(m.value), text=details["title"])
                for m, details in _GRAYSCALE_DETAILS.items()
            ],
            value=str(GrayscaleMethod.LUMINANCE.value),
            visible=False,
        )

        # Slider de bits
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

        # ── Progresso ────────────────────────────────────────────────────
        self._progress_bar = ft.ProgressBar(
            value=0.0,
            color=theme.PRIMARY,
            height=8,
            border_radius=4,
        )
        self._progress_text = ft.Text(
            "Aguardando início...",
            color=ft.Colors.ON_SURFACE_VARIANT,
            size=theme.FONT_CAPTION,
        )
        self._progress_percent = ft.Text(
            "0%",
            color=ft.Colors.ON_SURFACE,
            size=theme.FONT_BODY,
            weight=ft.FontWeight.BOLD,
        )
        self._log_list = ft.ListView(
            height=160,
            spacing=2,
            auto_scroll=True,
        )

        # ── Botões de Ação ───────────────────────────────────────────────
        self._btn_sample_batch = ft.Button(
            content="⚡ Carregar 5 Amostras do App",
            icon=ft.Icons.AUTO_AWESOME,
            on_click=self._on_select_sample_batch,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_select_files = ft.OutlinedButton(
            content="Selecionar Arquivos",
            icon=ft.Icons.PERM_MEDIA,
            on_click=self._on_select_files,
        )
        self._btn_input = ft.OutlinedButton(
            content="Pasta de Entrada",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_select_input_dir,
            visible=not is_web,
        )
        self._btn_output = ft.Button(
            content="Mudar Pasta de Saída",
            icon=ft.Icons.DRIVE_FILE_MOVE,
            on_click=self._on_select_output_dir,
            bgcolor=theme.PRIMARY_DARK,
            color="#FFFFFF",
            visible=not is_web,
        )

        # ── Cards de Resultados ──────────────────────────────────────────
        self._summary_card = ft.Container(visible=False)
        self._download_section = ft.Container(visible=False)
        self._results_gallery = ft.Column(controls=[], spacing=14)
        self._results_gallery_container = ft.Container(
            content=ft.Column(
                controls=[
                    theme.section_title("🖼️  Galeria Interativa de Resultados do Lote"),
                    ft.Divider(height=1),
                    ft.Text(
                        "Para cada imagem processada, você pode abrir o pop-up com Zoom Interativo (10×) ou "
                        "inspecionar as Entranhas do Processamento (Amostra 5×5, Aritmética, Tabela de Quantização e Heatmap de Erro).",
                        size=theme.FONT_CAPTION,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                    self._results_gallery,
                ],
                spacing=10,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=theme.BORDER_RADIUS,
            padding=theme.PADDING_CARD,
            visible=False,
        )

        # Banner web
        self._web_banner = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.INFO_OUTLINE, color=theme.PRIMARY_LIGHT, size=18),
                    ft.Text(
                        "Modo Web: selecione as imagens pelo botão abaixo. "
                        "Os resultados poderão ser visualizados interativamente e baixados individualmente ou em ZIP.",
                        size=theme.FONT_CAPTION,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                        expand=True,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=10,
            visible=is_web,
        )

    def _assemble_layout(self) -> None:
        """Monta o layout da view."""
        is_web = self._is_web

        input_buttons: list[ft.Control] = [self._btn_sample_batch, self._btn_select_files]
        if not is_web:
            input_buttons.append(self._btn_input)

        output_row = ft.Row(
            controls=[self._btn_output, self._output_label],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            visible=not is_web,
        )

        input_info_row = ft.Row(
            controls=[
                ft.Text("Entrada:", weight=ft.FontWeight.BOLD, size=theme.FONT_CAPTION),
                self._input_label,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.controls = [
            # Card 1: Seleção de Imagens e Diretórios
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("📁  Seleção de Imagens e Fila do Lote"),
                        ft.Divider(height=1),
                        self._web_banner,
                        ft.Row(controls=input_buttons, spacing=10, wrap=True),
                        input_info_row,
                        output_row,
                        self._image_count_text,
                        self._queue_container,
                    ],
                    spacing=12,
                )
            ),
            # Card 2: Configurações do Processamento em Lote
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("⚙️  Configurações de Quantização do Lote"),
                        ft.Divider(height=1),
                        self._preprocess_box,
                        self._gray_section_container,
                        ft.Divider(height=1),
                        ft.Text("3. Técnica e Resolução em Bits:", weight=ft.FontWeight.BOLD, size=theme.FONT_BODY),
                        self._technique_dropdown,
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
                                self._bits_slider_hint,
                            ],
                            spacing=6,
                        ),
                        ft.Divider(height=1),
                        self._btn_start,
                    ],
                    spacing=12,
                )
            ),
            # Card 3: Progresso e Log de Execução
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("⏳  Progresso do Processamento"),
                        ft.Divider(height=1),
                        ft.Row(
                            controls=[self._progress_text, self._progress_percent],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        self._progress_bar,
                        ft.Container(
                            content=self._log_list,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            border_radius=8,
                            padding=8,
                            height=160,
                        ),
                    ],
                    spacing=10,
                )
            ),
            # Card 4: Resumo Geral e Ações Globais
            self._summary_card,
            # Card 5: Galeria de Resultados Detalhada
            self._results_gallery_container,
            # Card 6: Seção de downloads web legado
            self._download_section,
        ]

    # -----------------------------------------------------------------------
    # Feedback e Notificações
    # -----------------------------------------------------------------------

    def _show_message(self, message: str, color: str = theme.SUCCESS) -> None:
        """Exibe uma notificação ou SnackBar na tela."""
        snack = ft.SnackBar(ft.Text(message), bgcolor=color)
        if hasattr(self._page, "show_dialog"):
            self._page.show_dialog(snack)
        elif hasattr(self._page, "show_snack_bar"):
            self._page.show_snack_bar(snack)
        self._page.update()

    # -----------------------------------------------------------------------
    # Gerenciamento da Fila de Imagens
    # -----------------------------------------------------------------------

    def _render_queue_preview(self) -> None:
        """Atualiza a grade visual de miniaturas dos itens carregados na fila."""
        self._queue_grid.controls.clear()

        if not self._queue_items:
            self._queue_container.visible = False
            self._page.update()
            return

        for item in self._queue_items:
            img_ctrl = ft.Image(
                src=_bytes_to_data_uri(item.thumb_bytes),
                width=80,
                height=80,
                fit=getattr(ft.BoxFit, "COVER", None) if hasattr(ft, "BoxFit") else None,
                border_radius=6,
            )

            status_text = ft.Text(
                item.status,
                size=11,
                weight=ft.FontWeight.BOLD,
                color=item.status_color,
            )
            item.status_badge_ref = status_text

            btn_zoom = ft.IconButton(
                icon=ft.Icons.ZOOM_IN,
                icon_size=18,
                tooltip=f"Ampliar original: {item.name}",
                on_click=lambda _, it=item: open_zoom_dialog(
                    self._page, f"Imagem de Entrada — {it.name}", it.get_full_png_bytes()
                ),
            )

            item_card = ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Container(
                            content=img_ctrl,
                            on_click=lambda _, it=item: open_zoom_dialog(
                                self._page, f"Imagem de Entrada — {it.name}", it.get_full_png_bytes()
                            ),
                            tooltip="Clique para ampliar original",
                            ink=True,
                            border_radius=6,
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    item.name,
                                    weight=ft.FontWeight.BOLD,
                                    size=12,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                    width=130,
                                ),
                                ft.Text(f"{item.dimensions} • {item.color_type}", size=11, color=theme.TEXT_SECONDARY),
                                ft.Row(controls=[status_text, btn_zoom], spacing=2, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ],
                            spacing=2,
                        ),
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=ft.Colors.SURFACE_CONTAINER,
                border_radius=8,
                padding=8,
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
                width=260,
            )
            item.card_ref = item_card
            self._queue_grid.controls.append(item_card)

        self._queue_container.visible = True
        self._update_start_button_state()
        self._page.update()

    # -----------------------------------------------------------------------
    # Handlers de Seleção
    # -----------------------------------------------------------------------

    def _on_select_sample_batch(self, _: ft.ControlEvent) -> None:
        """Carrega as 5 imagens de amostra embutidas no app com suporte a Web e Desktop."""
        sample_items: list[tuple[str, Path, bytes]] = []
        for s in SAMPLE_OPTIONS:
            name = s["name"]
            try:
                p = get_sample_path(name)
                b = load_sample_bytes(name, max_dim=MAX_IMAGE_DIMENSION)
                sample_items.append((name, p, b))
            except Exception:
                continue

        if not sample_items:
            self._image_count_text.value = "⚠️  Nenhuma imagem de amostra encontrada na pasta assets."
            self._image_count_text.color = theme.WARNING
            self._show_message("Nenhuma imagem de amostra encontrada na pasta assets.", theme.WARNING)
            self._page.update()
            return

        self._selected_images = [p for _, p, _ in sample_items]
        self._web_file_data = [(name, b) for name, _, b in sample_items]
        self._input_dir = None

        self._queue_items = [
            _BatchQueueItem(name=name, path=p, raw_bytes=b)
            for name, p, b in sample_items
        ]

        self._input_label.value = (
            f"Amostras do App ({len(sample_items)} imagens: Retrato, Benchmark, Lena, Ayla, Pentágono)"
        )
        self._input_label.italic = False
        self._input_label.color = theme.TEXT_PRIMARY

        if not self._is_web and not self._output_dir:
            self._output_dir = ASSETS_DIR / "lote_resultado"
            self._output_label.value = str(self._output_dir)
            self._output_label.italic = False
            self._output_label.color = theme.TEXT_PRIMARY

        self._image_count_text.value = f"✅  {len(sample_items)} imagens carregadas e prontas para quantização."
        self._image_count_text.color = theme.SUCCESS

        self._render_queue_preview()
        self._show_message(f"✅ {len(sample_items)} imagens carregadas na fila! Clique no botão verde 'Iniciar' para quantizar.")

    async def _on_select_input_dir(self, _: ft.ControlEvent) -> None:
        """Desktop: abre diálogo para selecionar pasta inteira."""
        try:
            path_str = await self._input_picker.get_directory_path(
                dialog_title="Selecionar Pasta de Entrada"
            )
        except Exception as exc:
            self._image_count_text.value = f"⚠️  Erro ao abrir seletor: {exc}"
            self._image_count_text.color = theme.WARNING
            self._show_message(f"Erro ao abrir seletor: {exc}", theme.ACCENT)
            self._page.update()
            return

        if not path_str:
            return

        self._input_dir = Path(path_str)
        self._selected_images = []
        self._web_file_data = []
        self._input_label.value = str(self._input_dir)
        self._input_label.italic = False
        self._input_label.color = theme.TEXT_PRIMARY

        if not self._output_dir:
            self._output_dir = self._input_dir / "lote_resultado"
            self._output_label.value = str(self._output_dir)
            self._output_label.italic = False
            self._output_label.color = theme.TEXT_PRIMARY

        try:
            images = discover_images(self._input_dir)
            self._selected_images = images
            self._queue_items = [
                _BatchQueueItem(name=p.name, path=p) for p in images
            ]
            self._image_count_text.value = f"✅  {len(images)} imagem(ns) encontrada(s) na pasta."
            self._image_count_text.color = theme.SUCCESS
            self._render_queue_preview()
            self._show_message(f"✅ {len(images)} imagem(ns) adicionada(s) à fila!")
        except ValueError:
            self._image_count_text.value = "⚠️  Nenhuma imagem suportada encontrada nesta pasta."
            self._image_count_text.color = theme.WARNING
            self._queue_items = []
            self._render_queue_preview()
            self._show_message("Nenhuma imagem suportada encontrada nesta pasta.", theme.WARNING)

    async def _on_select_files(self, _: ft.ControlEvent) -> None:
        """Seleciona múltiplos arquivos de imagem individualmente."""
        try:
            exts = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]
            files = await self._files_picker.pick_files(
                dialog_title="Selecionar Imagens",
                allow_multiple=True,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=exts,
                with_data=True,
            )
        except Exception as exc:
            self._image_count_text.value = f"⚠️  Erro ao selecionar arquivos: {exc}"
            self._image_count_text.color = theme.WARNING
            self._show_message(f"Erro ao selecionar arquivos: {exc}", theme.ACCENT)
            self._page.update()
            return

        if not files:
            return

        self._input_dir = None
        self._queue_items = []

        if self._is_web:
            self._web_file_data = [(f.name, f.bytes) for f in files if f.bytes is not None]
            self._selected_images = []
            for name, raw in self._web_file_data:
                self._queue_items.append(_BatchQueueItem(name=name, raw_bytes=raw))
            count = len(self._web_file_data)
        else:
            self._selected_images = [Path(f.path) for f in files if f.path]
            self._web_file_data = []
            for f in files:
                p = Path(f.path) if f.path else None
                self._queue_items.append(
                    _BatchQueueItem(name=f.name, path=p, raw_bytes=f.bytes)
                )
            count = len(self._selected_images)

            if self._selected_images and not self._output_dir:
                self._output_dir = self._selected_images[0].parent / "lote_resultado"
                self._output_label.value = str(self._output_dir)
                self._output_label.italic = False
                self._output_label.color = theme.TEXT_PRIMARY

        names_preview = ", ".join(f.name for f in files[:3])
        if len(files) > 3:
            names_preview += f" e mais {len(files) - 3}..."
        self._input_label.value = f"{count} arquivo(s): {names_preview}"
        self._input_label.italic = False
        self._input_label.color = theme.TEXT_PRIMARY
        self._image_count_text.value = f"✅  {count} arquivo(s) selecionado(s)."
        self._image_count_text.color = theme.SUCCESS

        self._render_queue_preview()
        self._show_message(f"✅ {count} imagem(ns) adicionada(s) à fila!")

    async def _on_select_output_dir(self, _: ft.ControlEvent) -> None:
        """Desktop: seleciona pasta de destino dos arquivos quantizados."""
        try:
            path_str = await self._output_picker.get_directory_path(
                dialog_title="Selecionar Pasta de Saída"
            )
        except Exception as exc:
            self._output_label.value = f"⚠️  Erro ao abrir seletor: {exc}"
            self._output_label.color = theme.WARNING
            self._show_message(f"Erro ao abrir seletor: {exc}", theme.ACCENT)
            self._page.update()
            return

        if not path_str:
            return

        self._output_dir = Path(path_str)
        self._output_label.value = str(self._output_dir)
        self._output_label.italic = False
        self._output_label.color = theme.TEXT_PRIMARY
        self._update_start_button_state()
        self._show_message("Pasta de saída configurada.")
        self._page.update()

    # -----------------------------------------------------------------------
    # Handlers de Configuração
    # -----------------------------------------------------------------------

    def _on_gray_category_changed(self, event: ft.ControlEvent) -> None:
        if not event.control.selected:
            return
        cat = next(iter(event.control.selected), "weighted")
        if cat == "weighted":
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
        else:
            self._gray_options_selector.segments = [
                ft.Segment(
                    value=str(GrayscaleMethod.CHANNEL_R.value),
                    label=ft.Text("Canal R (Vermelho)"),
                    icon=ft.Icon(ft.Icons.LOOKS_ONE),
                ),
                ft.Segment(
                    value=str(GrayscaleMethod.CHANNEL_G.value),
                    label=ft.Text("Canal G (Verde)"),
                    icon=ft.Icon(ft.Icons.LOOKS_TWO),
                ),
                ft.Segment(
                    value=str(GrayscaleMethod.CHANNEL_B.value),
                    label=ft.Text("Canal B (Azul)"),
                    icon=ft.Icon(ft.Icons.LOOKS_3),
                ),
            ]
            self._selected_gray_method = GrayscaleMethod.CHANNEL_R

        self._gray_options_selector.selected = [str(self._selected_gray_method.value)]
        self._update_gray_info_box()
        self._page.update()

    def _on_gray_method_segmented_changed(self, event: ft.ControlEvent) -> None:
        if not event.control.selected:
            return
        method_val = int(next(iter(event.control.selected)))
        for method in GrayscaleMethod:
            if method.value == method_val:
                self._selected_gray_method = method
                break
        self._update_gray_info_box()
        self._page.update()

    def _update_gray_info_box(self) -> None:
        details = _GRAYSCALE_DETAILS[self._selected_gray_method]
        self._gray_info_title.value = details["title"]
        self._gray_info_formula.value = f"Fórmula: {details['formula']}"
        self._gray_info_desc.value = details["desc"]

    def _on_convert_grayscale_toggled(self, event: ft.ControlEvent) -> None:
        """Alterna entre modo Grayscale (1 canal) e modo Colorido Direto (3 canais RGB) no lote."""
        selected = getattr(event.control, "selected", None)
        if not selected:
            return
        mode_str = next(iter(selected))
        self._convert_to_gray = (mode_str == "yes")
        self._gray_section_container.visible = self._convert_to_gray
        self._update_bits_label()
        self._page.update()

    def _update_bits_label(self) -> None:
        """Atualiza dinamicamente o texto explicativo do slider conforme o modo e a técnica."""
        if self._convert_to_gray:
            n_tons = 2 ** self._bits_value
            self._bits_label.value = f"{self._bits_value} bits  —  {n_tons} tons de cinza"
            self._bits_slider_hint.value = "1=2 tons · 2=4 tons · 3=8 tons · 4=16 tons · 5=32 tons · 6=64 tons · 7=128 tons · 8=256 tons"
        else:
            if self._selected_technique == QuantizationTechnique.KMEANS:
                n_cores = 2 ** self._bits_value
                self._bits_label.value = f"Paleta de {n_cores} cores  —  K-Means 3D ({self._bits_value} bits)"
                self._bits_slider_hint.value = f"Quantização vetorial 3D agrupando em {n_cores} centróides de cores RGB"
            else:
                total_cores = (2 ** self._bits_value) ** 3
                self._bits_label.value = f"{self._bits_value} bits/canal  —  {total_cores} cores no total ((2^{self._bits_value})^3)"
                self._bits_slider_hint.value = f"Mapeamento escalar por canal: R, G, B em {2**self._bits_value} níveis cada -> {total_cores} cores"

    def _on_technique_changed(self, event: ft.ControlEvent) -> None:
        val = event.control.value
        try:
            int_val = int(val)
            for t in QuantizationTechnique:
                if t.value == int_val:
                    self._selected_technique = t
                    if t == QuantizationTechnique.KMEANS:
                        self._check_and_notify_kmeans_lazy()
                    break
        except ValueError:
            self._selected_technique = val
            if val == "BOTH":
                self._check_and_notify_kmeans_lazy()
        self._update_bits_label()
        self._page.update()

    def _check_and_notify_kmeans_lazy(self) -> None:
        """Garante a inicialização do módulo K-Means de forma silenciosa e não-bloqueante."""
        if not is_kmeans_loaded():
            get_kmeans_class()

    def _on_bits_changed(self, event: ft.ControlEvent) -> None:
        self._bits_value = int(event.control.value)
        self._update_bits_label()
        self._page.update()

    # -----------------------------------------------------------------------
    # Lógica de Execução do Lote
    # -----------------------------------------------------------------------

    def _on_start(self, _: ft.ControlEvent) -> None:
        """Inicia o processamento em background."""
        if self._is_processing:
            return

        if not self._queue_items and not self._selected_images and not self._web_file_data and not self._input_dir:
            self._show_message("Adicione ao menos uma imagem à fila antes de iniciar.", theme.WARNING)
            return

        if not self._is_web and not self._output_dir:
            self._output_dir = ASSETS_DIR / "lote_resultado"
            self._output_label.value = str(self._output_dir)
            self._output_label.italic = False
            self._output_label.color = theme.TEXT_PRIMARY

        self._is_processing = True
        self._btn_start.disabled = True
        self._btn_start_top.disabled = True
        self._btn_input.disabled = True
        self._btn_select_files.disabled = True
        self._btn_sample_batch.disabled = True
        self._btn_output.disabled = True

        self._log_list.controls.clear()
        self._summary_card.visible = False
        self._results_gallery_container.visible = False
        self._results_gallery.controls.clear()
        self._progress_bar.value = None
        self._progress_percent.value = "0%"
        self._progress_text.value = "🚀 Iniciando processamento do lote..."

        for q_item in self._queue_items:
            q_item.status = "⏳ Aguardando..."
            q_item.status_color = theme.WARNING
            if q_item.status_badge_ref:
                q_item.status_badge_ref.value = q_item.status
                q_item.status_badge_ref.color = q_item.status_color

        self._show_message("🚀 Iniciando quantização em lote...", theme.INFO)
        self._page.update()

        technique = self._get_selected_technique()
        gray_method = self._selected_gray_method
        bits = self._bits_value
        convert_to_gray = self._convert_to_gray

        if self._is_web or self._web_file_data:
            if hasattr(self._page, "run_thread"):
                self._page.run_thread(self._run_worker_memory, technique, bits, gray_method, convert_to_gray)
            else:
                import threading
                threading.Thread(
                    target=self._run_worker_memory,
                    args=(technique, bits, gray_method, convert_to_gray),
                    daemon=True,
                ).start()
        else:
            if hasattr(self._page, "run_thread"):
                self._page.run_thread(self._run_worker_filesystem, technique, bits, gray_method, convert_to_gray)
            else:
                import threading
                threading.Thread(
                    target=self._run_worker_filesystem,
                    args=(technique, bits, gray_method, convert_to_gray),
                    daemon=True,
                ).start()

    def _run_worker_memory(
        self,
        technique: QuantizationTechnique,
        bits: int,
        grayscale_method: GrayscaleMethod,
        convert_to_gray: bool = True,
    ) -> None:
        """Worker em memória (web ou imagens carregadas com bytes)."""
        try:
            if self._web_file_data:
                images_data = self._web_file_data
            elif self._queue_items:
                images_data = []
                for it in self._queue_items:
                    if it.raw_bytes:
                        images_data.append((it.name, it.raw_bytes))
                    elif it.path:
                        images_data.append((it.name, it.path.read_bytes()))
                    elif it.array is not None:
                        images_data.append((it.name, _ndarray_to_png_bytes(it.array)))
            else:
                raise ValueError("Nenhuma imagem disponível.")

            result = process_bytes_batch(
                images=images_data,
                technique=technique,
                bits=bits,
                grayscale_method=grayscale_method,
                convert_to_grayscale=convert_to_gray,
                progress_callback=self._on_progress,
                item_callback=self._on_item_finished,
            )
            self._batch_result = result
            self._on_batch_complete(result)
        except Exception as exc:
            self._progress_text.value = f"Erro no lote: {exc}"
            self._show_message(f"Erro no processamento: {exc}", theme.ACCENT)
            self._reset_controls()
            self._page.update()

    def _run_worker_filesystem(
        self,
        technique: QuantizationTechnique,
        bits: int,
        grayscale_method: GrayscaleMethod,
        convert_to_gray: bool = True,
    ) -> None:
        """Worker no disco (desktop)."""
        try:
            images = self._selected_images
            if not images and self._input_dir:
                images = discover_images(self._input_dir)

            if not images and self._queue_items:
                images = [it.path for it in self._queue_items if it.path is not None]

            if not images:
                self._run_worker_memory(technique, bits, grayscale_method, convert_to_gray)
                return

            out_dir = self._output_dir or (ASSETS_DIR / "lote_resultado")
            result = process_file_list(
                images=images,
                output_dir=out_dir,
                technique=technique,
                bits=bits,
                grayscale_method=grayscale_method,
                convert_to_grayscale=convert_to_gray,
                progress_callback=self._on_progress,
                item_callback=self._on_item_finished,
            )
            self._batch_result = result
            self._on_batch_complete(result)
        except Exception as exc:
            self._progress_text.value = f"Erro no lote: {exc}"
            self._show_message(f"Erro no processamento: {exc}", theme.ACCENT)
            self._reset_controls()
            self._page.update()

    # -----------------------------------------------------------------------
    # Callbacks de Progresso e Galeria de Resultados
    # -----------------------------------------------------------------------

    def _on_progress(self, processed: int, total: int, filename: str) -> None:
        """Atualiza a barra de progresso e texto em tempo real."""
        ratio = processed / total if total > 0 else 0.0
        self._progress_bar.value = ratio
        self._progress_percent.value = f"{int(ratio * 100)}%"
        self._progress_text.value = f"[{processed}/{total}]  Processando {filename}..."
        self._page.update()

    def _on_item_finished(self, item_result: BatchItemResult) -> None:
        """Chamado no término individual de cada item para atualizar fila e log."""
        for q_item in self._queue_items:
            if q_item.name == item_result.filename:
                if item_result.success:
                    psnr_txt = f"{item_result.metrics.psnr:.1f} dB" if item_result.metrics else ""
                    q_item.status = f"✅ Concluído ({item_result.elapsed_seconds:.2f}s • {psnr_txt})"
                    q_item.status_color = theme.SUCCESS
                else:
                    q_item.status = "❌ Falha"
                    q_item.status_color = theme.ACCENT

                if q_item.status_badge_ref:
                    q_item.status_badge_ref.value = q_item.status
                    q_item.status_badge_ref.color = q_item.status_color
                break

        if item_result.success:
            self._log_list.controls.append(
                ft.Text(
                    f"✅ {item_result.filename} — {item_result.elapsed_seconds:.2f}s (PSNR: {item_result.metrics.psnr:.2f} dB)",
                    color=theme.SUCCESS,
                    size=theme.FONT_CAPTION,
                )
            )
        else:
            self._log_list.controls.append(
                ft.Text(
                    f"❌ {item_result.filename} — Erro: {item_result.error}",
                    color=theme.ACCENT,
                    size=theme.FONT_CAPTION,
                )
            )
        self._page.update()

    def _on_batch_complete(self, result: BatchResult) -> None:
        """Monta o resumo estatístico e a galeria de resultados interativa."""
        self._reset_controls()
        self._progress_bar.value = 1.0
        self._progress_percent.value = "100%"
        self._progress_text.value = f"🎉 Processamento concluído em {result.total_elapsed_seconds:.2f}s!"

        # 1. Atualiza e exibe o resumo estatístico geral
        self._summary_card.content = self._build_summary_card(result)
        self._summary_card.visible = True

        # 2. Constrói a galeria com os cards modulares de cada item
        self._results_gallery.controls = [
            self._build_result_item_card(item)
            for item in result.items
            if item.success
        ]
        self._results_gallery_container.visible = bool(self._results_gallery.controls)

        self._show_message(f"🎉 Lote concluído com sucesso ({result.success_count} imagens quantizadas)!")
        self._page.update()

    def _build_summary_card(self, result: BatchResult) -> ft.Container:
        """Constrói o card de resumo geral com métricas agregadas do lote."""
        summary_col = ft.Column(
            controls=[
                theme.section_title("📊  Resumo Geral do Processamento em Lote"),
                ft.Divider(height=1),
                ft.Row(
                    controls=[
                        theme.metric_badge("Total", str(result.total)),
                        theme.metric_badge("Sucesso", str(result.success_count), color=theme.SUCCESS),
                        theme.metric_badge("Falhas", str(result.failure_count), color=theme.ACCENT if result.failure_count > 0 else theme.SUCCESS),
                        theme.metric_badge("Tempo Total", f"{result.total_elapsed_seconds:.2f}s", color=theme.WARNING),
                        theme.metric_badge("PSNR Médio", f"{result.avg_psnr:.2f} dB", color=theme.SUCCESS),
                        theme.metric_badge("MSE Médio", f"{result.avg_mse:.2f}"),
                        theme.metric_badge("Economia", f"{result.avg_savings_pct:.1f}%", color=theme.WARNING),
                    ],
                    spacing=10,
                    wrap=True,
                ),
                ft.Divider(height=1),
                ft.Row(
                    controls=[
                        ft.Button(
                            content="📦 Baixar Todos os Resultados (ZIP)",
                            icon=ft.Icons.FOLDER_ZIP,
                            on_click=self._on_download_zip_clicked,
                            bgcolor=theme.PRIMARY,
                            color="#FFFFFF",
                        ),
                        (
                            ft.Text(
                                f"📁 Arquivos salvos em: {result.output_dir}",
                                size=theme.FONT_CAPTION,
                                color=ft.Colors.ON_SURFACE_VARIANT,
                            )
                            if result.output_dir and not self._is_web
                            else ft.Container()
                        ),
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    wrap=True,
                ),
            ],
            spacing=10,
        )
        return theme.card(summary_col)

    def _build_result_item_card(self, item: BatchItemResult) -> ft.Container:
        """Constrói o card interativo de um item individual da galeria de resultados."""
        src_thumb = item.source_thumb_bytes or make_thumbnail_png(item.raw_array, MAX_THUMBNAIL_DIMENSION)
        quant_thumb = item.quantized_thumb_bytes or make_thumbnail_png(item.quantized_array, MAX_THUMBNAIL_DIMENSION)

        orig_img = ft.Image(
            src=_bytes_to_data_uri(src_thumb),
            width=110,
            height=110,
            fit=getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None,
            border_radius=8,
        )
        quant_img = ft.Image(
            src=_bytes_to_data_uri(quant_thumb),
            width=110,
            height=110,
            fit=getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None,
            border_radius=8,
        )

        btn_zoom = ft.Button(
            content="🔍 Zoom",
            icon=ft.Icons.ZOOM_IN,
            on_click=lambda _, it=item: open_zoom_dialog(
                self._page, f"Resultado Quantizado — {it.filename}", it.quantized_bytes
            ),
            bgcolor=theme.PRIMARY_LIGHT,
            color="#FFFFFF",
        )

        btn_inspect = ft.Button(
            content="🔬 Entranhas do Processo",
            icon=ft.Icons.ANALYTICS,
            on_click=lambda _, it=item: open_inspector_dialog(
                page=self._page,
                raw_image=it.raw_array,
                gray_image=it.gray_array,
                quantized_image=it.quantized_array,
                bits=self._bits_value,
                technique=self._get_selected_technique(),
                method=self._selected_gray_method,
            ),
            bgcolor=theme.ACCENT,
            color="#FFFFFF",
        )

        btn_save = ft.OutlinedButton(
            content="💾 Baixar",
            icon=ft.Icons.DOWNLOAD,
            on_click=lambda _, it=item: self._save_single_item(it),
        )

        m = item.metrics
        mse_str = f"{m.mse:.2f}" if m else "—"
        psnr_str = f"{m.psnr:.2f} dB" if m else "—"
        levels_str = f"{m.unique_levels} tons" if m else "—"

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(f"📄 {item.filename}", weight=ft.FontWeight.BOLD, size=theme.FONT_BODY),
                            ft.Text(f"⏱️ {item.elapsed_seconds:.2f}s", size=theme.FONT_CAPTION, color=theme.TEXT_SECONDARY),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Divider(height=1),
                    ft.Row(
                        controls=[
                            ft.Column(
                                controls=[
                                    ft.Text("Original", size=11, weight=ft.FontWeight.BOLD, color=theme.TEXT_SECONDARY),
                                    ft.Container(
                                        content=orig_img,
                                        on_click=lambda _, it=item: open_zoom_dialog(
                                            self._page, f"Original — {it.filename}", it.source_bytes
                                        ),
                                        ink=True,
                                        tooltip="Clique para zoom no original",
                                    ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            ft.Icon(ft.Icons.ARROW_FORWARD, color=theme.PRIMARY_LIGHT, size=20),
                            ft.Column(
                                controls=[
                                    ft.Text("Quantizada", size=11, weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT),
                                    ft.Container(
                                        content=quant_img,
                                        on_click=lambda _, it=item: open_zoom_dialog(
                                            self._page, f"Quantizada — {it.filename}", it.quantized_bytes
                                        ),
                                        ink=True,
                                        tooltip="Clique para zoom na quantizada",
                                    ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            ft.Column(
                                controls=[
                                    ft.Row(
                                        controls=[
                                            theme.metric_badge("MSE", mse_str),
                                            theme.metric_badge("PSNR", psnr_str, color=theme.SUCCESS),
                                            theme.metric_badge("Níveis", levels_str),
                                        ],
                                        spacing=6,
                                        wrap=True,
                                    ),
                                    ft.Row(
                                        controls=[btn_zoom, btn_inspect, btn_save],
                                        spacing=6,
                                        wrap=True,
                                    ),
                                ],
                                spacing=8,
                            ),
                        ],
                        spacing=14,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                ],
                spacing=8,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=10,
            padding=12,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
        )

    async def _save_single_item(self, item: BatchItemResult) -> None:
        """Salva ou faz download do arquivo PNG quantizado de um item específico."""
        if not item.quantized_bytes:
            return
        try:
            save_path = await self._save_picker.save_file(
                dialog_title=f"Salvar Imagem Quantizada — {item.filename}",
                file_name=f"quantizado_{item.filename}",
                allowed_extensions=["png"],
                src_bytes=item.quantized_bytes,
            )
            if save_path and not getattr(self._page, "web", False):
                Path(save_path).write_bytes(item.quantized_bytes)
            self._show_message(f"✅ Imagem '{item.filename}' salva com sucesso!", theme.SUCCESS)
        except Exception as exc:
            self._show_message(f"Erro ao salvar {item.filename}: {exc}", theme.ACCENT)

    async def _on_download_zip_clicked(self, _: ft.ControlEvent | None = None) -> None:
        """Gera e baixa um arquivo ZIP contendo todas as imagens quantizadas."""
        if not self._batch_result or not self._batch_result.items:
            self._show_message("Nenhum resultado de lote disponível para exportação.", theme.WARNING)
            return

        try:
            self._show_message("📦 Preparando arquivo ZIP do lote...", theme.INFO)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in self._batch_result.items:
                    if item.success and item.quantized_bytes is not None:
                        zf.writestr(f"quantizado_{item.filename}", item.quantized_bytes)

            zip_bytes = buf.getvalue()
            save_path = await self._save_picker.save_file(
                dialog_title="Salvar Lote em ZIP",
                file_name="lote_quantizado.zip",
                allowed_extensions=["zip"],
                src_bytes=zip_bytes,
            )
            if save_path and not getattr(self._page, "web", False):
                Path(save_path).write_bytes(zip_bytes)
            self._show_message("✅ Arquivo ZIP do lote salvo com sucesso!", theme.SUCCESS)
        except Exception as exc:
            self._show_message(f"Erro ao salvar ZIP: {exc}", theme.ACCENT)

    async def _download_zip_action(self) -> None:
        """Alias para compatibilidade legada."""
        await self._on_download_zip_clicked()

    # -----------------------------------------------------------------------
    # Helpers Privados
    # -----------------------------------------------------------------------

    def _reset_controls(self) -> None:
        """Reabilita controles após término."""
        self._is_processing = False
        self._btn_start.disabled = False
        self._btn_start_top.disabled = False
        self._btn_input.disabled = False
        self._btn_select_files.disabled = False
        self._btn_sample_batch.disabled = False
        self._btn_output.disabled = False

    def _update_start_button_state(self) -> None:
        """Atualiza o estado de habilitação dos botões 'Iniciar'."""
        has_input = bool(self._web_file_data or self._selected_images or self._input_dir or self._queue_items)
        if not self._is_web and not self._output_dir:
            self._output_dir = ASSETS_DIR / "lote_resultado"
            self._output_label.value = str(self._output_dir)
            self._output_label.italic = False
            self._output_label.color = theme.TEXT_PRIMARY

        self._btn_start.disabled = not has_input
        self._btn_start_top.disabled = not has_input
        self._page.update()

    def _get_selected_technique(self) -> QuantizationTechnique:
        if isinstance(self._selected_technique, QuantizationTechnique):
            return self._selected_technique
        try:
            int_v = int(self._selected_technique)
            for t in QuantizationTechnique:
                if t.value == int_v:
                    return t
        except Exception:
            pass
        return QuantizationTechnique.UNIFORM

    def _get_selected_gray_method(self) -> GrayscaleMethod:
        return self._selected_gray_method

    def update_responsive_layout(
        self, width: float | None = None, height: float | None = None
    ) -> None:
        """Adapta o layout da aba de lote às dimensões da tela."""
        pass

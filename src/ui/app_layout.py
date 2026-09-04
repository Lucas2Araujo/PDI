"""
app_layout.py — Shell Unificado da Aplicação PDI.

Orquestra os módulos didáticos (Tons de Cinza, Quantização, Operações Binárias),
os slots de entrada de imagem/escalar (Slot A e B), a visualização central
(ImageCanvas / BatchQueue) e a telemetria em tempo real (TelemetryPanel).

Substitui o padrão legado de God Class por uma arquitetura desacoplada,
reativa e 100% responsiva (Mobile < 768px vs Desktop >= 768px).
"""

import asyncio
from pathlib import Path
import time
from typing import Any, Callable
import flet as ft
import numpy as np

from src.core.image_io import MAX_IMAGE_DIMENSION, open_and_downscale_image
from src.core.samples import SAMPLE_OPTIONS, get_sample_path, load_sample_bytes
from src.ui import theme
from src.ui.common import (
    _bytes_to_data_uri,
    _ndarray_to_png_bytes,
    _register_file_pickers,
)
from src.ui.components.batch_queue import BatchQueue, BatchQueueItem
from src.ui.components.image_canvas import DisplayMode, ImageCanvas
from src.ui.components.input_slot import InputSlot
from src.ui.components.telemetry_panel import TelemetryPanel
from src.ui.dialogs import open_inspector_dialog, open_zoom_dialog
from src.ui.modules.base_module import BasePDIModule
from src.ui.modules.binary_ops_module import BinaryOpsModule
from src.ui.modules.grayscale_module import GrayscaleModule
from src.ui.modules.quantize_module import QuantizeModule
from src.ui.state.session_state import (
    EVENT_IMAGE_A_CHANGED,
    EVENT_RESULT_CHANGED,
    SessionState,
    get_session_state,
)

APP_TITLE = "PDI — Studio Digital"
APP_VERSION = "0.5"


class AppLayout(ft.Container):
    """
    Layout principal do Studio PDI com suporte a modo individual e em lote,
    responsividade fluida mobile/desktop e orquestração de módulos didáticos.
    """

    def __init__(
        self,
        page: ft.Page | None = None,
        session_state: SessionState | None = None,
        **kwargs: Any,
    ) -> None:
        self._page = page
        self.session_state = session_state or get_session_state()
        self.is_batch_mode: bool = False
        self._current_width: float | None = getattr(page, "width", 1200) if page else 1200
        self._current_height: float | None = getattr(page, "height", 800) if page else 800

        # FilePickers adicionais para Lote e Exportação
        self._file_picker_batch = ft.FilePicker()
        self._file_picker_export = ft.FilePicker()
        if self._page is not None:
            _register_file_pickers(self._page, self._file_picker_batch, self._file_picker_export)

        # 1. Instanciação dos Módulos Didáticos
        self.grayscale_module = GrayscaleModule(on_change=self._on_module_param_changed)
        self.quantize_module = QuantizeModule(on_change=self._on_module_param_changed)
        self.binary_ops_module = BinaryOpsModule(
            on_scalar_mode_changed=self._on_scalar_mode_changed,
            on_change=self._on_module_param_changed,
        )

        self.modules: list[BasePDIModule] = [
            self.quantize_module,
            self.binary_ops_module,
        ]
        self.active_module_index: int = 0  # Padrão: QuantizeModule

        # 2. Componentes de Entrada (Slots A e B)
        self.slot_a = InputSlot(
            slot_id="A",
            label="Imagem Primária (Slot A)",
            session_state=self.session_state,
            page=self._page,
            on_change=self._on_slot_changed,
        )
        self.slot_b = InputSlot(
            slot_id="B",
            label="Segunda Imagem / Escalar (Slot B)",
            session_state=self.session_state,
            page=self._page,
            on_change=self._on_slot_changed,
            supports_scalar=True,
        )
        self.slot_b.visible = self.current_module.requires_second_input

        # 3. Componentes Centrais de Exibição
        self.canvas = ImageCanvas(
            session_state=self.session_state,
            page=self._page,
            default_mode=DisplayMode.RESULT_ONLY,
            on_download=self._on_canvas_download,
            on_promote=self.promote_result,
        )

        self.batch_queue = BatchQueue(
            page=self._page,
            on_clear=self._on_batch_cleared,
        )

        # 4. Componente de Telemetria
        self.telemetry = TelemetryPanel(
            session_state=self.session_state,
            page=self._page,
            on_open_inspector=self._on_open_inspector,
        )

        # 5. Controles do Cabeçalho e Seletores
        self._segmented_mode: ft.SegmentedButton | None = None
        self._dd_module: ft.Dropdown | None = None
        self._btn_execute: ft.Button | None = None
        self._btn_export: ft.Button | None = None
        self._btn_promote: ft.Button | None = None
        self._btn_undo: ft.IconButton | None = None
        self._btn_redo: ft.IconButton | None = None
        self._header_container: ft.Container | None = None

        # Inscrições reativas no SessionState para atualização de pipeline
        self._unsub_layout_res = self.session_state.subscribe(EVENT_RESULT_CHANGED, self._on_session_result_changed)
        self._unsub_layout_a = self.session_state.subscribe(EVENT_IMAGE_A_CHANGED, self._on_session_image_a_changed)

        # 6. Estruturas de Containers
        self._module_container = ft.Container(content=self.current_module)
        self._batch_toolbar = self._build_batch_toolbar()
        self._batch_workspace = ft.Column(
            controls=[
                self._batch_toolbar,
                self.batch_queue,
            ],
            spacing=10,
            expand=True,
            visible=False,
        )

        self._workspace_area = ft.Container(
            content=ft.Stack(
                controls=[
                    self.canvas,
                    self._batch_workspace,
                ],
                expand=True,
            ),
            expand=True,
        )

        self._sidebar_content = ft.Column(
            controls=[],
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self._sidebar_container = ft.Container(
            content=self._sidebar_content,
            width=360,
            padding=10,
        )

        self._telemetry_container = ft.Container(
            content=self.telemetry,
            width=290,
            padding=8,
        )

        self._main_body_layout = ft.Container(expand=True)

        self._root_column = ft.Column(
            controls=[],
            spacing=0,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        container_kwargs: dict[str, Any] = {
            "content": self._root_column,
            "expand": True,
            "padding": 0,
        }
        container_kwargs.update(kwargs)
        super().__init__(**container_kwargs)

        self._build_ui()
        self.handle_resize(self._current_width, self._current_height)

    # -----------------------------------------------------------------------
    # Utilitários de Atualização Segura
    # -----------------------------------------------------------------------

    def _safe_update(self) -> None:
        """Invoca update() apenas se montado na página."""
        try:
            self.update()
        except (RuntimeError, AssertionError):
            pass

    def _get_active_page(self) -> ft.Page | None:
        try:
            return self.page
        except (RuntimeError, Exception):
            return self._page

    def _show_message(self, message: str, color: str = theme.INFO) -> None:
        """Exibe notificação do tipo SnackBar."""
        page = self._get_active_page()
        if page is None:
            return
        try:
            snack = ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE, size=theme.FONT_BODY),
                bgcolor=color,
            )
            if hasattr(page, "open"):
                page.open(snack)
            elif hasattr(page, "show_snack_bar"):
                page.show_snack_bar(snack)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Propriedades e Estado do Módulo Ativo
    # -----------------------------------------------------------------------

    @property
    def current_module(self) -> BasePDIModule:
        """Retorna a instância do módulo didático atualmente selecionado."""
        return self.modules[self.active_module_index]

    def select_module(self, index: int) -> None:
        """Altera programaticamente o módulo didático ativo."""
        if 0 <= index < len(self.modules):
            self.active_module_index = index
            self._module_container.content = self.current_module
            self._update_slot_b_visibility()
            if self._dd_module:
                self._dd_module.value = str(index)
            self._safe_update()

    def _update_slot_b_visibility(self) -> None:
        """Sincroniza a visibilidade do Slot B com os requisitos do módulo ativo."""
        req = self.current_module.requires_second_input
        self.slot_b.visible = req
        self._safe_update()

    def _on_module_param_changed(self) -> None:
        """Callback acionado quando os parâmetros de qualquer módulo mudam."""
        self._update_slot_b_visibility()

    def _on_scalar_mode_changed(self, is_scalar: bool) -> None:
        """Callback acionado pelo BinaryOpsModule para alternar o Slot B entre imagem e escalar."""
        self.slot_b.set_scalar_mode(is_scalar)

    def _on_slot_changed(self) -> None:
        """Callback disparado quando Slot A ou Slot B recebem nova imagem."""
        self._safe_update()

    # -----------------------------------------------------------------------
    # Construção de Controles e Header
    # -----------------------------------------------------------------------

    def _build_batch_toolbar(self) -> ft.Container:
        """Cria a barra de ferramentas para o modo de processamento em lote."""
        btn_add_files = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ADD_PHOTO_ALTERNATE, size=18),
                    ft.Text("Adicionar Arquivos"),
                ],
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            on_click=lambda _: self._on_pick_batch_files(),
        )

        btn_load_samples = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.COLLECTIONS, size=18, color=theme.PRIMARY_LIGHT),
                    ft.Text("Carregar Amostras (5 Imagens)"),
                ],
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            on_click=lambda _: self._load_sample_batch(),
        )

        btn_clear = ft.IconButton(
            icon=ft.Icons.DELETE_SWEEP,
            tooltip="Esvaziar Fila de Lote",
            icon_color=theme.ACCENT,
            on_click=lambda _: self.batch_queue.clear(),
        )

        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Row(controls=[btn_add_files, btn_load_samples], spacing=8, wrap=True),
                    btn_clear,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=theme.BORDER_RADIUS,
            padding=8,
        )

    def _build_header(self) -> ft.Container:
        """Constrói o cabeçalho superior unificado com logo, título, ações e modo."""
        # 1. Identidade Visual
        logo_icon = ft.Container(
            content=ft.Icon(ft.Icons.AUTO_AWESOME_MOTION, size=24, color=theme.PRIMARY_LIGHT),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=8,
            padding=6,
        )

        title_text = ft.Text(
            APP_TITLE,
            size=theme.FONT_SUBTITLE,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.ON_SURFACE,
        )

        version_badge = ft.Container(
            content=ft.Text(f"v{APP_VERSION}", size=theme.FONT_CAPTION, color=theme.PRIMARY_LIGHT, weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=6,
            padding=ft.Padding.symmetric(horizontal=6, vertical=2) if hasattr(ft, "Padding") else 4,
        )

        brand_row = ft.Row(
            controls=[
                logo_icon,
                title_text,
                version_badge,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # 2. Alternador de Modo: Individual vs Em Lote
        self._segmented_mode = ft.SegmentedButton(
            segments=[
                ft.Segment(
                    value="single",
                    label=ft.Text("Individual", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.IMAGE, size=16),
                ),
                ft.Segment(
                    value="batch",
                    label=ft.Text("Em Lote", size=theme.FONT_CAPTION),
                    icon=ft.Icon(ft.Icons.BURST_MODE, size=16),
                ),
            ],
            selected=["single"],
            on_change=self._on_mode_toggled,
            show_selected_icon=False,
        )

        # 3. Botão de Execução Rápida (Emerald/Green)
        self._btn_execute = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.BOLT, size=20, color=ft.Colors.WHITE),
                    ft.Text("Executar Processamento", weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                ],
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            bgcolor=theme.SUCCESS,
            on_click=lambda _: self.run_processing(),
        )

        # 4. Ações de Pipeline e Histórico (Encadeamento e Undo/Redo)
        self._btn_promote = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.SUBDIRECTORY_ARROW_RIGHT, size=18, color=theme.PRIMARY_LIGHT),
                    ft.Text("Usar como Entrada A", size=theme.FONT_CAPTION, weight=ft.FontWeight.BOLD),
                ],
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            tooltip="Definir resultado atual como Entrada A para aplicar nova transformação em sequência (Pipeline)",
            disabled=(self.session_state.result_image is None),
            on_click=lambda _: self.promote_result(),
        )

        self._btn_undo = ft.IconButton(
            icon=ft.Icons.UNDO,
            tooltip="Desfazer etapa anterior (Undo)",
            disabled=not self.session_state.can_undo(),
            on_click=lambda _: self.undo(),
        )

        self._btn_redo = ft.IconButton(
            icon=ft.Icons.REDO,
            tooltip="Refazer etapa posterior (Redo)",
            disabled=not self.session_state.can_redo(),
            on_click=lambda _: self.redo(),
        )

        history_controls = ft.Row(
            controls=[
                self._btn_undo,
                self._btn_redo,
            ],
            spacing=2,
            tight=True,
        )

        # 5. Botão de Exportação
        self._btn_export = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.DOWNLOAD, size=18),
                    ft.Text("Exportar"),
                ],
                spacing=6,
            ),
            on_click=lambda _: self.export_result(),
        )

        def _on_theme_change(e: ft.ControlEvent) -> None:
            if not e.control.selected:
                return
            mode_str = next(iter(e.control.selected), "system")
            p = self._get_active_page()
            if p is not None:
                p.theme_mode = {"light": ft.ThemeMode.LIGHT, "dark": ft.ThemeMode.DARK}.get(mode_str, ft.ThemeMode.SYSTEM)
                p.update()

        theme_selector = ft.SegmentedButton(
            segments=[
                ft.Segment(value="system", label=ft.Text("Auto", size=theme.FONT_CAPTION), icon=ft.Icon(ft.Icons.BRIGHTNESS_AUTO, size=16)),
                ft.Segment(value="light", label=ft.Text("Claro", size=theme.FONT_CAPTION), icon=ft.Icon(ft.Icons.LIGHT_MODE, size=16)),
                ft.Segment(value="dark", label=ft.Text("Escuro", size=theme.FONT_CAPTION), icon=ft.Icon(ft.Icons.DARK_MODE, size=16)),
            ],
            selected=["system"],
            on_change=_on_theme_change,
            show_selected_icon=False,
        )

        self._header_container = ft.Container(
            content=ft.Row(
                controls=[
                    brand_row,
                    ft.Row(
                        controls=[
                            self._segmented_mode,
                            self._btn_execute,
                            self._btn_promote,
                            history_controls,
                            self._btn_export,
                            theme_selector,
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            padding=ft.Padding.symmetric(horizontal=14, vertical=10) if hasattr(ft, "Padding") else 10,
            border=ft.Border(bottom=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)) if hasattr(ft, "Border") else None,
        )
        return self._header_container

    def _build_sidebar_controls(self) -> None:
        """Monta o seletor de módulos e a lista de controles da barra lateral."""
        module_options = [
            ft.dropdown.Option(key="0", text="1. Quantização & Dithering"),
            ft.dropdown.Option(key="1", text="2. Operações Aritméticas & Lógicas"),
        ]

        self._dd_module = ft.Dropdown(
            label="Módulo da Ementa de PDI",
            value=str(self.active_module_index),
            options=module_options,
            content_padding=ft.Padding.symmetric(horizontal=14, vertical=12) if hasattr(ft, "Padding") else 12,
            border_radius=8,
            on_select=lambda e: self.select_module(int(e.control.value)),
        )

        inputs_header = ft.Row(
            controls=[
                ft.Icon(ft.Icons.IMAGE_SEARCH, size=18, color=theme.PRIMARY_LIGHT),
                ft.Text("Entradas de Imagem", size=theme.FONT_BODY, weight=ft.FontWeight.BOLD),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._sidebar_content.controls = [
            ft.Text("Configuração do Algoritmo", size=theme.FONT_BODY, weight=ft.FontWeight.BOLD),
            self._dd_module,
            self._module_container,
            ft.Divider(height=1),
            inputs_header,
            self.slot_a,
            self.slot_b,
        ]

    def _build_ui(self) -> None:
        """Monta a estrutura inicial da interface gráfica."""
        header = self._build_header()
        self._build_sidebar_controls()

        self._root_column.controls = [
            header,
            self._main_body_layout,
        ]

    # -----------------------------------------------------------------------
    # Alternância de Modo (Individual vs Lote)
    # -----------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Define o modo ativo ('single' ou 'batch')."""
        self.is_batch_mode = bool(mode == "batch")
        if self._segmented_mode:
            self._segmented_mode.selected = [mode]

        if self.is_batch_mode:
            self.canvas.visible = False
            self._batch_workspace.visible = True
            self.batch_queue.visible = True
        else:
            self.canvas.visible = True
            self._batch_workspace.visible = False
            self.batch_queue.visible = False

        if self._btn_promote is not None:
            self._btn_promote.visible = not self.is_batch_mode
        if self._btn_undo is not None:
            self._btn_undo.visible = not self.is_batch_mode
        if self._btn_redo is not None:
            self._btn_redo.visible = not self.is_batch_mode

        self._safe_update()

    def _on_mode_toggled(self, e: ft.ControlEvent) -> None:
        """Disparado quando o usuário clica no seletor de modo."""
        if not e.control.selected:
            return
        mode = next(iter(e.control.selected), "single")
        self.set_mode(mode)

    # -----------------------------------------------------------------------
    # Operações em Lote (Batch)
    # -----------------------------------------------------------------------

    def _load_sample_batch(self) -> None:
        """Carrega as 5 amostras padrão na fila de lote."""
        sample_items: list[BatchQueueItem] = []
        for s in SAMPLE_OPTIONS:
            name = s["name"]
            try:
                p = get_sample_path(name)
                b = load_sample_bytes(name, max_dim=MAX_IMAGE_DIMENSION)
                sample_items.append(BatchQueueItem(name=name, path=p, raw_bytes=b))
            except Exception:
                continue

        if not sample_items:
            self._show_message("Nenhuma amostra encontrada na pasta assets.", theme.WARNING)
            return

        self.batch_queue.set_items(sample_items)
        self.batch_queue.visible = True
        self._show_message(f"✅ {len(sample_items)} amostras adicionadas à fila!", theme.SUCCESS)

    def _on_pick_batch_files(self) -> None:
        """Dispara seleção de múltiplos arquivos de imagem para a fila de lote."""
        page = self._get_active_page()
        if page is None:
            return

        async def _pick_task():
            try:
                files = await self._file_picker_batch.pick_files(
                    dialog_title="Selecionar Imagens para Lote",
                    file_type=ft.FilePickerFileType.IMAGE,
                    allow_multiple=True,
                    with_data=True,
                )
                if not files:
                    return

                new_items: list[BatchQueueItem] = []
                for f in files:
                    p = Path(f.path) if f.path else None
                    new_items.append(BatchQueueItem(name=f.name, path=p, raw_bytes=f.bytes))

                current_items = self.batch_queue.items
                current_items.extend(new_items)
                self.batch_queue.set_items(current_items)
                self.batch_queue.visible = True
                self._show_message(f"✅ {len(new_items)} arquivos adicionados à fila de lote.", theme.SUCCESS)
            except Exception as exc:
                self._show_message(f"Erro ao selecionar arquivos: {exc}", theme.ACCENT)

        if hasattr(page, "run_task"):
            page.run_task(_pick_task)
        else:
            try:
                asyncio.create_task(_pick_task())
            except Exception:
                pass

    def _on_batch_cleared(self) -> None:
        """Callback disparado quando a fila de lote é esvaziada."""
        self._show_message("Fila de processamento em lote esvaziada.", theme.INFO)

    # -----------------------------------------------------------------------
    # Orquestração do Processamento (Pipeline Execution)
    # -----------------------------------------------------------------------

    def run_processing(self) -> None:
        """
        Executa o pipeline de processamento de acordo com o modo atual
        (Individual sobre Slot A/B ou Em Lote sobre os itens da fila).
        """
        if self.is_batch_mode:
            self._run_batch_processing()
        else:
            self._run_single_processing()

    def _run_single_processing(self) -> None:
        """Executa o processamento individual."""
        img_a = self.slot_a.image_array if self.slot_a.image_array is not None else self.session_state.image_a
        if img_a is None:
            self._show_message("Carregue ao menos uma imagem no Slot A para processar!", theme.WARNING)
            return

        params: dict[str, Any] = {}
        if self.current_module.requires_second_input:
            if self.slot_b.is_scalar_mode:
                params["scalar_val"] = self.slot_b.scalar_value
            else:
                img_b = self.slot_b.image_array if self.slot_b.image_array is not None else self.session_state.image_b
                if img_b is None:
                    self._show_message(
                        "O módulo ativo requer uma segunda imagem no Slot B ou ative o modo escalar!",
                        theme.WARNING,
                    )
                    return
                params["img_b"] = img_b

        try:
            res_array, metrics_dict = self.current_module.process(img_a, **params)
            out_name = f"{self.slot_a.image_name or 'imagem'}_resultado.png"
            self.session_state.set_result(
                res_array,
                metrics=metrics_dict,
                name=out_name,
                module_name=self.current_module.title,
            )
            self._show_message(
                f"✅ Processamento concluído em {metrics_dict.get('time_ms', 0):.1f} ms!",
                theme.SUCCESS,
            )
        except Exception as exc:
            self._show_message(f"Erro no processamento: {exc}", theme.ACCENT)

    def _run_batch_processing(self) -> None:
        """Executa o processamento sequencial da fila em lote."""
        items = self.batch_queue.items
        if not items:
            self._show_message("Nenhuma imagem na fila de lote para processar.", theme.WARNING)
            return

        total = len(items)
        self.batch_queue.set_progress(0.01, f"Iniciando lote com {total} imagens...")

        t_start = time.perf_counter()
        success_count = 0

        for idx, item in enumerate(items):
            self.batch_queue.update_item_status(idx, "⚙️ Processando...", theme.WARNING)
            try:
                arr = item.array
                if arr is None:
                    if item.raw_bytes is not None:
                        arr = open_and_downscale_image(item.raw_bytes, max_dim=MAX_IMAGE_DIMENSION)
                    elif item.path is not None and item.path.exists():
                        arr = open_and_downscale_image(item.path, max_dim=MAX_IMAGE_DIMENSION)

                if arr is None:
                    self.batch_queue.update_item_status(idx, "❌ Erro ao decodificar", theme.ACCENT)
                    continue

                params: dict[str, Any] = {}
                if self.current_module.requires_second_input:
                    if self.slot_b.is_scalar_mode:
                        params["scalar_val"] = self.slot_b.scalar_value
                    else:
                        img_b = self.slot_b.image_array if self.slot_b.image_array is not None else self.session_state.image_b
                        params["img_b"] = img_b

                res, metrics = self.current_module.process(arr, **params)
                t_ms = metrics.get("time_ms", 0.0)
                self.batch_queue.update_item_status(idx, f"✅ Concluído ({t_ms:.0f}ms)", theme.SUCCESS)
                success_count += 1
            except Exception as exc:
                self.batch_queue.update_item_status(idx, f"❌ Erro: {exc}", theme.ACCENT)

            prog = (idx + 1) / total
            self.batch_queue.set_progress(prog, f"{idx + 1}/{total} processados")

        total_elapsed = (time.perf_counter() - t_start) * 1000.0
        self._show_message(
            f"🎉 Lote concluído! {success_count}/{total} imagens processadas em {total_elapsed:.0f} ms.",
            theme.SUCCESS,
        )

    # -----------------------------------------------------------------------
    # Exportação e Auditoria
    # -----------------------------------------------------------------------

    def export_result(self) -> None:
        """Exporta o resultado processado."""
        if self.is_batch_mode:
            self._show_message("No modo lote, os resultados ficam disponíveis na fila.", theme.INFO)
            return

        res = self.session_state.result_image
        if res is None:
            self._show_message("Nenhum resultado processado para exportar!", theme.WARNING)
            return

        page = self._get_active_page()
        png_bytes = _ndarray_to_png_bytes(res)
        default_name = self.session_state.result_name or "resultado_pdi.png"

        if page is not None:
            async def _save_task():
                try:
                    save_path = await self._file_picker_export.save_file(
                        dialog_title="Salvar Imagem Processada",
                        file_name=default_name,
                        allowed_extensions=["png"],
                        src_bytes=png_bytes,
                    )
                    if save_path and not getattr(page, "web", False):
                        Path(save_path).write_bytes(png_bytes)
                    self._show_message(f"✅ Arquivo '{default_name}' exportado com sucesso!", theme.SUCCESS)
                except Exception as exc:
                    self._show_message(f"Erro ao salvar arquivo: {exc}", theme.ACCENT)

            if hasattr(page, "run_task"):
                page.run_task(_save_task)
            else:
                try:
                    asyncio.create_task(_save_task())
                except Exception:
                    self._show_message(f"Download iniciado: {default_name}", theme.SUCCESS)
        else:
            self._show_message(f"Download iniciado: {default_name}", theme.SUCCESS)

    def _on_canvas_download(self, png_bytes: bytes, name: str) -> None:
        """Callback de download direto do ImageCanvas."""
        self.export_result()

    def _on_open_inspector(self) -> None:
        """Abre o modal didático 'Entranhas do Processo' com auditoria matemática."""
        page = self._get_active_page()
        if page is None:
            return

        res_img = self.session_state.result_image
        raw_img = self.slot_a.image_array if self.slot_a.image_array is not None else self.session_state.image_a
        if res_img is None or raw_img is None:
            self._show_message(
                "Processe uma imagem primeiro para inspecionar as entranhas do algoritmo.",
                theme.WARNING,
            )
            return

        if isinstance(self.current_module, QuantizeModule):
            params = self.current_module.get_params()
            bits = params.get("bits", 4)
            tech = params.get("technique", "UNIFORM")
            from src.core.grayscale import GrayscaleMethod, to_grayscale
            gray = to_grayscale(raw_img, GrayscaleMethod.LUMINANCE)
            open_inspector_dialog(
                page,
                raw_image=raw_img,
                gray_image=gray,
                quantized_image=res_img,
                bits=bits,
                technique=tech,
                method=GrayscaleMethod.LUMINANCE,
            )
        elif isinstance(self.current_module, GrayscaleModule):
            params = self.current_module.get_params()
            method = params.get("method")
            from src.core.grayscale import to_grayscale
            gray = to_grayscale(raw_img, method=method) if raw_img.ndim == 3 else raw_img
            open_inspector_dialog(
                page,
                raw_image=raw_img,
                gray_image=gray,
                quantized_image=res_img,
                bits=8,
                technique="UNIFORM",
                method=method,
            )
        else:
            open_zoom_dialog(
                page,
                title="Inspeção Visual do Resultado",
                image_bytes=_ndarray_to_png_bytes(res_img),
            )

    # -----------------------------------------------------------------------
    # Ações de Pipeline e Histórico (Encadeamento e Undo/Redo)
    # -----------------------------------------------------------------------

    def promote_result(self) -> bool:
        """
        Promove o resultado atual para ser a nova Imagem Primária (Slot A),
        permitindo o encadeamento de transformações (Pipeline de Composição).

        Returns:
            True se a promoção ocorreu com sucesso, False se não havia resultado.
        """
        if self.session_state.promote_result_to_input_a():
            self._show_message("Resultado aplicado como Entrada A para encadeamento!", theme.SUCCESS)
            self._update_pipeline_controls()
            return True
        return False

    def undo(self) -> bool:
        """
        Desfaz a última transformação aplicada no pipeline.

        Returns:
            True se desfeito com sucesso, False se o histórico estiver vazio.
        """
        if self.session_state.undo():
            self._show_message("Etapa desfeita.", theme.INFO)
            self._update_pipeline_controls()
            return True
        return False

    def redo(self) -> bool:
        """
        Refaz a etapa desfeita do pipeline.

        Returns:
            True se refeito com sucesso, False se não houver etapas futuras.
        """
        if self.session_state.redo():
            self._show_message("Etapa refeita.", theme.INFO)
            self._update_pipeline_controls()
            return True
        return False

    def _update_pipeline_controls(self) -> None:
        """Sincroniza os estados habilitado/desabilitado dos controles de pipeline."""
        if self._btn_promote is not None:
            self._btn_promote.disabled = (self.session_state.result_image is None)
        if self._btn_undo is not None:
            self._btn_undo.disabled = not self.session_state.can_undo()
        if self._btn_redo is not None:
            self._btn_redo.disabled = not self.session_state.can_redo()
        self._safe_update()

    def _on_session_result_changed(self, **_: Any) -> None:
        """Atualiza os controles de pipeline quando o resultado é alterado."""
        self._update_pipeline_controls()

    def _on_session_image_a_changed(self, **_: Any) -> None:
        """Atualiza os controles de pipeline quando a Imagem A é alterada."""
        self._update_pipeline_controls()

    # -----------------------------------------------------------------------
    # Responsividade Adaptativa (Desktop 3-Colunas vs Mobile 1-Coluna)
    # -----------------------------------------------------------------------

    def handle_resize(self, width: float | None, height: float | None) -> None:
        """
        Reconfigura a distribuição espacial dos componentes de acordo com a largura da tela.

        - width < 768px: Layout Mobile em coluna única, encapsulando módulos e telemetria
          em ExpansionTiles para priorizar a visualização do Canvas.
        - width >= 768px: Layout Estúdio Desktop com 3 colunas independentes:
          [Sidebar 320px] | [Workspace Central expand=True] | [Telemetria 290px].
        """
        self._current_width = width
        self._current_height = height
        is_mobile = bool(width is not None and width < 768)

        # Repassa redimensionamento para os componentes filhos
        self.canvas.update_responsive_layout(width, height)
        self.telemetry.update_responsive_layout(width, height)

        if is_mobile:
            # Layout Mobile: Coluna única fluida com ExpansionTiles
            self._sidebar_container.width = None
            self._telemetry_container.width = None

            mobile_module_tile = ft.ExpansionTile(
                title=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.TUNE, size=18, color=theme.PRIMARY_LIGHT),
                        ft.Text("Módulo & Controles Didáticos", weight=ft.FontWeight.BOLD),
                    ],
                    spacing=6,
                ),
                controls=[
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                self._dd_module,
                                self._module_container,
                            ],
                            spacing=10,
                        ),
                        padding=10,
                    )
                ],
                expanded=False,
            )

            mobile_inputs_tile = ft.ExpansionTile(
                title=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.IMAGE_SEARCH, size=18, color=theme.PRIMARY_LIGHT),
                        ft.Text("Imagens de Entrada (Slots)", weight=ft.FontWeight.BOLD),
                    ],
                    spacing=6,
                ),
                controls=[
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                self.slot_a,
                                self.slot_b,
                            ],
                            spacing=10,
                        ),
                        padding=10,
                    )
                ],
                expanded=False,
            )

            mobile_content = ft.Column(
                controls=[
                    ft.Container(
                        content=self._workspace_area,
                        height=420,
                        padding=8,
                    ),
                    mobile_module_tile,
                    mobile_inputs_tile,
                    ft.Container(
                        content=self.telemetry,
                        padding=8,
                    ),
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=8,
                expand=True,
            )
            self._main_body_layout.content = mobile_content
        else:
            # Layout Desktop: Sidebar fixa de 360px + Workspace expand=True (sem sidebar de telemetria)
            self._sidebar_container.width = 360
            self._build_sidebar_controls()

            desktop_content = ft.Row(
                controls=[
                    self._sidebar_container,
                    self._workspace_area,
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                expand=True,
            )
            self._main_body_layout.content = desktop_content

        self._safe_update()

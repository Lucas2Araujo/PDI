"""
input_slot.py — Componente de Entrada de Imagem / Escalar para Slots A e B.

Oferece área de carregamento responsiva com suporte a:
- Seleção direta de amostras clássicas de PDI (Retrato, Benchmark, Lena, Ayla, Pentágono).
- Upload via FilePicker compatível com Desktop e WebAssembly (with_data=True).
- Estados visuais dedicados: Vazio (área pontilhada) vs Carregado (miniatura, metadados e zoom).
- Alternância para Modo Escalar (constante numérica para operações como A + 50).
- Integração reativa bidirecional com o SessionState.
"""

from typing import Any, Callable
from pathlib import Path
import asyncio
import flet as ft
import numpy as np

from src.core.grayscale import GrayscaleMethod, to_grayscale
from src.core.image_io import (
    MAX_IMAGE_DIMENSION,
    MAX_THUMBNAIL_DIMENSION,
    make_thumbnail_png,
    open_and_downscale_image,
)
from src.core.samples import SAMPLE_OPTIONS, load_sample_array
from src.ui import theme
from src.ui.common import (
    _bytes_to_data_uri,
    _ndarray_to_png_bytes,
    _register_file_pickers,
)
from src.ui.dialogs import open_zoom_dialog
from src.ui.state.session_state import (
    EVENT_IMAGE_A_CHANGED,
    EVENT_IMAGE_B_CHANGED,
    SessionState,
    get_session_state,
)


class InputSlot(ft.Container):
    """
    Componente visual que gerencia a seleção, visualização e metadados de uma imagem de entrada
    ou valor escalar associado a um Slot (A ou B).
    """

    def __init__(
        self,
        slot_id: str = "A",
        label: str = "Imagem de Entrada",
        session_state: SessionState | None = None,
        page: ft.Page | None = None,
        on_change: Callable[[], None] | None = None,
        supports_scalar: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o InputSlot.

        Args:
            slot_id: Identificador do slot ("A" ou "B").
            label: Rótulo de exibição (ex: "Imagem Primária (Slot A)").
            session_state: Instância do SessionState (padrão: singleton global).
            page: Página Flet para registro do FilePicker e modais.
            on_change: Callback acionado ao alterar imagem ou valor escalar.
            supports_scalar: Se permite alternar para entrada de constante numérica.
            **kwargs: Parâmetros repassados ao ft.Container.
        """
        self.slot_id = slot_id.upper()
        self.label = label
        self.session_state = session_state or get_session_state()
        self._page = page
        self.on_change = on_change
        self.supports_scalar = supports_scalar

        # Estado interno
        self._image_array: np.ndarray | None = None
        self._image_name: str = ""
        self._thumb_uri: str | None = None
        self._is_scalar_mode: bool = False
        self._scalar_value: float = 50.0

        # FilePickers integrados (Upload e Salvamento)
        self._file_picker = ft.FilePicker()
        self._save_picker = ft.FilePicker()
        if self._page is not None:
            _register_file_pickers(self._page, self._file_picker, self._save_picker)

        # Inscrição reativa no SessionState
        self._event_name = EVENT_IMAGE_A_CHANGED if self.slot_id == "A" else EVENT_IMAGE_B_CHANGED
        self._unsub_state = self.session_state.subscribe(self._event_name, self._on_session_image_changed)

        # Construção da UI
        self._content_column = ft.Column(
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        container_kwargs: dict[str, Any] = {
            "bgcolor": ft.Colors.SURFACE_CONTAINER,
            "border_radius": theme.BORDER_RADIUS,
            "padding": 12,
            "content": self._content_column,
        }
        container_kwargs.update(kwargs)
        super().__init__(**container_kwargs)

        self._render_state()

    # -----------------------------------------------------------------------
    # Utilitários de Atualização Segura
    # -----------------------------------------------------------------------

    def _safe_update(self) -> None:
        """Executa update() apenas se o controle estiver montado na página."""
        try:
            self.update()
        except (RuntimeError, AssertionError):
            pass

    def _get_active_page(self) -> ft.Page | None:
        try:
            return self.page
        except (RuntimeError, Exception):
            return self._page

    # -----------------------------------------------------------------------
    # Propriedades Públicas
    # -----------------------------------------------------------------------

    @property
    def image_array(self) -> np.ndarray | None:
        """Retorna o array da imagem atualmente carregada no slot."""
        return self._image_array

    @property
    def image_name(self) -> str:
        """Retorna o nome amigável da imagem carregada."""
        return self._image_name

    @property
    def is_scalar_mode(self) -> bool:
        """Indica se o slot está em modo de valor escalar numérico."""
        return self._is_scalar_mode

    @property
    def scalar_value(self) -> float:
        """Retorna o valor escalar configurado no slot."""
        return self._scalar_value

    def set_scalar_mode(self, enabled: bool, default_val: float | None = None) -> None:
        """Ativa ou desativa o modo escalar."""
        self._is_scalar_mode = bool(enabled)
        if default_val is not None:
            self._scalar_value = float(default_val)
        self._render_state()
        self._safe_update()

    def set_image(self, image: np.ndarray | None, name: str = "", sync_session: bool = True) -> None:
        """Define a imagem do slot programaticamente."""
        self._image_array = image
        self._image_name = name
        self._is_scalar_mode = False

        if image is not None:
            thumb_bytes = make_thumbnail_png(image, max_size=MAX_THUMBNAIL_DIMENSION)
            self._thumb_uri = _bytes_to_data_uri(thumb_bytes)
        else:
            self._thumb_uri = None

        if sync_session:
            if self.slot_id == "A":
                self.session_state.set_image_a(image, name)
            else:
                self.session_state.set_image_b(image, name)

        self._render_state()
        self._safe_update()

        if self.on_change:
            self.on_change()

    def clear(self, sync_session: bool = True) -> None:
        """Limpa o slot, descartando imagens ou resetando o escalar."""
        if self.slot_id == "A":
            self.session_state.clear_history()
        self.set_image(None, "", sync_session=sync_session)

    # -----------------------------------------------------------------------
    # Ciclo de Vida e Eventos
    # -----------------------------------------------------------------------

    def _on_session_image_changed(self, image: np.ndarray | None = None, name: str = "", **_: Any) -> None:
        """Atualiza a visão do slot quando o SessionState muda externamente."""
        if self._is_scalar_mode:
            return
        if self._image_array is not image or self._image_name != name:
            self._image_array = image
            self._image_name = name
            if image is not None:
                thumb_bytes = make_thumbnail_png(image, max_size=MAX_THUMBNAIL_DIMENSION)
                self._thumb_uri = _bytes_to_data_uri(thumb_bytes)
            else:
                self._thumb_uri = None
            self._render_state()
            self._safe_update()

    async def _trigger_picker(self, _: ft.ControlEvent | None = None) -> None:
        """Aciona o FilePicker para seleção de imagem (Web e Desktop)."""
        page = self._get_active_page()
        if page is not None:
            _register_file_pickers(page, self._file_picker)

        try:
            files = await self._file_picker.pick_files(
                dialog_title=f"Selecionar Imagem para Slot {self.slot_id}",
                allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                allow_multiple=False,
                with_data=True,
            )
        except Exception as exc:
            if page is not None:
                page.open(ft.SnackBar(content=ft.Text(f"Erro ao abrir seletor: {exc}"), bgcolor=theme.ACCENT))
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
            if self.slot_id == "A":
                self.session_state.clear_history()
            self.set_image(img_arr, file_name, sync_session=True)
        except Exception as exc:
            if page is not None:
                page.open(ft.SnackBar(content=ft.Text(f"Erro ao decodificar imagem: {exc}"), bgcolor=theme.ACCENT))

    def _on_select_sample(self, sample_id: str) -> None:
        """Carrega uma amostra didática clássica pelo identificador."""
        for opt in SAMPLE_OPTIONS:
            if opt["id"] == sample_id:
                arr = load_sample_array(opt["name"], max_dim=MAX_IMAGE_DIMENSION)
                if self.slot_id == "A":
                    self.session_state.clear_history()
                self.set_image(arr, opt["title"], sync_session=True)
                break

    def _open_zoom(self) -> None:
        """Abre o modal de zoom de alta definição para a imagem carregada."""
        page = self._get_active_page()
        if page is not None and self._image_array is not None:
            png_bytes = _ndarray_to_png_bytes(self._image_array)
            open_zoom_dialog(page, title=f"Slot {self.slot_id} — {self._image_name}", image_bytes=png_bytes)

    def _save_loaded_as_grayscale(self) -> None:
        """Converte a imagem atualmente carregada para escala de cinza (8 bits) e salva."""
        if self._image_array is None:
            return

        page = self._get_active_page()
        if page is not None:
            _register_file_pickers(page, self._file_picker, self._save_picker)

        if self._image_array.ndim == 3:
            gray = to_grayscale(self._image_array, method=GrayscaleMethod.LUMINANCE)
        else:
            gray = self._image_array.copy()

        png_bytes = _ndarray_to_png_bytes(gray)
        stem = Path(self._image_name).stem if self._image_name else f"slot_{self.slot_id.lower()}"
        default_name = f"{stem}_cinza_8bits.png"

        async def _task():
            try:
                save_path = await self._save_picker.save_file(
                    dialog_title="Salvar Carregada em Tons de Cinza (8 bits)",
                    file_name=default_name,
                    allowed_extensions=["png"],
                    src_bytes=png_bytes,
                )
                if save_path and not getattr(page, "web", False):
                    Path(save_path).write_bytes(png_bytes)
                if page is not None:
                    page.open(ft.SnackBar(content=ft.Text(f"✅ Salva em Tons de Cinza (8 bits): {default_name}"), bgcolor=theme.SUCCESS))
            except Exception as exc:
                if page is not None:
                    page.open(ft.SnackBar(content=ft.Text(f"Erro ao salvar: {exc}"), bgcolor=theme.ACCENT))

        if page is not None and hasattr(page, "run_task"):
            page.run_task(_task)
        else:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_task())
            except RuntimeError:
                pass

    # -----------------------------------------------------------------------
    # Renderização de Estados Visuais
    # -----------------------------------------------------------------------

    def _render_state(self) -> None:
        """Renderiza o conteúdo interno de acordo com o estado ativo."""
        self._content_column.controls.clear()

        # Cabeçalho do Slot com ID e Label
        header_controls: list[ft.Control] = [
            ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Text(
                            self.slot_id,
                            size=theme.FONT_CAPTION,
                            weight=ft.FontWeight.BOLD,
                            color="#FFFFFF",
                        ),
                        bgcolor=theme.PRIMARY if self.slot_id == "A" else theme.INFO,
                        border_radius=4,
                        padding=ft.Padding.symmetric(horizontal=6, vertical=2) if hasattr(ft, "Padding") else 4,
                    ),
                    ft.Text(
                        self.label,
                        size=theme.FONT_SUBTITLE,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.ON_SURFACE,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ]

        # Feedback visual de composição: badge discreto no topo do Slot A se histórico > 0
        if self.slot_id == "A" and self.session_state.history_count > 0:
            step_num = self.session_state.history_count + 1
            pipeline_badge = ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.ACCOUNT_TREE, size=13, color=theme.SUCCESS),
                        ft.Text(f"Etapa {step_num}", size=11, weight=ft.FontWeight.BOLD, color=theme.SUCCESS),
                    ],
                    spacing=4,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                border_radius=6,
                padding=ft.Padding.symmetric(horizontal=6, vertical=2) if hasattr(ft, "Padding") else 4,
                tooltip=f"Pipeline Composto — Etapa {step_num}",
            )
            header_controls.append(pipeline_badge)

        header_row = ft.Row(
            controls=header_controls,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self._content_column.controls.append(header_row)

        if self._is_scalar_mode:
            self._render_scalar_state()
        elif self._image_array is not None:
            self._render_loaded_state()
        else:
            self._render_empty_state()

    def _render_empty_state(self) -> None:
        """Renderiza o card de estado vazio com botões de upload e amostras."""
        # Menu Popup compacto de amostras didáticas
        sample_menu_items = [
            ft.PopupMenuItem(
                content=opt["button_label"],
                icon=getattr(ft.Icons, opt["icon"], ft.Icons.IMAGE),
                on_click=lambda _, sid=opt["id"]: self._on_select_sample(sid),
            )
            for opt in SAMPLE_OPTIONS
        ]

        sample_btn = ft.PopupMenuButton(
            icon=ft.Icons.COLLECTIONS,
            tooltip="Carregar Amostra Didática Clássica",
            items=sample_menu_items,
        )

        upload_btn = ft.Button(
            content="Escolher Arquivo",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self._trigger_picker,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )

        empty_box = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(ft.Icons.ADD_PHOTO_ALTERNATE_OUTLINED, size=32, color=theme.TEXT_SECONDARY),
                    ft.Text(
                        "Nenhuma imagem selecionada",
                        size=theme.FONT_CAPTION,
                        color=theme.TEXT_SECONDARY,
                        italic=True,
                    ),
                    ft.Row(
                        controls=[upload_btn, sample_btn],
                        spacing=8,
                        alignment=ft.MainAxisAlignment.CENTER,
                        wrap=True,
                    ),
                ],
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
            border_radius=8,
            padding=12,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
        )
        self._content_column.controls.append(empty_box)

    def _render_loaded_state(self) -> None:
        """Renderiza o estado com miniatura, badges de dimensão e ações."""
        if self._image_array is None:
            return

        h, w = self._image_array.shape[:2]
        is_color = bool(self._image_array.ndim == 3 and self._image_array.shape[2] >= 3)
        bpp_text = "RGB (24 bpp)" if is_color else "Cinza (8 bpp)"

        thumb_img = ft.Image(
            src=self._thumb_uri or "",
            width=72,
            height=72,
            fit=getattr(ft.BoxFit, "COVER", None) if hasattr(ft, "BoxFit") else None,
            border_radius=6,
        )

        thumb_container = ft.Container(
            content=thumb_img,
            on_click=lambda _: self._open_zoom(),
            tooltip="Clique para ampliar",
            ink=True,
            border_radius=6,
        )

        info_col = ft.Column(
            controls=[
                ft.Text(
                    self._image_name or "Imagem Carregada",
                    weight=ft.FontWeight.BOLD,
                    size=theme.FONT_BODY,
                    color=ft.Colors.ON_SURFACE,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Row(
                    controls=[
                        ft.Container(
                            content=ft.Text(f"{w}×{h} px", size=11, weight=ft.FontWeight.BOLD),
                            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            padding=ft.Padding.symmetric(horizontal=6, vertical=2) if hasattr(ft, "Padding") else 4,
                            border_radius=4,
                        ),
                        ft.Container(
                            content=ft.Text(bpp_text, size=11, color=theme.TEXT_SECONDARY),
                            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            padding=ft.Padding.symmetric(horizontal=6, vertical=2) if hasattr(ft, "Padding") else 4,
                            border_radius=4,
                        ),
                    ],
                    spacing=6,
                    wrap=True,
                ),
            ],
            spacing=4,
            expand=True,
        )

        actions_row = ft.Row(
            controls=[
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_size=20,
                    icon_color=theme.ACCENT,
                    tooltip="Remover Imagem deste Slot",
                    on_click=lambda _: self.clear(sync_session=True),
                ),
            ],
            spacing=2,
            tight=True,
        )

        btn_save_gray = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.DOWNLOAD_FOR_OFFLINE, size=15, color=theme.PRIMARY_LIGHT),
                    ft.Text("Salvar como Tons de Cinza (8 bits)", size=11, weight=ft.FontWeight.W_500),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=6,
            ),
            tooltip="Converte e exporta esta imagem de entrada em Tons de Cinza de 8 bits (ITU-R BT.601)",
            on_click=lambda _: self._save_loaded_as_grayscale(),
        )

        loaded_card = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            thumb_container,
                            info_col,
                            actions_row,
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    btn_save_gray,
                ],
                spacing=6,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=8,
        )
        self._content_column.controls.append(loaded_card)

    def _render_scalar_state(self) -> None:
        """Renderiza controles numéricos quando em modo escalar."""
        slider = ft.Slider(
            min=0,
            max=255,
            value=self._scalar_value,
            divisions=255,
            label="{value}",
            expand=True,
            on_change=self._on_slider_changed,
        )

        value_text = ft.Text(
            f"{int(self._scalar_value)}",
            size=theme.FONT_BODY,
            weight=ft.FontWeight.BOLD,
            color=theme.PRIMARY_LIGHT,
            width=40,
            text_align=ft.TextAlign.RIGHT,
        )
        self._scalar_val_text = value_text

        scalar_box = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Constante Escalar Numérica (C):", size=theme.FONT_CAPTION, color=theme.TEXT_SECONDARY),
                    ft.Row(
                        controls=[slider, value_text],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=4,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=10,
        )
        self._content_column.controls.append(scalar_box)

    def _on_slider_changed(self, e: ft.ControlEvent) -> None:
        """Manipula alteração de valor no slider escalar."""
        self._scalar_value = float(e.control.value)
        if hasattr(self, "_scalar_val_text"):
            self._scalar_val_text.value = f"{int(self._scalar_value)}"
            try:
                self._scalar_val_text.update()
            except (RuntimeError, AssertionError):
                pass
        if self.on_change:
            self.on_change()


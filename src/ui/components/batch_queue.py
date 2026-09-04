"""
batch_queue.py — Componente Modular da Fila de Processamento em Lote.

Gerencia a exibição visual da fila de imagens para processamento em lote:
- Cards responsivos e fluidos (eliminação total de larguras fixas rígidas como width=260).
- Miniaturas com abertura de zoom individual de alta definição via open_zoom_dialog.
- Atualização em tempo real de status por imagem (⏳ Pronto, ⚙️ Processando, ✅ Concluído, ❌ Erro).
- Barra de progresso integrada e contador dinâmico de itens.
"""

from pathlib import Path
from typing import Any, Callable
import flet as ft
import numpy as np

from src.core.image_io import (
    MAX_IMAGE_DIMENSION,
    MAX_THUMBNAIL_DIMENSION,
    make_thumbnail_png,
    open_and_downscale_image,
)
from src.ui import theme
from src.ui.common import _bytes_to_data_uri, _ndarray_to_png_bytes
from src.ui.dialogs import open_zoom_dialog


class BatchQueueItem:
    """Representação de um item carregado na fila de processamento em lote."""

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

        self._inspect()

    def _inspect(self) -> None:
        """Extrai dimensões, tipo de cor e miniatura PNG."""
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
        """Retorna os bytes PNG em alta resolução sob demanda."""
        if self.array is not None:
            return _ndarray_to_png_bytes(self.array)
        if self.raw_bytes is not None:
            return self.raw_bytes
        if self.path is not None and self.path.exists():
            return self.path.read_bytes()
        return None


class BatchQueue(ft.Container):
    """
    Componente visual da fila de imagens em lote com grid adaptativo e telemetria de progresso.
    """

    def __init__(
        self,
        page: ft.Page | None = None,
        on_clear: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o BatchQueue.

        Args:
            page: Instância ativa da página Flet.
            on_clear: Callback executado ao esvaziar a fila.
            **kwargs: Parâmetros repassados ao ft.Container.
        """
        self._page = page
        self.on_clear = on_clear
        self._items: list[BatchQueueItem] = []

        # Controles internos de progresso
        self._progress_bar = ft.ProgressBar(value=0.0, visible=False, color=theme.PRIMARY_LIGHT)
        self._count_badge = ft.Text("0 itens na fila", size=theme.FONT_CAPTION, color=theme.TEXT_SECONDARY)
        self._grid_container = ft.Row(
            spacing=8,
            wrap=True,
            alignment=ft.MainAxisAlignment.START,
        )

        # Cabeçalho da fila
        header = ft.Row(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.QUEUE, size=20, color=theme.PRIMARY_LIGHT),
                        ft.Text("Fila de Processamento", size=theme.FONT_SUBTITLE, weight=ft.FontWeight.BOLD),
                        self._count_badge,
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.IconButton(
                    icon=ft.Icons.DELETE_SWEEP,
                    tooltip="Limpar Fila",
                    icon_color=theme.ACCENT,
                    on_click=lambda _: self.clear(),
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._content_col = ft.Column(
            controls=[
                header,
                self._progress_bar,
                self._grid_container,
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        container_kwargs: dict[str, Any] = {
            "bgcolor": ft.Colors.SURFACE_CONTAINER,
            "border_radius": theme.BORDER_RADIUS,
            "padding": 12,
            "visible": False,
            "content": self._content_col,
        }
        container_kwargs.update(kwargs)
        super().__init__(**container_kwargs)

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

    # -----------------------------------------------------------------------
    # Gerenciamento de Itens
    # -----------------------------------------------------------------------

    @property
    def items(self) -> list[BatchQueueItem]:
        """Retorna a lista atual de itens na fila."""
        return list(self._items)

    def set_items(self, items: list[BatchQueueItem]) -> None:
        """Substitui os itens da fila e renderiza a grade."""
        self._items = list(items)
        self._render_queue()
        self._safe_update()

    def add_item(self, item: BatchQueueItem) -> None:
        """Adiciona um item à fila."""
        self._items.append(item)
        self._render_queue()
        self._safe_update()

    def clear(self) -> None:
        """Esvazia todos os itens da fila e oculta o container."""
        self._items.clear()
        self._render_queue()
        if self.on_clear:
            self.on_clear()
        self._safe_update()

    def update_item_status(self, index: int, status: str, color: str = theme.INFO) -> None:
        """Atualiza o status de um item específico por índice."""
        if 0 <= index < len(self._items):
            item = self._items[index]
            item.status = status
            item.status_color = color
            self._render_queue()
            self._safe_update()

    def update_progress(self, current: int, total: int) -> None:
        """Atualiza a barra de progresso da fila."""
        if total <= 0:
            self._progress_bar.visible = False
        else:
            self._progress_bar.visible = True
            self._progress_bar.value = max(0.0, min(1.0, current / total))
        self._safe_update()

    def set_progress(self, value: float, text: str = "") -> None:
        """Define o valor de progresso diretamente entre 0.0 e 1.0."""
        self._progress_bar.visible = True
        self._progress_bar.value = max(0.0, min(1.0, float(value)))
        if text:
            self._count_badge.value = text
        self._safe_update()

    # -----------------------------------------------------------------------
    # Renderização da Grade
    # -----------------------------------------------------------------------

    def _open_zoom(self, item: BatchQueueItem) -> None:
        page = self._get_active_page()
        png_data = item.get_full_png_bytes()
        if page is not None and png_data is not None:
            open_zoom_dialog(page, title=f"Fila — {item.name}", image_bytes=png_data)

    def _render_queue(self) -> None:
        """Gera os cards responsivos na grade."""
        self._grid_container.controls.clear()

        count = len(self._items)
        self.visible = count > 0
        self._count_badge.value = f"{count} item{'s' if count != 1 else ''} na fila"

        for item in self._items:
            img_ctrl = ft.Image(
                src=_bytes_to_data_uri(item.thumb_bytes),
                width=64,
                height=64,
                fit=getattr(ft.BoxFit, "COVER", None) if hasattr(ft, "BoxFit") else None,
                border_radius=6,
            )

            thumb_btn = ft.Container(
                content=img_ctrl,
                on_click=lambda _, it=item: self._open_zoom(it),
                tooltip="Clique para ampliar",
                ink=True,
                border_radius=6,
            )

            status_badge = ft.Text(
                item.status,
                size=11,
                weight=ft.FontWeight.BOLD,
                color=item.status_color,
            )

            btn_zoom = ft.IconButton(
                icon=ft.Icons.ZOOM_IN,
                icon_size=18,
                tooltip=f"Ampliar {item.name}",
                on_click=lambda _, it=item: self._open_zoom(it),
            )

            # Card fluido e sem largura fixa rígida (> 300)
            item_card = ft.Container(
                content=ft.Row(
                    controls=[
                        thumb_btn,
                        ft.Column(
                            controls=[
                                ft.Text(
                                    item.name,
                                    weight=ft.FontWeight.BOLD,
                                    size=12,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                    max_lines=1,
                                ),
                                ft.Text(
                                    f"{item.dimensions} • {item.color_type}",
                                    size=10,
                                    color=theme.TEXT_SECONDARY,
                                ),
                                ft.Row(
                                    controls=[status_badge, btn_zoom],
                                    spacing=2,
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                ),
                            ],
                            spacing=2,
                            expand=True,
                        ),
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                border_radius=8,
                padding=8,
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
                width=240,
            )

            self._grid_container.controls.append(item_card)

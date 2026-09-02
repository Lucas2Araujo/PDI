"""
app.py — Aplicação Principal Flet do Projeto PDI.

Define a estrutura raiz da interface:
  - Barra de título com nome do projeto e badge de versão.
  - Sistema de abas (Processamento Individual | Processamento em Lote).
  - Configurações globais de tema (Dark Mode padrão).

Ponto de entrada da UI chamado por main.py.
"""

from pathlib import Path
from typing import Callable

import flet as ft

from src.ui import theme
from src.ui.views.batch_view import BatchView
from src.ui.views.single_view import SingleView

# Versão atual do aplicativo
APP_VERSION = "0.4"
APP_TITLE = "PDI — Quantização de Imagens"


def build_app(page: ft.Page) -> None:
    """
    Configura e constrói toda a interface gráfica do aplicativo.

    Este é o callback principal passado para `ft.app()`. Ele recebe a
    instância de `ft.Page` e é responsável por definir o tema, o layout
    e todos os componentes de nível raiz.

    Args:
        page: Instância da página Flet gerenciada pelo framework.
    """
    _configure_page(page)

    single_view = SingleView(page)
    batch_view = BatchView(page)

    t1 = ft.Tab(label="Imagem Individual", icon=ft.Icons.IMAGE)
    t2 = ft.Tab(label="Processamento em Lote", icon=ft.Icons.BURST_MODE)

    p_w = getattr(page, "width", None)
    p_h = getattr(page, "height", None)

    v1 = ft.Container(
        content=single_view,
        padding=theme.get_page_padding(p_w),
        expand=True,
    )
    v2 = ft.Container(
        content=batch_view,
        padding=theme.get_page_padding(p_w),
        expand=True,
    )

    tab_bar = ft.TabBar(tabs=[t1, t2])
    tab_view = ft.TabBarView(controls=[v1, v2], expand=True)

    tabs = ft.Tabs(
        length=2,
        selected_index=0,
        expand=True,
        content=ft.Column(
            controls=[tab_bar, tab_view],
            expand=True,
        ),
    )

    header_container, update_header_layout = _build_header(page)

    def _on_page_resize(_: ft.ControlEvent) -> None:
        cur_w = getattr(page, "width", None)
        cur_h = getattr(page, "height", None)
        pad = theme.get_page_padding(cur_w)
        v1.padding = pad
        v2.padding = pad
        update_header_layout(cur_w)
        if hasattr(single_view, "update_responsive_layout"):
            single_view.update_responsive_layout(cur_w, cur_h)
        if hasattr(batch_view, "update_responsive_layout"):
            batch_view.update_responsive_layout(cur_w, cur_h)
        page.update()

    page.on_resized = _on_page_resize

    page.add(
        header_container,
        ft.Divider(height=1),
        tabs,
    )

    # Inicializa layout responsivo nas views e no cabeçalho
    update_header_layout(p_w)
    if hasattr(single_view, "update_responsive_layout"):
        single_view.update_responsive_layout(p_w, p_h)
    if hasattr(batch_view, "update_responsive_layout"):
        batch_view.update_responsive_layout(p_w, p_h)


# ---------------------------------------------------------------------------
# Funções Privadas de Configuração
# ---------------------------------------------------------------------------


def _configure_page(page: ft.Page) -> None:
    """Define as propriedades globais da página Flet com suporte a temas dinâmicos e ícones."""
    page.title = APP_TITLE
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.theme = theme.build_light_theme()
    page.dark_theme = theme.build_dark_theme()
    page.padding = 0
    page.spacing = 0
    if hasattr(page, "window") and page.window is not None:
        page.window.width = 1200
        page.window.height = 860
        page.window.min_width = 400
        page.window.min_height = 500

        # Ícone da janela Desktop nativa
        from src.core.samples import ASSETS_DIR
        ico_file = ASSETS_DIR / "favicon.ico"
        png_file = ASSETS_DIR / "favicon.png"
        if ico_file.exists():
            page.window.icon = str(ico_file)
        elif png_file.exists():
            page.window.icon = str(png_file)


_THEME_MODE_MAP: dict[str, ft.ThemeMode] = {
    "light": ft.ThemeMode.LIGHT,
    "dark": ft.ThemeMode.DARK,
    "system": ft.ThemeMode.SYSTEM,
}


def _get_header_padding(width: float | None) -> ft.Padding | int:
    """Retorna o padding responsivo do cabeçalho isolando a verificação de compatibilidade."""
    if not hasattr(ft, "Padding"):
        return 10
    is_mob = theme.is_mobile(width)
    return ft.Padding.symmetric(
        horizontal=theme.get_page_padding(width),
        vertical=8 if is_mob else 12,
    )


def _create_app_logo() -> ft.Container:
    """Cria o componente visual do logo da aplicação com fallback."""
    data_src = "favicon.png"
    try:
        from src.core.samples import ASSETS_DIR
        png_path = ASSETS_DIR / "favicon.png"
        if not png_path.exists():
            alt_path = Path(__file__).resolve().parent.parent / "assets" / "favicon.png"
            if alt_path.exists():
                png_path = alt_path
        if png_path.exists():
            import base64
            encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
            data_src = f"data:image/png;base64,{encoded}"
    except Exception:
        data_src = "favicon.png"

    return ft.Container(
        content=ft.Image(
            src=data_src,
            width=36,
            height=36,
            fit=getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None,
            border_radius=8,
            error_content=ft.Icon(ft.Icons.GRADIENT, size=32, color=theme.PRIMARY_LIGHT),
        ),
        border_radius=8,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )


def _create_version_badge() -> ft.Container:
    """Cria o badge com a versão atual da aplicação."""
    return ft.Container(
        content=ft.Text(
            f"v{APP_VERSION}",
            size=theme.FONT_CAPTION,
            color=theme.PRIMARY_LIGHT,
            weight=ft.FontWeight.BOLD,
        ),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=6,
        padding=ft.Padding.symmetric(horizontal=8, vertical=4) if hasattr(ft, "Padding") else 6,
    )


def _create_theme_selector(page: ft.Page) -> ft.SegmentedButton:
    """Cria o seletor segmentado para alternância de tema (Auto / Claro / Escuro)."""
    def _on_theme_change(e: ft.ControlEvent) -> None:
        if not e.control.selected:
            return
        mode_str = next(iter(e.control.selected), "system")
        page.theme_mode = _THEME_MODE_MAP.get(mode_str, ft.ThemeMode.SYSTEM)
        page.update()

    return ft.SegmentedButton(
        segments=[
            ft.Segment(
                value="system",
                label=ft.Text("Auto", size=theme.FONT_CAPTION),
                icon=ft.Icon(ft.Icons.BRIGHTNESS_AUTO, size=16),
            ),
            ft.Segment(
                value="light",
                label=ft.Text("Claro", size=theme.FONT_CAPTION),
                icon=ft.Icon(ft.Icons.LIGHT_MODE, size=16),
            ),
            ft.Segment(
                value="dark",
                label=ft.Text("Escuro", size=theme.FONT_CAPTION),
                icon=ft.Icon(ft.Icons.DARK_MODE, size=16),
            ),
        ],
        selected=["system"],
        on_change=_on_theme_change,
        show_selected_icon=False,
    )


def _build_brand_section(
    title: ft.Text,
    version_badge: ft.Container,
    subtitle: ft.Text,
    app_logo: ft.Container,
    spacing: int = 12,
) -> ft.Row:
    """Monta a seção de identidade visual (logo + títulos)."""
    return ft.Row(
        controls=[
            app_logo,
            ft.Column(
                controls=[
                    ft.Row(
                        controls=[title, version_badge],
                        spacing=spacing // 2,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                    subtitle,
                ],
                spacing=2,
                expand=True,
            ),
        ],
        spacing=spacing,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _build_header(page: ft.Page) -> tuple[ft.Container, Callable[[float | None], None]]:
    """
    Constrói a barra de cabeçalho da aplicação com logo oficial, título,
    badge de versão e seletor interativo de tema, com suporte a layout responsivo.
    """
    title = ft.Text(
        APP_TITLE,
        size=theme.FONT_HEADLINE,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.ON_SURFACE,
    )
    version_badge = _create_version_badge()
    subtitle = ft.Text(
        "Processamento Digital de Imagens • UFMA",
        size=theme.FONT_CAPTION,
        color=ft.Colors.ON_SURFACE_VARIANT,
    )
    app_logo = _create_app_logo()
    theme_selector = _create_theme_selector(page)
    brand_section = _build_brand_section(title, version_badge, subtitle, app_logo)

    header_container = ft.Container(
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        padding=_get_header_padding(getattr(page, "width", None)),
    )

    def _update_header_layout(width: float | None) -> None:
        is_mob = theme.is_mobile(width)
        title.size = theme.FONT_SUBTITLE if is_mob else theme.FONT_HEADLINE
        header_container.padding = _get_header_padding(width)

        if is_mob:
            header_container.content = ft.Column(
                controls=[
                    brand_section,
                    ft.Row(controls=[theme_selector], alignment=ft.MainAxisAlignment.CENTER),
                ],
                spacing=8,
            )
        else:
            header_container.content = ft.Row(
                controls=[brand_section, theme_selector],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            )

    _update_header_layout(getattr(page, "width", None))
    return header_container, _update_header_layout




"""
app.py — Aplicação Principal Flet do Projeto PDI.

Define a estrutura raiz da interface:
  - Barra de título com nome do projeto e badge de versão.
  - Sistema de abas (Processamento Individual | Processamento em Lote).
  - Configurações globais de tema (Dark Mode padrão).

Ponto de entrada da UI chamado por main.py.
"""

from pathlib import Path

import flet as ft

from src.ui import theme
from src.ui.views.batch_view import BatchView
from src.ui.views.single_view import SingleView

# Versão atual do aplicativo
APP_VERSION = "0.1"
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

    v1 = ft.Container(
        content=single_view,
        padding=theme.PADDING_PAGE,
        expand=True,
    )
    v2 = ft.Container(
        content=batch_view,
        padding=theme.PADDING_PAGE,
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

    page.add(
        _build_header(page),
        ft.Divider(height=1),
        tabs,
    )


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
        page.window.width = 1100
        page.window.height = 860
        page.window.min_width = 800
        page.window.min_height = 600

        # Ícone da janela Desktop nativa
        assets_dir = Path(__file__).resolve().parent.parent.parent / "assets"
        ico_file = assets_dir / "favicon.ico"
        png_file = assets_dir / "favicon.png"
        if ico_file.exists():
            page.window.icon = str(ico_file)
        elif png_file.exists():
            page.window.icon = str(png_file)


def _build_header(page: ft.Page) -> ft.Container:
    """
    Constrói a barra de cabeçalho da aplicação com logo oficial, título,
    badge de versão e seletor interativo de tema (Automático / Claro / Escuro).

    Args:
        page: Instância da página Flet para alternância dinâmica de tema.

    Returns:
        Container Flet com o header estilizado e adaptativo.
    """
    title = ft.Text(
        APP_TITLE,
        size=theme.FONT_HEADLINE,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.ON_SURFACE,
    )

    version_badge = ft.Container(
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

    subtitle = ft.Text(
        "Processamento Digital de Imagens • UFMA",
        size=theme.FONT_CAPTION,
        color=ft.Colors.ON_SURFACE_VARIANT,
    )

    # Logo do aplicativo com fallback para ícone gradiente
    app_logo = ft.Container(
        content=ft.Image(
            src="/favicon.png",
            width=36,
            height=36,
            fit=getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None,
            border_radius=8,
            error_content=ft.Icon(ft.Icons.GRADIENT, size=32, color=theme.PRIMARY_LIGHT),
        ),
        border_radius=8,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )

    # Seletor de Tema (Sistema / Claro / Escuro)
    def _on_theme_change(e: ft.ControlEvent) -> None:
        selected = e.control.selected
        if not selected:
            return
        mode_str = next(iter(selected))
        if mode_str == "light":
            page.theme_mode = ft.ThemeMode.LIGHT
        elif mode_str == "dark":
            page.theme_mode = ft.ThemeMode.DARK
        else:
            page.theme_mode = ft.ThemeMode.SYSTEM
        page.update()

    theme_selector = ft.SegmentedButton(
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

    return ft.Container(
        content=ft.Row(
            controls=[
                ft.Row(
                    controls=[
                        app_logo,
                        ft.Column(
                            controls=[
                                ft.Row(controls=[title, version_badge], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                subtitle,
                            ],
                            spacing=2,
                        ),
                    ],
                    spacing=14,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                theme_selector,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        padding=ft.Padding.symmetric(horizontal=theme.PADDING_PAGE, vertical=14) if hasattr(ft, "Padding") else 14,
    )



"""
theme.py — Paleta de Cores, Constantes de Estilo e Tema da Aplicação.

Centraliza todas as definições visuais da interface gráfica para garantir
consistência e facilitar futuras customizações de design.
"""

import flet as ft

# ---------------------------------------------------------------------------
# Paleta de Cores Principal
# ---------------------------------------------------------------------------

PRIMARY = "#1565C0"          # Azul primário (ações principais)
PRIMARY_LIGHT = "#5E92F3"    # Azul claro (hover, destaques)
PRIMARY_DARK = "#003C8F"     # Azul escuro (headers, ênfase)

ACCENT = "#E53935"           # Vermelho de destaque (alertas, erros)
SUCCESS = "#2E7D32"          # Verde (sucesso, confirmações)
WARNING = "#F57F17"          # Amarelo/âmbar (avisos)
INFO = "#0288D1"             # Azul informativo

# Paleta Modo Escuro (Dark)
SURFACE = "#1E1E2E"          # Fundo principal da aplicação
SURFACE_CARD = "#2A2A3E"     # Fundo dos cards internos
SURFACE_INPUT = "#313145"    # Fundo dos campos de entrada
TEXT_PRIMARY = "#ECEFF4"     # Texto principal (branco suave)
TEXT_SECONDARY = "#9EA3B0"   # Texto secundário (rótulos, dicas)
TEXT_DISABLED = "#555566"    # Texto desabilitado
DIVIDER = "#3D3D55"          # Linhas divisórias e bordas

# Paleta Modo Claro (Light)
LIGHT_SURFACE = "#F4F6F9"
LIGHT_SURFACE_CARD = "#FFFFFF"
LIGHT_SURFACE_INPUT = "#EAEFF5"
LIGHT_TEXT_PRIMARY = "#1A202C"
LIGHT_TEXT_SECONDARY = "#5A6578"
LIGHT_TEXT_DISABLED = "#A0AEC0"
LIGHT_DIVIDER = "#D8DEE9"

# Cores dos histogramas (consistentes com histogram.py)
HISTOGRAM_ORIGINAL = "#555555"
HISTOGRAM_UNIFORM = "#4a90d9"
HISTOGRAM_KMEANS = "#e8624a"

# ---------------------------------------------------------------------------
# Constantes de Tipografia
# ---------------------------------------------------------------------------

FONT_HEADLINE = 22    # Título principal do app
FONT_TITLE = 18       # Títulos de seções
FONT_SUBTITLE = 15    # Subtítulos de cards
FONT_BODY = 14        # Texto padrão
FONT_CAPTION = 12     # Legendas, rótulos secundários
FONT_MONO = 13        # Valores numéricos, métricas

# ---------------------------------------------------------------------------
# Constantes de Layout & Responsividade
# ---------------------------------------------------------------------------

PADDING_PAGE = 20
PADDING_CARD = 16
BORDER_RADIUS = 12
CARD_ELEVATION = 2

BREAKPOINT_MOBILE = 720
BREAKPOINT_TABLET = 1024


def is_mobile(page_width: float | int | None) -> bool:
    """Verifica se a largura da tela é considerada formato mobile/celular."""
    if page_width is None:
        return False
    return float(page_width) < BREAKPOINT_MOBILE


def get_page_padding(page_width: float | int | None) -> int:
    """Retorna o padding de página adequado conforme a largura da viewport."""
    return 10 if is_mobile(page_width) else PADDING_PAGE


def get_card_padding(page_width: float | int | None) -> int:
    """Retorna o padding de card adequado conforme a largura da viewport."""
    return 10 if is_mobile(page_width) else PADDING_CARD

# ---------------------------------------------------------------------------
# Funções de Construção de Tema
# ---------------------------------------------------------------------------


def build_light_theme() -> ft.Theme:
    """
    Constrói e retorna o tema claro (Light Mode) da aplicação Flet.

    Returns:
        Instância de ft.Theme configurada para o modo claro.
    """
    return ft.Theme(
        color_scheme_seed=PRIMARY,
        color_scheme=ft.ColorScheme(
            primary=PRIMARY,
            on_primary="#FFFFFF",
            secondary=ACCENT,
            surface=LIGHT_SURFACE,
            surface_container=LIGHT_SURFACE_CARD,
            surface_container_highest=LIGHT_SURFACE_INPUT,
            on_surface=LIGHT_TEXT_PRIMARY,
            on_surface_variant=LIGHT_TEXT_SECONDARY,
            outline=LIGHT_DIVIDER,
        ),
        scaffold_bgcolor=LIGHT_SURFACE,
        card_bgcolor=LIGHT_SURFACE_CARD,
        divider_color=LIGHT_DIVIDER,
        visual_density=ft.VisualDensity.COMFORTABLE if hasattr(ft, "VisualDensity") else None,
    )


def build_dark_theme() -> ft.Theme:
    """
    Constrói e retorna o tema escuro (Dark Mode) da aplicação Flet.

    Returns:
        Instância de ft.Theme configurada para o modo escuro.
    """
    return ft.Theme(
        color_scheme_seed=PRIMARY,
        color_scheme=ft.ColorScheme(
            primary=PRIMARY_LIGHT,
            on_primary="#FFFFFF",
            secondary=ACCENT,
            surface=SURFACE,
            surface_container=SURFACE_CARD,
            surface_container_highest=SURFACE_INPUT,
            on_surface=TEXT_PRIMARY,
            on_surface_variant=TEXT_SECONDARY,
            outline=DIVIDER,
        ),
        scaffold_bgcolor=SURFACE,
        card_bgcolor=SURFACE_CARD,
        divider_color=DIVIDER,
        visual_density=ft.VisualDensity.COMFORTABLE if hasattr(ft, "VisualDensity") else None,
    )


def build_app_theme() -> ft.Theme:
    """
    Retorna o tema padrão (Dark Mode) para retrocompatibilidade.
    """
    return build_dark_theme()


def card(content: ft.Control, padding: int = PADDING_CARD) -> ft.Container:
    """
    Cria um container estilizado como card de interface com tema adaptativo.

    Args:
        content: Controle Flet a ser exibido dentro do card.
        padding: Espaçamento interno do card em pixels.

    Returns:
        Container estilizado com bordas arredondadas e fundo de card.
    """
    return ft.Container(
        content=content,
        bgcolor=ft.Colors.SURFACE_CONTAINER,
        border_radius=BORDER_RADIUS,
        padding=padding,
    )


def section_title(text: str, icon: str | None = None) -> ft.Row | ft.Text:
    """
    Cria um texto de título de seção estilizado com ícone opcional.

    Args:
        text: Texto do título.
        icon: Ícone opcional do Flet (ex: ft.Icons.SETTINGS).

    Returns:
        Controle ft.Row ou ft.Text com estilo de título de seção.
    """
    title_text = ft.Text(
        value=text,
        size=FONT_TITLE,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.ON_SURFACE,
    )
    if icon:
        return ft.Row(
            controls=[
                ft.Icon(icon, size=22, color=PRIMARY_LIGHT),
                title_text,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    return title_text


def metric_badge(label: str, value: str, color: str = PRIMARY_LIGHT) -> ft.Container:
    """
    Cria um badge de métrica com rótulo e valor formatados e tema adaptativo.

    Args:
        label: Nome da métrica (ex: "MSE", "PSNR").
        value: Valor formatado da métrica (ex: "42.31 dB").
        color: Cor de destaque do badge.

    Returns:
        Container Flet estilizado como badge de métrica.
    """
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(label, size=FONT_CAPTION, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(value, size=FONT_MONO, weight=ft.FontWeight.BOLD, color=color),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=2,
        ),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=8,
        padding=ft.Padding.symmetric(horizontal=14, vertical=8) if hasattr(ft, "Padding") else 10,
    )


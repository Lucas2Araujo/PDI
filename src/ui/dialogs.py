"""
dialogs.py — Modais e Diálogos Compartilhados da Interface (UI).

Centraliza a exibição de:
1. Visualizador Modal de Imagens com Zoom Interativo (até 10×), Pan e Reset.
2. Modal Didático "Entranhas do Processo" (Raio-X de 4 abas do Pipeline de PDI).

Totalmente desacoplado para uso tanto na visualização individual quanto no lote.
"""

from typing import Any, Callable
import flet as ft
import numpy as np

from src.core.grayscale import GrayscaleMethod
from src.core.inspector import extract_pipeline_telemetry
from src.core.quantization import QuantizationTechnique
from src.ui import theme
from src.ui.common import _bytes_to_data_uri


# ---------------------------------------------------------------------------
# 1. Visualizador Modal com Zoom Interativo e Pan
# ---------------------------------------------------------------------------


def open_zoom_dialog(
    page: ft.Page,
    title: str = "Visualização com Zoom",
    image_bytes: bytes | str | None = None,
    on_download: Callable[[], None] | None = None,
) -> None:
    """
    Abre um pop-up modal (AlertDialog) dedicado para visualização de imagem
    em alta resolução com ferramentas completas de zoom (0.25x a 10x), pan
    e botão opcional para download direto do arquivo em alta resolução.

    Args:
        page: Instância ativa da página Flet.
        title: Título da janela modal de zoom.
        image_bytes: Dados da imagem em bytes PNG ou string Data-URI.
        on_download: Callback opcional executado para baixar a imagem inspecionada.
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

    def _close_dialog(_: ft.ControlEvent | None = None) -> None:
        dialog.open = False
        page.pop_dialog()
        page.update()

    def _update_zoom_ui() -> None:
        zoom_label.value = f"{int(scale_val[0] * 100)}%"
        dialog.update()

    def _on_zoom_in(_: ft.ControlEvent) -> None:
        scale_val[0] = round(min(10.0, scale_val[0] + 0.25), 2)
        img_control.scale = ft.Scale(scale_val[0])
        _update_zoom_ui()

    def _on_zoom_out(_: ft.ControlEvent) -> None:
        scale_val[0] = round(max(0.25, scale_val[0] - 0.25), 2)
        img_control.scale = ft.Scale(scale_val[0])
        _update_zoom_ui()

    def _on_zoom_reset(_: ft.ControlEvent) -> None:
        scale_val[0] = 1.0
        img_control.scale = ft.Scale(1.0)
        _update_zoom_ui()

    p_w = getattr(page, "width", None) or 800
    p_h = getattr(page, "height", None) or 600
    dlg_w = min(int(p_w * 0.92), 960)
    dlg_h = min(int(p_h * 0.72), 580)
    is_mob = theme.is_mobile(p_w)

    title_actions: list[ft.Control] = [
        ft.IconButton(
            icon=ft.Icons.ZOOM_OUT,
            icon_size=18 if is_mob else 22,
            tooltip="Diminuir Zoom (-25%)",
            on_click=_on_zoom_out,
        ),
        ft.Container(
            content=zoom_label,
            padding=ft.Padding.symmetric(horizontal=4) if hasattr(ft, "Padding") else 4,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
        ),
        ft.IconButton(
            icon=ft.Icons.ZOOM_IN,
            icon_size=18 if is_mob else 22,
            tooltip="Aumentar Zoom (+25%)",
            on_click=_on_zoom_in,
        ),
        ft.IconButton(
            icon=ft.Icons.RESTART_ALT,
            icon_size=18 if is_mob else 22,
            tooltip="Resetar Zoom (100%)",
            on_click=_on_zoom_reset,
        ),
    ]
    if on_download is not None:
        title_actions.append(
            ft.IconButton(
                icon=ft.Icons.DOWNLOAD,
                icon_size=18 if is_mob else 22,
                tooltip="Baixar Imagem",
                on_click=lambda _: on_download(),
            )
        )
    title_actions.append(
        ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_size=18 if is_mob else 22,
            tooltip="Fechar",
            on_click=_close_dialog,
        )
    )

    dialog.title = ft.Row(
        controls=[
            ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ZOOM_IN, size=20 if is_mob else 24, color=theme.PRIMARY_LIGHT),
                    ft.Text(
                        title,
                        weight=ft.FontWeight.BOLD,
                        size=theme.FONT_SUBTITLE if is_mob else theme.FONT_TITLE,
                        color=ft.Colors.ON_SURFACE,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
                spacing=6,
                wrap=True,
            ),
            ft.Row(
                controls=title_actions,
                spacing=2,
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        wrap=True,
    )

    dialog.content = ft.Container(
        content=interactive_viewer,
        width=dlg_w,
        height=dlg_h,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border_radius=8,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )

    action_buttons: list[ft.Control] = []
    if on_download is not None:
        action_buttons.append(
            ft.Button(
                content="Baixar Imagem",
                icon=ft.Icons.DOWNLOAD,
                on_click=lambda _: on_download(),
                bgcolor=theme.PRIMARY,
                color="#FFFFFF",
            )
        )

    action_buttons.append(
        ft.Button(
            content="Fechar",
            on_click=_close_dialog,
            bgcolor=theme.PRIMARY_DARK if on_download is not None else theme.PRIMARY,
            color="#FFFFFF",
        )
    )

    dialog.actions = [
        ft.Row(
            controls=[
                ft.Text(
                    "💡 Dica: Toque/arraste para mover. Use pinch ou botões para zoom."
                    if is_mob
                    else "💡 Dica: Use a roda do mouse ou os botões de zoom acima. Arraste com o cursor para mover a imagem (Pan).",
                    size=theme.FONT_CAPTION,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                ft.Row(controls=action_buttons, spacing=8),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            wrap=True,
        )
    ]

    page.show_dialog(dialog)
    page.update()


def open_histogram_zoom_dialog(
    page: ft.Page,
    title: str,
    data: np.ndarray | None,
    color: str = theme.PRIMARY_LIGHT,
    is_rgb: bool = False,
    is_quantized: bool = False,
    chart_height: int = 340,
) -> None:
    """
    Abre um pop-up modal (AlertDialog) dedicado para visualização e análise
    estatística em alta resolução do histograma nativo via Flutter Canvas.
    """
    if data is None:
        return

    from src.ui.components.histogram_chart import NativeHistogramChart

    p_w = getattr(page, "width", None) or 800
    p_h = getattr(page, "height", None) or 600
    dlg_w = min(int(p_w * 0.92), 960)
    dlg_h = min(int(p_h * 0.76), 620)
    is_mob = theme.is_mobile(p_w)

    chart = NativeHistogramChart(
        title=title,
        image_or_data=data,
        color=color,
        is_rgb=is_rgb,
        is_quantized=is_quantized,
        chart_height=chart_height,
    )

    # Métricas estatísticas adicionais
    stats_cards: list[ft.Control] = []
    if isinstance(data, np.ndarray):
        flat = data.flatten()
        mean_val = float(np.mean(flat))
        std_val = float(np.std(flat))
        min_val = int(np.min(flat))
        max_val = int(np.max(flat))
        unique_levels = int(len(np.unique(flat)))
        total_px = int(flat.size)

        def _stat_item(label: str, val: str) -> ft.Container:
            return ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(label, size=10, color=ft.Colors.ON_SURFACE_VARIANT, weight=ft.FontWeight.BOLD),
                        ft.Text(val, size=13, color=theme.PRIMARY_LIGHT, weight=ft.FontWeight.BOLD),
                    ],
                    spacing=2,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                padding=ft.Padding.symmetric(horizontal=10, vertical=6) if hasattr(ft, "Padding") else 6,
                border_radius=6,
            )

        stats_cards = [
            _stat_item("Resolução", f"{total_px:,} px"),
            _stat_item("Níveis Únicos", str(unique_levels)),
            _stat_item("Mín / Máx", f"{min_val} / {max_val}"),
            _stat_item("Média (μ)", f"{mean_val:.1f}"),
            _stat_item("Desvio (σ)", f"{std_val:.1f}"),
        ]

    dialog = ft.AlertDialog(
        modal=True,
        content_padding=12,
        title_padding=ft.Padding.only(left=20, top=16, right=16, bottom=8) if hasattr(ft, "Padding") else 16,
        actions_padding=ft.Padding.only(left=20, right=20, bottom=16) if hasattr(ft, "Padding") else 16,
    )

    def _close_dialog(_: ft.ControlEvent | None = None) -> None:
        dialog.open = False
        page.pop_dialog()
        page.update()

    dialog.title = ft.Row(
        controls=[
            ft.Row(
                controls=[
                    ft.Icon(ft.Icons.BAR_CHART, size=20 if is_mob else 24, color=theme.PRIMARY_LIGHT),
                    ft.Text(title, weight=ft.FontWeight.BOLD, size=theme.FONT_TITLE, color=ft.Colors.ON_SURFACE),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.IconButton(
                icon=ft.Icons.CLOSE,
                icon_size=18 if is_mob else 22,
                tooltip="Fechar",
                on_click=_close_dialog,
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    stats_row = (
        ft.Row(controls=stats_cards, spacing=8, wrap=True, alignment=ft.MainAxisAlignment.CENTER)
        if stats_cards
        else ft.Container()
    )

    dialog.content = ft.Container(
        content=ft.Column(
            controls=[
                chart,
                stats_row,
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            scroll=ft.ScrollMode.AUTO,
        ),
        width=dlg_w,
        height=dlg_h,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border_radius=8,
        padding=8,
    )

    dialog.actions = [
        ft.Row(
            controls=[
                ft.Text(
                    "📊 Histograma nativo renderizado em tempo real acelerado via Flutter Canvas.",
                    size=theme.FONT_CAPTION,
                    color=ft.Colors.ON_SURFACE_VARIANT,
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
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            wrap=True,
        )
    ]

    page.show_dialog(dialog)
    page.update()


# ---------------------------------------------------------------------------
# 2. Modal Didático "Entranhas do Processo" (Raio-X de 4 Abas)
# ---------------------------------------------------------------------------


def _build_matrix_tab(telemetry) -> ft.Container:
    """Constrói a Aba 1: Estrutura Matricial da Imagem Digital."""
    sample_rows = []
    s_r, s_c = telemetry.sample_coords
    for r_idx in range(telemetry.sample_gray.shape[0]):
        cells = [ft.DataCell(ft.Text(f"L{s_r + r_idx}", weight=ft.FontWeight.BOLD, size=11))]
        for c_idx in range(telemetry.sample_gray.shape[1]):
            if telemetry.is_color:
                r_v = telemetry.sample_raw[r_idx, c_idx, 0]
                g_v = telemetry.sample_raw[r_idx, c_idx, 1]
                b_v = telemetry.sample_raw[r_idx, c_idx, 2]
                txt = f"[{r_v},{g_v},{b_v}]"
            else:
                txt = str(telemetry.sample_gray[r_idx, c_idx])
            cells.append(ft.DataCell(ft.Text(txt, size=11)))
        sample_rows.append(ft.DataRow(cells=cells))

    cols = [ft.DataColumn(ft.Text("Coord", weight=ft.FontWeight.BOLD, size=11))]
    for c_idx in range(telemetry.sample_gray.shape[1]):
        cols.append(ft.DataColumn(ft.Text(f"C{s_c + c_idx}", weight=ft.FontWeight.BOLD, size=11)))

    matrix_table = ft.DataTable(
        columns=cols,
        rows=sample_rows,
        border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
        horizontal_lines=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "BorderSide") else None,
        vertical_lines=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "BorderSide") else None,
        show_bottom_border=True,
        heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        column_spacing=14,
        heading_row_height=36,
        data_row_min_height=32,
        data_row_max_height=38,
    )

    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("1. Estrutura Matricial da Imagem Digital", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE, color=theme.PRIMARY_LIGHT),
                ft.Text(
                    f"Dimensões: {telemetry.image_shape[1]}×{telemetry.image_shape[0]} pixels · "
                    f"Formato: {'Colorido RGB (3 canais / 24 bpp)' if telemetry.is_color else 'Monocromático (1 canal / 8 bpp)'} · "
                    f"Total de Pixels: {telemetry.image_shape[0] * telemetry.image_shape[1]:,} pixels",
                    size=theme.FONT_BODY,
                ),
                ft.Divider(height=1),
                ft.Text("🔬 Amostra Numérica Central 5×5 (Valores de Intensidade Discreta 0–255):", weight=ft.FontWeight.BOLD, size=theme.FONT_CAPTION),
                ft.Row([matrix_table], scroll=ft.ScrollMode.AUTO),
                ft.Text(
                    "💡 Cada pixel digital é representado por um valor inteiro quantizado de 8 bits (0 a 255). No caso RGB, é uma tupla [R, G, B].",
                    color=ft.Colors.ON_SURFACE_VARIANT,
                    size=theme.FONT_CAPTION,
                ),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=10,
    )


def _build_arithmetic_tab(telemetry) -> ft.Container:
    """Constrói a Aba 2: Matemática e Aritmética da Conversão para Tons de Cinza."""
    calc_items = [
        ft.Text(calc, size=12, font_family="monospace") for calc in telemetry.pixel_calculations[:25]
    ]
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("2. Aritmética da Conversão para Tons de Cinza", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE, color=theme.PRIMARY_LIGHT),
                ft.Text(f"Método Utilizado: {telemetry.grayscale_method_name}", weight=ft.FontWeight.BOLD, size=theme.FONT_BODY),
                ft.Divider(height=1),
                ft.Text("🧮 Equações Aplicadas Pixel a Pixel na Região Central (Amostra 5×5):", weight=ft.FontWeight.BOLD, size=theme.FONT_CAPTION),
                ft.Container(
                    content=ft.ListView(controls=calc_items, spacing=4, height=220),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=8,
                    padding=8,
                ),
                ft.Text(
                    "💡 A conversão ponderada (ITU-R BT.601) respeita a curva de sensibilidade do olho humano aos comprimentos de onda verde (58.7%), vermelho (29.9%) e azul (11.4%).",
                    color=ft.Colors.ON_SURFACE_VARIANT,
                    size=theme.FONT_CAPTION,
                ),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=10,
    )


def _build_quant_table_tab(telemetry) -> ft.Container:
    """Constrói a Aba 3: Particionamento e Tabela de Mapeamento dos Níveis."""
    q_rows = []
    for r_dict in telemetry.quant_info.table_rows:
        if "range" in r_dict and "reconstruction" in r_dict:
            q_rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(r_dict.get("index", "")), weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Text(str(r_dict.get("range", "")))),
                        ft.DataCell(ft.Text(str(r_dict.get("reconstruction", "")), color=theme.SUCCESS, weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Text(f"{r_dict.get('count', 0):,}")),
                        ft.DataCell(ft.Text(str(r_dict.get("pct", "")))),
                    ]
                )
            )

    q_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Nível", weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Faixa / Cluster", weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Reconstrução (Tom)", weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Qtd Pixels", weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("% Imagem", weight=ft.FontWeight.BOLD)),
        ],
        rows=q_rows,
        border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
        horizontal_lines=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "BorderSide") else None,
        vertical_lines=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "BorderSide") else None,
        show_bottom_border=True,
        heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        column_spacing=24,
        heading_row_height=38,
        data_row_min_height=34,
        data_row_max_height=42,
    )

    step_info_str = f" · Tamanho do Passo (Δ): {telemetry.quant_info.step_size}" if telemetry.quant_info.step_size else ""
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(f"3. Particionamento e Níveis — {telemetry.quantization_technique_name}", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE, color=theme.PRIMARY_LIGHT),
                ft.Text(
                    f"Resolução: {telemetry.bits} bits · Níveis de Saída (L = 2^b): {telemetry.n_levels} tons{step_info_str}",
                    size=theme.FONT_BODY,
                ),
                ft.Divider(height=1),
                ft.Text("📊 Tabela de Mapeamento dos Intervalos de Decisão e Centróides:", weight=ft.FontWeight.BOLD, size=theme.FONT_CAPTION),
                ft.Container(
                    content=ft.Row([q_table], scroll=ft.ScrollMode.AUTO),
                    border_radius=8,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                ),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=10,
    )


def _build_heatmap_tab(page: ft.Page, telemetry) -> ft.Container:
    """Constrói a Aba 4: Auditoria de Erro Residual e Mapa de Calor."""
    heatmap_img = ft.Image(
        src=_bytes_to_data_uri(telemetry.heatmap_bytes),
        fit=getattr(ft.BoxFit, "CONTAIN", None) if hasattr(ft, "BoxFit") else None,
        expand=True,
    )

    title_str = "4. Auditoria de Erro Residual — Mapa de Calor (Heatmap)"
    btn_heatmap_zoom = ft.Button(
        content="🔍 Ampliar Mapa de Calor",
        icon=ft.Icons.ZOOM_IN,
        on_click=lambda _: open_zoom_dialog(page, title_str, telemetry.heatmap_bytes),
        bgcolor=theme.PRIMARY,
        color="#FFFFFF",
    )

    heatmap_box = ft.Container(
        content=heatmap_img,
        height=280,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border_radius=8,
        padding=6,
        alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
        on_click=lambda _: open_zoom_dialog(page, title_str, telemetry.heatmap_bytes),
        ink=True,
        tooltip="Clique para abrir o mapa de calor no visualizador com zoom",
    )

    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("4. Auditoria de Erro Residual e Análise Termográfica", weight=ft.FontWeight.BOLD, size=theme.FONT_SUBTITLE, color=theme.PRIMARY_LIGHT),
                ft.Row(
                    controls=[
                        theme.metric_badge("MSE", f"{telemetry.mse:.2f}"),
                        theme.metric_badge("PSNR", f"{telemetry.psnr:.2f} dB", color=theme.SUCCESS),
                        theme.metric_badge("Erro Máx", str(telemetry.max_error)),
                        theme.metric_badge("Erro Médio", f"{telemetry.mean_error:.2f}"),
                        theme.metric_badge("Economia", f"{telemetry.memory_savings_pct:.1f}%", color=theme.WARNING),
                    ],
                    spacing=10,
                    wrap=True,
                ),
                ft.Divider(height=1),
                ft.Row(
                    controls=[
                        btn_heatmap_zoom,
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
                heatmap_box,
                ft.Text(
                    "💡 O Mapa de Calor (Heatmap) mapeia pixel a pixel o erro residual absoluto |I_cinza - I_quant|. Áreas amarelas/brancas indicam maior discrepância de intensidade, enquanto áreas escuras/roxas indicam conservação exata.",
                    color=ft.Colors.ON_SURFACE_VARIANT,
                    size=theme.FONT_CAPTION,
                ),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=10,
    )


def open_inspector_dialog(
    page: ft.Page,
    raw_image: np.ndarray,
    gray_image: np.ndarray,
    quantized_image: np.ndarray,
    bits: int,
    technique: QuantizationTechnique | str,
    method: GrayscaleMethod,
) -> None:
    """
    Abre o modal didático de telemetria completa com 4 abas didáticas:
    1. Estrutura Matricial Original (Amostra central 5×5)
    2. Aritmética da Conversão para Tons de Cinza (Equações detalhadas)
    3. Tabela de Quantização e Centróides (Intervalos de decisão completos)
    4. Auditoria de Erro Residual e Mapa de Calor com suporte a Zoom.
    """
    telemetry = extract_pipeline_telemetry(
        raw_image=raw_image,
        gray_image=gray_image,
        quantized_image=quantized_image,
        bits=bits,
        technique=technique,
        method=method,
    )

    p_w = getattr(page, "width", None) or 800
    p_h = getattr(page, "height", None) or 600
    dlg_w = min(int(p_w * 0.95), 1050)
    dlg_h = min(int(p_h * 0.85), 680)
    is_mob = theme.is_mobile(p_w)

    dialog = ft.AlertDialog(
        modal=True,
        content_padding=ft.Padding.all(12) if hasattr(ft, "Padding") else 12,
        actions_padding=ft.Padding.all(10) if hasattr(ft, "Padding") else 10,
    )

    def _close_dialog(_: ft.ControlEvent | None = None) -> None:
        dialog.open = False
        page.pop_dialog()
        page.update()

    tab1_content = _build_matrix_tab(telemetry)
    tab2_content = _build_arithmetic_tab(telemetry)
    tab3_content = _build_quant_table_tab(telemetry)
    tab4_content = _build_heatmap_tab(page, telemetry)

    inspector_tabs = ft.Tabs(
        selected_index=0,
        length=4,
        content=ft.Column(
            controls=[
                ft.TabBar(
                    tabs=[
                        ft.Tab(label="1. Matriz Original", icon=ft.Icons.GRID_ON),
                        ft.Tab(label="2. Aritmética Cinza", icon=ft.Icons.CALCULATE),
                        ft.Tab(label="3. Tabela de Quantização", icon=ft.Icons.TABLE_CHART),
                        ft.Tab(label="4. Mapa de Calor (Erro)", icon=ft.Icons.LOCAL_FIRE_DEPARTMENT),
                    ]
                ),
                ft.TabBarView(
                    controls=[tab1_content, tab2_content, tab3_content, tab4_content],
                    expand=True,
                ),
            ],
            expand=True,
        ),
        expand=True,
    )

    dialog.title = ft.Row(
        controls=[
            ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ANALYTICS, size=24, color=theme.PRIMARY_LIGHT),
                    ft.Text(
                        "🔬 Entranhas do Processo — Raio-X Didático do Pipeline de PDI",
                        weight=ft.FontWeight.BOLD,
                        size=theme.FONT_SUBTITLE if is_mob else theme.FONT_TITLE,
                        color=ft.Colors.ON_SURFACE,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
                spacing=8,
                wrap=True,
            ),
            ft.IconButton(
                icon=ft.Icons.CLOSE,
                tooltip="Fechar",
                on_click=_close_dialog,
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        wrap=True,
    )

    dialog.content = ft.Container(
        content=inspector_tabs,
        width=dlg_w,
        height=dlg_h,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        border_radius=8,
    )

    dialog.actions = [
        ft.Button(
            content="Entendido / Fechar",
            icon=ft.Icons.CHECK,
            on_click=_close_dialog,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
    ]

    page.show_dialog(dialog)
    page.update()


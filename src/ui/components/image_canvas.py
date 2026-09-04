"""
image_canvas.py — Área Central de Trabalho para Exibição Interativa do Resultado.

Oferece modos de exibição flexíveis e 100% responsivos:
- Resultado Puro: Imagem processada em alta definição com ajuste adaptativo.
- Lado a Lado (Comparativo): Entrada vs Resultado (Row no Desktop, Column em Mobile < 600px).
- Grade Tripla: Entrada × Intermediário/Máscara × Resultado.
- Interatividade com Zoom (0.25× a 10×) via open_zoom_dialog e download direto.
- Zero alturas fixas rígidas (eliminação total do padrão height=520).
- Integração reativa automática com o SessionState (evento result_changed).
"""

from enum import Enum, auto
from typing import Any, Callable
import flet as ft
import numpy as np

from src.core.grayscale import GrayscaleMethod, to_grayscale
from src.ui import theme
from src.ui.common import (
    TRANSPARENT_PIXEL_PNG_URI,
    _bytes_to_data_uri,
    _ndarray_to_png_bytes,
)
from src.ui.components.histogram_chart import NativeHistogramChart
from src.ui.dialogs import (
    open_histogram_zoom_dialog,
    open_inspector_dialog,
    open_zoom_dialog,
)
from src.ui.state.session_state import (
    EVENT_IMAGE_A_CHANGED,
    EVENT_IMAGE_B_CHANGED,
    EVENT_RESULT_CHANGED,
    SessionState,
    get_session_state,
)


class DisplayMode(Enum):
    """Modos de visualização suportados pelo ImageCanvas."""
    RESULT_ONLY = auto()     # 🖼️ Apenas imagem resultante
    SIDE_BY_SIDE = auto()    # 🌓 Comparativo lado a lado (Entrada A × Resultado)
    TRIPLE = auto()          # 📑 Grade tripla (Entrada A × Intermediário × Resultado)
    ANALYTICS = auto()       # 📊 Painel Analítico & Histogramas
    COMPARISON = auto()      # ⚡ Comparação Quádrupla dos 4 Algoritmos


class ImageCanvas(ft.Container):
    """
    Área de trabalho central e responsiva para exibição de imagens processadas,
    comparativos lado a lado, grade tripla, painel analítico com histogramas e comparador múltiplo.
    """

    def __init__(
        self,
        session_state: SessionState | None = None,
        page: ft.Page | None = None,
        default_mode: DisplayMode = DisplayMode.RESULT_ONLY,
        on_download: Callable[[bytes, str], None] | None = None,
        on_promote: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o ImageCanvas.

        Args:
            session_state: Gerenciador de estado reativo (padrão: singleton).
            page: Instância ativa da página Flet.
            default_mode: Modo inicial de exibição.
            on_download: Callback customizado de download de imagem.
            on_promote: Callback de promoção do resultado para Entrada A (Pipeline).
            **kwargs: Parâmetros repassados ao ft.Container.
        """
        self.session_state = session_state or get_session_state()
        self._page = page
        self.display_mode = default_mode
        self.on_download = on_download
        self.on_promote = on_promote
        self._btn_promote_canvas: ft.Button | None = None

        # Dados das imagens ativas
        self._result_image: np.ndarray | None = None
        self._result_name: str = "resultado.png"
        self._intermediate_image: np.ndarray | None = None
        self._intermediate_name: str = "intermediario.png"
        self._metrics: dict[str, Any] = {}

        # Dados do modo de comparação quádrupla
        self._comparison_results: dict[str, Any] | None = None
        self._comparison_winner: str = ""

        # Controles visuais de exibição
        self._img_result = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=getattr(ft.BoxFit, "CONTAIN", None), expand=True)
        self._img_slot_a = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=getattr(ft.BoxFit, "CONTAIN", None), expand=True)
        self._img_inter = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=getattr(ft.BoxFit, "CONTAIN", None), expand=True)
        self._img_analytics = ft.Image(src=TRANSPARENT_PIXEL_PNG_URI, fit=getattr(ft.BoxFit, "CONTAIN", None), expand=True)
        self._figure_bytes: bytes | None = None

        # Componentes do Painel Analítico de Histogramas
        self._histogram_chart_a = NativeHistogramChart(
            title="Histograma — Entrada A",
            chart_height=180,
            on_zoom_fn=self._zoom_histogram_a,
        )
        self._histogram_chart_res = NativeHistogramChart(
            title="Histograma — Resultado",
            chart_height=180,
            on_zoom_fn=self._zoom_histogram_res,
        )
        self._analytics_badges_row = ft.Row(
            spacing=8,
            wrap=True,
            alignment=ft.MainAxisAlignment.START,
        )

        # Inscrições reativas
        self._unsub_result = self.session_state.subscribe(EVENT_RESULT_CHANGED, self._on_result_changed)
        self._unsub_a = self.session_state.subscribe(EVENT_IMAGE_A_CHANGED, self._on_image_a_changed)
        self._unsub_b = self.session_state.subscribe(EVENT_IMAGE_B_CHANGED, self._on_image_b_changed)

        # Contêineres de layout
        self._toolbar_row = ft.Row(
            spacing=8,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            wrap=True,
        )
        self._display_area = ft.Container(
            expand=True,
            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
        )

        main_col = ft.Column(
            controls=[
                self._toolbar_row,
                self._display_area,
            ],
            spacing=10,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        container_kwargs: dict[str, Any] = {
            "bgcolor": ft.Colors.SURFACE_CONTAINER_LOW,
            "border_radius": theme.BORDER_RADIUS,
            "padding": 12,
            "expand": True,
            "content": main_col,
        }
        container_kwargs.update(kwargs)
        super().__init__(**container_kwargs)

        self._build_toolbar()
        self._render_canvas()

    # -----------------------------------------------------------------------
    # Utilitários de Atualização Segura
    # -----------------------------------------------------------------------

    def _safe_update(self) -> None:
        """Invoca update() apenas se montado na árvore da página."""
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
    # Métodos Públicos
    # -----------------------------------------------------------------------

    def set_display_mode(self, mode: DisplayMode) -> None:
        """Altera o modo de exibição ativo e atualiza a interface."""
        self.display_mode = mode
        self._build_toolbar()
        self._render_canvas()
        self._safe_update()

    def set_result(self, image: np.ndarray | None, name: str = "resultado.png") -> None:
        """Define o resultado exibido no canvas."""
        self._result_image = image
        self._result_name = name
        self._update_result_image_control()
        self._render_canvas()
        self._safe_update()

    def set_intermediate(self, image: np.ndarray | None, name: str = "intermediario.png") -> None:
        """Define a imagem intermediária (ex: tons de cinza ou dither) para o modo triplo."""
        self._intermediate_image = image
        self._intermediate_name = name
        if image is not None:
            self._img_inter.src = _bytes_to_data_uri(_ndarray_to_png_bytes(image))
        else:
            self._img_inter.src = TRANSPARENT_PIXEL_PNG_URI
        self._render_canvas()
        self._safe_update()

    # -----------------------------------------------------------------------
    # Reatividade ao SessionState
    # -----------------------------------------------------------------------

    def _on_result_changed(
        self,
        image: np.ndarray | None = None,
        metrics: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        """Trata evento de alteração do resultado no SessionState."""
        self._result_image = image
        if hasattr(self, "_btn_promote_canvas") and self._btn_promote_canvas is not None:
            self._btn_promote_canvas.disabled = (image is None)

        if metrics is not None:
            self._metrics = dict(metrics)
            if metrics.get("is_comparison") and "comparison_results" in metrics:
                self._comparison_results = metrics["comparison_results"]
                self._comparison_winner = metrics.get("winner", "")
                self.display_mode = DisplayMode.COMPARISON
            elif not metrics.get("is_comparison"):
                self._comparison_results = None
                self._comparison_winner = ""

        # Resolução e cache da imagem intermediária para o modo TRIPLE
        if metrics and metrics.get("intermediate_image") is not None:
            self._intermediate_image = metrics["intermediate_image"]
        elif self.session_state.image_b is not None:
            self._intermediate_image = self.session_state.image_b
        elif self.session_state.image_a is not None:
            if self.session_state.image_a.ndim == 3:
                self._intermediate_image = to_grayscale(self.session_state.image_a, GrayscaleMethod.LUMINANCE)
            else:
                self._intermediate_image = self.session_state.image_a

        if self._intermediate_image is not None:
            self._img_inter.src = _bytes_to_data_uri(_ndarray_to_png_bytes(self._intermediate_image))

        if self.session_state.image_a is not None:
            self._img_slot_a.src = _bytes_to_data_uri(_ndarray_to_png_bytes(self.session_state.image_a))
            is_rgb_a = bool(self.session_state.image_a.ndim == 3 and self.session_state.image_a.shape[2] >= 3)
            self._histogram_chart_a.set_data(
                self.session_state.image_a,
                title="Histograma — Entrada A",
                is_rgb=is_rgb_a,
            )

        if self._result_image is not None:
            is_rgb_res = bool(self._result_image.ndim == 3 and self._result_image.shape[2] >= 3)
            self._histogram_chart_res.set_data(
                self._result_image,
                title="Histograma — Resultado",
                is_rgb=is_rgb_res,
                is_quantized=True,
            )

        # Geração ou extração da figura analítica consolidada (Imagens + Histogramas integrados)
        if metrics and metrics.get("figure_bytes"):
            self._figure_bytes = metrics["figure_bytes"]
        elif self._result_image is not None and self.session_state.image_a is not None:
            self._figure_bytes = self._generate_analytics_figure_bytes()
        else:
            self._figure_bytes = None

        if self._figure_bytes:
            self._img_analytics.src = _bytes_to_data_uri(self._figure_bytes)

        self._update_result_image_control()
        self._build_toolbar()
        self._render_canvas()
        self._safe_update()

    def _generate_analytics_figure_bytes(self) -> bytes | None:
        """Gera a figura analítica consolidada em alta resolução (Imagens + Histogramas) via Pillow."""
        if self._result_image is None:
            return None
        orig = self.session_state.image_a if self.session_state.image_a is not None else self._result_image
        bits = int(self._metrics.get("bits", 4))
        tech = str(self._metrics.get("technique", "Resultado"))
        m_label = self._metrics.get("gray_method_name", None)

        from src.core.histogram import generate_comparison_figure, generate_color_comparison_figure
        try:
            is_color = bool(orig.ndim == 3 and orig.shape[2] >= 3)
            if is_color:
                return generate_color_comparison_figure(
                    color_image=orig,
                    quantized=self._result_image,
                    bits=bits,
                    technique_name=tech,
                    gray_image=self._intermediate_image if (self._intermediate_image is not None and self._intermediate_image.ndim == 2) else None,
                    gray_method_name=m_label,
                )
            else:
                return generate_comparison_figure(
                    original=orig,
                    quantized=self._result_image,
                    bits=bits,
                    technique_name=tech,
                    gray_method_name=m_label,
                )
        except Exception:
            return None

    def _on_image_a_changed(self, image: np.ndarray | None = None, **_: Any) -> None:
        """Atualiza a referência da imagem de Entrada A para comparativos e histograma."""
        if image is not None:
            self._img_slot_a.src = _bytes_to_data_uri(_ndarray_to_png_bytes(image))
            is_rgb_a = bool(image.ndim == 3 and image.shape[2] >= 3)
            self._histogram_chart_a.set_data(
                image,
                title="Histograma — Entrada A",
                is_rgb=is_rgb_a,
            )
            # Atualiza também intermediário padrão se modo triplo
            if self._intermediate_image is None:
                if image.ndim == 3:
                    self._intermediate_image = to_grayscale(image, GrayscaleMethod.LUMINANCE)
                else:
                    self._intermediate_image = image
                self._img_inter.src = _bytes_to_data_uri(_ndarray_to_png_bytes(self._intermediate_image))
        else:
            self._img_slot_a.src = TRANSPARENT_PIXEL_PNG_URI
            self._img_inter.src = TRANSPARENT_PIXEL_PNG_URI
            self._intermediate_image = None

        if self.display_mode != DisplayMode.RESULT_ONLY:
            self._render_canvas()
            self._safe_update()

    def _on_image_b_changed(self, **_: Any) -> None:
        """Atualização de imagem B."""
        if self.display_mode != DisplayMode.RESULT_ONLY:
            self._render_canvas()
            self._safe_update()

    def _update_result_image_control(self) -> None:
        """Converte o resultado atual em Data URI para o controle Flet."""
        if self._result_image is not None:
            png_bytes = _ndarray_to_png_bytes(self._result_image)
            self._img_result.src = _bytes_to_data_uri(png_bytes)
        else:
            self._img_result.src = TRANSPARENT_PIXEL_PNG_URI

    # -----------------------------------------------------------------------
    # Diálogos Auxiliares de Zoom e Auditoria
    # -----------------------------------------------------------------------

    def _zoom_histogram_a(self) -> None:
        page = self._get_active_page()
        if page is not None and self.session_state.image_a is not None:
            open_histogram_zoom_dialog(
                page=page,
                title="Histograma — Entrada A (Alta Resolução)",
                data=self.session_state.image_a,
            )

    def _zoom_histogram_res(self) -> None:
        page = self._get_active_page()
        if page is not None and self._result_image is not None:
            open_histogram_zoom_dialog(
                page=page,
                title="Histograma — Resultado (Alta Resolução)",
                data=self._result_image,
                is_quantized=True,
            )

    def _open_inspector_from_canvas(self) -> None:
        page = self._get_active_page()
        if page is None:
            return
        res = self._result_image
        raw = self.session_state.image_a
        if res is None or raw is None:
            return
        gray = self._intermediate_image if self._intermediate_image is not None else (
            to_grayscale(raw, GrayscaleMethod.LUMINANCE) if raw.ndim == 3 else raw
        )
        bits = self._metrics.get("bits", 4)
        tech = self._metrics.get("technique", "Uniforme")
        open_inspector_dialog(
            page=page,
            raw_image=raw,
            gray_image=gray,
            quantized_image=res,
            bits=bits,
            technique=tech,
            method=GrayscaleMethod.LUMINANCE,
        )

    # -----------------------------------------------------------------------
    # Construção de Toolbar e Modos
    # -----------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Gera os botões de alternância de modo, zoom e download."""
        mode_buttons = [
            ft.Button(
                content="Resultado",
                icon=ft.Icons.IMAGE,
                bgcolor=theme.PRIMARY if self.display_mode == DisplayMode.RESULT_ONLY else ft.Colors.SURFACE_CONTAINER,
                color="#FFFFFF" if self.display_mode == DisplayMode.RESULT_ONLY else ft.Colors.ON_SURFACE,
                on_click=lambda _: self.set_display_mode(DisplayMode.RESULT_ONLY),
            ),
            ft.Button(
                content="Lado a Lado",
                icon=ft.Icons.COMPARE,
                bgcolor=theme.PRIMARY if self.display_mode == DisplayMode.SIDE_BY_SIDE else ft.Colors.SURFACE_CONTAINER,
                color="#FFFFFF" if self.display_mode == DisplayMode.SIDE_BY_SIDE else ft.Colors.ON_SURFACE,
                on_click=lambda _: self.set_display_mode(DisplayMode.SIDE_BY_SIDE),
            ),
            ft.Button(
                content="Triplo",
                icon=ft.Icons.VIEW_COLUMN,
                bgcolor=theme.PRIMARY if self.display_mode == DisplayMode.TRIPLE else ft.Colors.SURFACE_CONTAINER,
                color="#FFFFFF" if self.display_mode == DisplayMode.TRIPLE else ft.Colors.ON_SURFACE,
                on_click=lambda _: self.set_display_mode(DisplayMode.TRIPLE),
            ),
            ft.Button(
                content="📊 Painel Analítico & Histogramas",
                icon=ft.Icons.QUERY_STATS,
                bgcolor=theme.PRIMARY if self.display_mode == DisplayMode.ANALYTICS else ft.Colors.SURFACE_CONTAINER,
                color="#FFFFFF" if self.display_mode == DisplayMode.ANALYTICS else ft.Colors.ON_SURFACE,
                on_click=lambda _: self.set_display_mode(DisplayMode.ANALYTICS),
            ),
        ]

        if self._comparison_results:
            mode_buttons.append(
                ft.Button(
                    content="⚡ Comparador Quádruplo",
                    icon=ft.Icons.GRID_VIEW,
                    bgcolor=theme.PRIMARY if self.display_mode == DisplayMode.COMPARISON else ft.Colors.SURFACE_CONTAINER,
                    color="#FFFFFF" if self.display_mode == DisplayMode.COMPARISON else ft.Colors.ON_SURFACE,
                    on_click=lambda _: self.set_display_mode(DisplayMode.COMPARISON),
                )
            )

        action_buttons: list[ft.Control] = []
        if self.on_promote is not None:
            self._btn_promote_canvas = ft.Button(
                content=ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.SUBDIRECTORY_ARROW_RIGHT, size=16),
                        ft.Text("Usar como Entrada A", size=theme.FONT_CAPTION),
                    ],
                    spacing=4,
                    tight=True,
                ),
                tooltip="Promover resultado atual para Entrada A (Pipeline)",
                disabled=(self._result_image is None),
                on_click=lambda _: self.on_promote() if self.on_promote else None,
            )
            action_buttons.append(self._btn_promote_canvas)

        action_buttons.extend([
            ft.IconButton(
                icon=ft.Icons.ZOOM_IN,
                tooltip="Ampliar com Zoom 10×",
                on_click=lambda _: self._open_zoom_dialog(),
            ),
            ft.IconButton(
                icon=ft.Icons.DOWNLOAD,
                tooltip="Baixar Imagem Resultante",
                on_click=lambda _: self._trigger_download(),
            ),
        ])

        self._toolbar_row.controls = [
            ft.Row(controls=mode_buttons, spacing=6, wrap=True),
            ft.Row(controls=action_buttons, spacing=4, wrap=True),
        ]

    def update_responsive_layout(self, width: float | None = None, height: float | None = None) -> None:
        """Atualiza a renderização do canvas conforme as dimensões da viewport."""
        self._page_width = width
        self._page_height = height
        self._render_canvas()
        self._safe_update()

    def _is_mobile_screen(self) -> bool:
        if getattr(self, "_page_width", None) is not None:
            return float(self._page_width) < 600
        page = self._get_active_page()
        if page is not None and getattr(page, "width", None) is not None:
            return float(page.width) < 600
        return False

    def _open_zoom_dialog(
        self,
        target_image: np.ndarray | None = None,
        title: str | None = None,
        custom_bytes: bytes | None = None,
    ) -> None:
        """Abre modal de zoom para a imagem selecionada ou bytes de figura fornecidos."""
        page = self._get_active_page()
        if page is None:
            return
        if custom_bytes is not None:
            open_zoom_dialog(page, title=title or "Visualização em Alta Definição", image_bytes=custom_bytes)
            return
        img = target_image if target_image is not None else self._result_image
        if img is None:
            return
        t = title or f"Resultado — {self._result_name}"
        open_zoom_dialog(page, title=t, image_bytes=_ndarray_to_png_bytes(img))

    def _trigger_download(self) -> None:
        """Executa ou repassa o download da imagem resultante ou figura analítica consolidada."""
        if self.display_mode == DisplayMode.ANALYTICS and self._figure_bytes:
            bits = self._metrics.get("bits", 4)
            name = f"painel_analitico_histogramas_{bits}bits.png"
            if self.on_download is not None:
                self.on_download(self._figure_bytes, name)
            else:
                page = self._get_active_page()
                if page is not None:
                    page.open(
                        ft.SnackBar(content=ft.Text(f"Download iniciado: {name}"), bgcolor=theme.SUCCESS)
                    )
            return

        if self._result_image is None:
            return
        png_bytes = _ndarray_to_png_bytes(self._result_image)
        if self.on_download is not None:
            self.on_download(png_bytes, self._result_name)
        else:
            page = self._get_active_page()
            if page is not None:
                page.open(
                    ft.SnackBar(content=ft.Text(f"Download iniciado: {self._result_name}"), bgcolor=theme.SUCCESS)
                )

    # -----------------------------------------------------------------------
    # Views Especializadas: Painel Analítico & Comparador Múltiplo
    # -----------------------------------------------------------------------

    def _get_active_algorithm_info(self) -> tuple[str, str]:
        """
        Determina o nome completo e o badge resumido do algoritmo/técnica do resultado atual.

        Returns:
            Tupla (nome_completo, badge_curto)
        """
        if self._metrics:
            # 1. Par explícito nos metadados
            if "algorithm" in self._metrics and "algorithm_badge" in self._metrics:
                return str(self._metrics["algorithm"]), str(self._metrics["algorithm_badge"])

            # 2. Comparação múltipla vencedora
            if self._metrics.get("is_comparison"):
                winner_title = self._metrics.get("winner_title") or self._metrics.get("winner", "")
                bits = self._metrics.get("bits")
                bits_suffix = f" ({bits} bits)" if bits is not None else ""
                badge_suffix = f" ({bits}b)" if bits is not None else ""
                full = f"Comparação — Vencedor: {winner_title}{bits_suffix}" if winner_title else "Comparação Múltipla"
                badge = f"{winner_title}{badge_suffix}" if winner_title else "Comparação"
                return full, badge

            # 3. Quantização de algoritmo único
            tech = self._metrics.get("technique")
            bits = self._metrics.get("bits")
            if tech:
                bits_suffix = f" ({bits} bits)" if bits is not None else ""
                badge_suffix = f" ({bits}b)" if bits is not None else ""
                return f"{tech}{bits_suffix}", f"{tech}{badge_suffix}"

            # 4. Operação binária / aritmética / lógica
            op = self._metrics.get("operation")
            if op:
                alpha = self._metrics.get("alpha")
                alpha_str = f" (α={alpha:.2f})" if alpha is not None else ""
                return f"Operação {op}{alpha_str}", f"{op}{alpha_str}"

            # 5. Método de conversão cinza
            method = self._metrics.get("method")
            if method:
                return f"Conversão Cinza ({method})", str(method)

            if "algorithm" in self._metrics:
                alg = str(self._metrics["algorithm"])
                return alg, alg

        # 6. Fallback para o último módulo aplicado
        if self.session_state.last_applied_module_name:
            name = self.session_state.last_applied_module_name
            return name, name

        return "", ""

    def _build_analytics_view(self, is_mobile: bool) -> ft.Control:
        """Constrói a visualização com a Figura Analítica Consolidada (Imagens + Histogramas Integrados num arquivo único)."""
        algo_name, algo_badge = self._get_active_algorithm_info()

        self._analytics_badges_row.controls.clear()
        if algo_badge or algo_name:
            self._analytics_badges_row.controls.append(
                theme.metric_badge("Algoritmo", algo_badge or algo_name, color=theme.PRIMARY_LIGHT)
            )
        if "mse" in self._metrics:
            self._analytics_badges_row.controls.append(
                theme.metric_badge("MSE", f"{self._metrics['mse']:.2f}", color=theme.ACCENT)
            )
        if "psnr" in self._metrics:
            val = self._metrics["psnr"]
            psnr_str = "Inf dB" if np.isinf(val) else f"{val:.2f} dB"
            self._analytics_badges_row.controls.append(
                theme.metric_badge("PSNR", psnr_str, color=theme.PRIMARY_LIGHT)
            )
        if "unique_levels" in self._metrics:
            self._analytics_badges_row.controls.append(
                theme.metric_badge("Níveis", str(self._metrics["unique_levels"]), color=theme.INFO)
            )
        if "time_ms" in self._metrics:
            self._analytics_badges_row.controls.append(
                theme.metric_badge("Tempo", f"{self._metrics['time_ms']:.1f} ms", color=theme.SUCCESS)
            )

        # Garante a geração de figure_bytes se ainda não foi gerada
        if not self._figure_bytes:
            self._figure_bytes = self._generate_analytics_figure_bytes()
        if self._figure_bytes:
            self._img_analytics.src = _bytes_to_data_uri(self._figure_bytes)

        fig_zoom_title = f"Painel Analítico HD — {algo_name}" if algo_name else "Figura Analítica Consolidada (Imagens + Histogramas)"
        btn_zoom_figure = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ZOOM_IN, size=18, color=ft.Colors.WHITE),
                    ft.Text("🔍 Ampliar Painel HD", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
                ],
                spacing=6,
            ),
            bgcolor=theme.PRIMARY,
            tooltip="Ampliar Figura Analítica Completa em Resolução Ultra-HD (2200×1450)",
            on_click=lambda _: self._open_zoom_dialog(
                target_image=None,
                title=fig_zoom_title,
                custom_bytes=self._figure_bytes,
            ),
        )

        btn_download_figure = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.DOWNLOAD, size=18),
                    ft.Text("📥 Baixar Arquivo Único (PNG)"),
                ],
                spacing=6,
            ),
            tooltip="Baixar imagem consolidada contendo imagens e histogramas integrados",
            on_click=lambda _: self._trigger_download(),
        )

        btn_inspector = ft.Button(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ANALYTICS_OUTLINED, size=18, color=ft.Colors.WHITE),
                    ft.Text("🔬 Entranhas do Processo", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
                ],
                spacing=6,
            ),
            bgcolor=theme.PRIMARY_DARK,
            on_click=lambda _: self._open_inspector_from_canvas(),
        )

        actions = [btn_zoom_figure, btn_download_figure, btn_inspector]
        top_bar = ft.Row(
            controls=[
                self._analytics_badges_row,
                ft.Row(controls=actions, spacing=6, wrap=True),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            wrap=True,
        )

        fig_card_label = (
            f"Figura Analítica Consolidada — {algo_name}"
            if algo_name
            else "Figura Analítica Consolidada — Imagens e Histogramas Integrados em Arquivo Único de Alta Resolução"
        )
        figure_card = self._wrap_interactive_image(
            self._img_analytics,
            label=fig_card_label,
            on_click_fn=lambda: self._open_zoom_dialog(
                target_image=None,
                title=fig_zoom_title,
                custom_bytes=self._figure_bytes,
            ),
            algorithm_badge=algo_badge if algo_badge else None,
        )

        return ft.Column(
            controls=[
                top_bar,
                ft.Divider(height=1),
                figure_card,
            ],
            spacing=10,
            expand=True,
        )

    def _build_comparison_view(self, is_mobile: bool) -> ft.Control:
        """Constrói a visualização em grade 2x2 com miniaturas, badges de PSNR/MSE e vencedor destacado."""
        if not self._comparison_results:
            return ft.Column(
                controls=[
                    ft.Icon(ft.Icons.GRID_VIEW, size=48, color=theme.TEXT_SECONDARY),
                    ft.Text("Nenhuma comparação múltipla executada ainda.", italic=True, color=theme.TEXT_SECONDARY),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            )

        cards: list[ft.Control] = []
        for name, item in self._comparison_results.items():
            is_winner = bool(name == self._comparison_winner)
            img_arr = item["image"]
            png_bytes = _ndarray_to_png_bytes(img_arr)
            data_uri = _bytes_to_data_uri(png_bytes)

            header_controls: list[ft.Control] = [
                ft.Text(
                    item["title"],
                    weight=ft.FontWeight.BOLD,
                    size=theme.FONT_CAPTION,
                    color=theme.PRIMARY_LIGHT if is_winner else ft.Colors.ON_SURFACE,
                    overflow=ft.TextOverflow.ELLIPSIS,
                )
            ]

            if is_winner:
                winner_badge = ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.EMOJI_EVENTS, size=14, color=ft.Colors.AMBER_ACCENT_400),
                            ft.Text(f"🏆 Vencedor: {item['psnr']:.2f} dB", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.AMBER_ACCENT_400),
                        ],
                        spacing=4,
                    ),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=4,
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2) if hasattr(ft, "Padding") else 4,
                )
                header_controls.append(winner_badge)

            header_row = ft.Row(
                controls=header_controls,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

            img_ctrl = ft.Image(
                src=data_uri,
                fit=getattr(ft.BoxFit, "CONTAIN", None),
                expand=True,
            )

            metrics_row = ft.Row(
                controls=[
                    theme.metric_badge("PSNR", f"{item['psnr']:.2f} dB", color=theme.PRIMARY_LIGHT if is_winner else theme.INFO),
                    theme.metric_badge("MSE", f"{item['mse']:.2f}", color=theme.ACCENT),
                    theme.metric_badge("Tempo", f"{item['time_ms']:.0f} ms", color=theme.SUCCESS),
                ],
                spacing=4,
                wrap=True,
            )

            zoom_btn = ft.IconButton(
                icon=ft.Icons.ZOOM_IN,
                icon_size=20,
                tooltip=f"Ampliar {item['title']}",
                on_click=lambda _, arr=img_arr, t=item["title"]: self._open_zoom_dialog(arr, t),
            )

            footer_row = ft.Row(
                controls=[
                    metrics_row,
                    zoom_btn,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

            card_border = ft.Border.all(2, theme.PRIMARY) if is_winner and hasattr(ft, "Border") else (
                ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None
            )

            card = ft.Container(
                content=ft.Column(
                    controls=[
                        header_row,
                        ft.Container(
                            content=img_ctrl,
                            expand=True,
                            alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                            on_click=lambda _, arr=img_arr, t=item["title"]: self._open_zoom_dialog(arr, t),
                            ink=True,
                        ),
                        footer_row,
                    ],
                    spacing=6,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    expand=True,
                ),
                bgcolor=ft.Colors.SURFACE_CONTAINER,
                border=card_border,
                border_radius=8,
                padding=8,
                expand=True,
            )
            cards.append(card)

        if is_mobile or len(cards) <= 1:
            return ft.Column(controls=cards, spacing=8, expand=True, scroll=ft.ScrollMode.AUTO)
        elif len(cards) == 2:
            return ft.Row(controls=cards, spacing=8, expand=True)
        else:
            row1 = ft.Row(controls=cards[:2], spacing=8, expand=True)
            row2 = ft.Row(controls=cards[2:4], spacing=8, expand=True)
            return ft.Column(controls=[row1, row2], spacing=8, expand=True)

    # -----------------------------------------------------------------------
    # Renderização do Canvas Central
    # -----------------------------------------------------------------------

    def _render_canvas(self) -> None:
        """Renderiza os contêineres de imagem de acordo com o modo e viewport."""
        if self._result_image is None and not self._comparison_results:
            self._display_area.content = ft.Column(
                controls=[
                    ft.Icon(ft.Icons.IMAGE_SEARCH, size=48, color=theme.TEXT_SECONDARY),
                    ft.Text(
                        "Nenhum resultado processado ainda",
                        size=theme.FONT_BODY,
                        color=theme.TEXT_SECONDARY,
                        italic=True,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            )
            return

        is_mobile = self._is_mobile_screen()
        algo_name, algo_badge = self._get_active_algorithm_info()
        res_zoom_title = f"Imagem Processada — {algo_name}" if algo_name else "Imagem Processada"

        if self.display_mode == DisplayMode.ANALYTICS:
            self._display_area.content = self._build_analytics_view(is_mobile)

        elif self.display_mode == DisplayMode.COMPARISON:
            self._display_area.content = self._build_comparison_view(is_mobile)

        elif self.display_mode == DisplayMode.RESULT_ONLY:
            self._display_area.content = self._wrap_interactive_image(
                self._img_result,
                label="Imagem Processada",
                on_click_fn=lambda: self._open_zoom_dialog(self._result_image, res_zoom_title),
                algorithm_badge=algo_badge if algo_badge else None,
            )

        elif self.display_mode == DisplayMode.SIDE_BY_SIDE:
            card_a = self._wrap_interactive_image(
                self._img_slot_a,
                label="Entrada A",
                on_click_fn=lambda: self._open_zoom_dialog(self.session_state.image_a, "Entrada A"),
            )
            card_res = self._wrap_interactive_image(
                self._img_result,
                label="Imagem Processada",
                on_click_fn=lambda: self._open_zoom_dialog(self._result_image, res_zoom_title),
                algorithm_badge=algo_badge if algo_badge else None,
            )

            if is_mobile:
                self._display_area.content = ft.Column(
                    controls=[card_a, card_res],
                    spacing=8,
                    expand=True,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                )
            else:
                self._display_area.content = ft.Row(
                    controls=[card_a, card_res],
                    spacing=8,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                )

        elif self.display_mode == DisplayMode.TRIPLE:
            card_a = self._wrap_interactive_image(
                self._img_slot_a,
                label="Entrada A",
                on_click_fn=lambda: self._open_zoom_dialog(self.session_state.image_a, "Entrada A"),
            )
            card_mid = self._wrap_interactive_image(
                self._img_inter,
                label="Intermediário (Cinza)",
                on_click_fn=lambda: self._open_zoom_dialog(self._intermediate_image, "Intermediário"),
            )
            card_res = self._wrap_interactive_image(
                self._img_result,
                label="Imagem Processada",
                on_click_fn=lambda: self._open_zoom_dialog(self._result_image, res_zoom_title),
                algorithm_badge=algo_badge if algo_badge else None,
            )

            if is_mobile:
                self._display_area.content = ft.Column(
                    controls=[card_a, card_mid, card_res],
                    spacing=8,
                    expand=True,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                )
            else:
                self._display_area.content = ft.Row(
                    controls=[card_a, card_mid, card_res],
                    spacing=8,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                )

    def _wrap_interactive_image(
        self,
        img_control: ft.Image,
        label: str,
        on_click_fn: Callable[[], None],
        algorithm_badge: str | None = None,
    ) -> ft.Container:
        """Envolve um controle de imagem com rótulo, badge de algoritmo, contêiner adaptativo e evento de zoom ao clicar."""
        header_controls: list[ft.Control] = [
            ft.Text(label, size=theme.FONT_CAPTION, weight=ft.FontWeight.BOLD, color=theme.TEXT_SECONDARY),
        ]
        if algorithm_badge:
            header_controls.append(
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.AUTO_AWESOME, size=13, color=theme.PRIMARY_LIGHT),
                            ft.Text(algorithm_badge, size=11, weight=ft.FontWeight.BOLD, color=theme.PRIMARY_LIGHT),
                        ],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border=ft.Border.all(1, theme.PRIMARY) if hasattr(ft, "Border") else None,
                    border_radius=4,
                    padding=ft.Padding.symmetric(horizontal=8, vertical=2) if hasattr(ft, "Padding") else 4,
                )
            )

        header_row = ft.Row(
            controls=header_controls,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
            wrap=True,
        )

        tooltip_text = (
            f"Clique para ampliar {label} ({algorithm_badge})"
            if algorithm_badge
            else f"Clique para ampliar {label}"
        )

        return ft.Container(
            content=ft.Column(
                controls=[
                    header_row,
                    ft.Container(
                        content=img_control,
                        expand=True,
                        alignment=getattr(ft.Alignment, "CENTER", ft.Alignment(0, 0)) if hasattr(ft, "Alignment") else None,
                    ),
                ],
                spacing=4,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=8,
            padding=6,
            expand=True,
            ink=True,
            on_click=lambda _: on_click_fn(),
            tooltip=tooltip_text,
        )


"""
telemetry_panel.py — Painel Lateral / Gaveta Retrátil de Telemetria e Auditoria Matemática.

Centraliza as métricas didáticas e auditoria científica de Processamento Digital de Imagens:
- Badges limpos e compactos para MSE, PSNR, Níveis Únicos e Tempo de Execução.
- Botão "🔬 Entranhas do Processo" para auditoria profunda via open_inspector_dialog.
- Seção expansível para Histograma Nativo em tempo real via NativeHistogramChart.
- Layout adaptativo: Card lateral estético em Desktop; ExpansionTile compacto em Mobile (< 600px).
- Sincronização automática com SessionState (evento result_changed).
"""

from typing import Any, Callable
import flet as ft
import numpy as np

from src.core.grayscale import GrayscaleMethod
from src.core.quantization import QuantizationTechnique
from src.ui import theme
from src.ui.components.histogram_chart import NativeHistogramChart
from src.ui.dialogs import open_inspector_dialog
from src.ui.state.session_state import (
    EVENT_RESULT_CHANGED,
    SessionState,
    get_session_state,
)


class TelemetryPanel(ft.Container):
    """
    Componente responsivo de exibição de métricas analíticas e telemetria do resultado.
    """

    def __init__(
        self,
        session_state: SessionState | None = None,
        page: ft.Page | None = None,
        on_open_inspector: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o TelemetryPanel.

        Args:
            session_state: Instância do SessionState.
            page: Página Flet ativa.
            on_open_inspector: Callback opcional disparado ao clicar em "Entranhas do Processo".
            **kwargs: Parâmetros repassados ao ft.Container.
        """
        self.session_state = session_state or get_session_state()
        self._page = page
        self.on_open_inspector = on_open_inspector

        # Armazenamento de métricas e contexto do pipeline
        self._metrics: dict[str, Any] = {}
        self._result_image: np.ndarray | None = None

        # Contexto opcional para auditoria das Entranhas do Processo
        self._raw_image: np.ndarray | None = None
        self._gray_image: np.ndarray | None = None
        self._bits: int = 8
        self._technique: QuantizationTechnique | str = QuantizationTechnique.UNIFORM
        self._gray_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE

        # Inscrição reativa no SessionState
        self._unsub_result = self.session_state.subscribe(EVENT_RESULT_CHANGED, self._on_result_changed)

        # Controles internos
        self._badges_row = ft.Row(
            spacing=8,
            wrap=True,
            alignment=ft.MainAxisAlignment.START,
        )

        self._btn_inspector = ft.Button(
            content="🔬 Entranhas do Processo",
            icon=ft.Icons.ANALYTICS_OUTLINED,
            on_click=lambda _: self._trigger_inspector(),
            bgcolor=theme.PRIMARY_DARK,
            color="#FFFFFF",
            visible=False,
        )

        self._histogram_chart = NativeHistogramChart(
            title="Histograma do Resultado",
            chart_height=140,
        )

        self._content_column = ft.Column(
            spacing=10,
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

        self._render_panel()

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
    # Métodos Públicos de Atualização
    # -----------------------------------------------------------------------

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        """Atualiza o dicionário de métricas e redesenha os badges."""
        self._metrics = dict(metrics)
        self._render_badges()
        self._safe_update()

    def set_pipeline_context(
        self,
        raw_image: np.ndarray | None,
        gray_image: np.ndarray | None,
        bits: int = 8,
        technique: QuantizationTechnique | str = QuantizationTechnique.UNIFORM,
        method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    ) -> None:
        """Define o contexto para habilitação do modal didático das Entranhas do Processo."""
        self._raw_image = raw_image
        self._gray_image = gray_image
        self._bits = bits
        self._technique = technique
        self._gray_method = method
        self._btn_inspector.visible = bool(raw_image is not None and gray_image is not None)
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
        """Reage à alteração de resultado e métricas no SessionState."""
        self._result_image = image
        if metrics is not None:
            self._metrics = dict(metrics)

        # Atualiza o gráfico de histograma se houver imagem
        if image is not None:
            is_rgb = bool(image.ndim == 3 and image.shape[2] >= 3)
            self._histogram_chart.set_data(
                image,
                title="Histograma do Resultado",
                is_rgb=is_rgb,
            )
            self._histogram_chart.visible = True
        else:
            self._histogram_chart.visible = False

        self._render_badges()
        self._render_panel()
        self._safe_update()

    # -----------------------------------------------------------------------
    # Renderização de Badges e Layout
    # -----------------------------------------------------------------------

    def _render_badges(self) -> None:
        """Gera os badges limpos para as métricas formatadas."""
        self._badges_row.controls.clear()

        if not self._metrics and self._result_image is None:
            self._badges_row.controls.append(
                ft.Text(
                    "Nenhuma métrica computada",
                    size=theme.FONT_CAPTION,
                    color=theme.TEXT_SECONDARY,
                    italic=True,
                )
            )
            return

        # MSE
        if "mse" in self._metrics:
            mse_val = self._metrics["mse"]
            formatted = f"{mse_val:.2f}" if isinstance(mse_val, (int, float)) else str(mse_val)
            self._badges_row.controls.append(theme.metric_badge("MSE", formatted, color=theme.ACCENT))

        # PSNR
        if "psnr" in self._metrics:
            psnr_val = self._metrics["psnr"]
            if isinstance(psnr_val, (int, float)):
                formatted = "Inf dB" if np.isinf(psnr_val) else f"{psnr_val:.2f} dB"
            else:
                formatted = str(psnr_val)
            self._badges_row.controls.append(theme.metric_badge("PSNR", formatted, color=theme.PRIMARY_LIGHT))

        # Níveis Únicos
        if "unique_levels" in self._metrics:
            levels = self._metrics["unique_levels"]
            self._badges_row.controls.append(theme.metric_badge("Níveis", str(levels), color=theme.INFO))
        elif self._result_image is not None:
            levels = len(np.unique(self._result_image))
            self._badges_row.controls.append(theme.metric_badge("Níveis", str(levels), color=theme.INFO))

        # Tempo de Execução
        if "time_ms" in self._metrics:
            t = self._metrics["time_ms"]
            formatted = f"{t:.1f} ms" if isinstance(t, (int, float)) else str(t)
            self._badges_row.controls.append(theme.metric_badge("Tempo", formatted, color=theme.SUCCESS))

        # Métricas adicionais arbitrárias
        for k, v in self._metrics.items():
            if k not in ("mse", "psnr", "unique_levels", "time_ms"):
                lbl = k.replace("_", " ").title()
                val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                self._badges_row.controls.append(theme.metric_badge(lbl, val_str, color=ft.Colors.ON_SURFACE))

    def update_responsive_layout(self, width: float | None = None, height: float | None = None) -> None:
        """Atualiza a renderização do painel conforme as dimensões da viewport."""
        self._page_width = width
        self._page_height = height
        self._render_panel()
        self._safe_update()

    def _is_mobile(self) -> bool:
        if getattr(self, "_page_width", None) is not None:
            return float(self._page_width) < 600
        page = self._get_active_page()
        if page is not None and getattr(page, "width", None) is not None:
            return float(page.width) < 600
        return False

    def _trigger_inspector(self) -> None:
        """Dispara o modal didático das Entranhas do Processo."""
        if self.on_open_inspector is not None:
            self.on_open_inspector()
            return

        page = self._get_active_page()
        if (
            page is not None
            and self._raw_image is not None
            and self._gray_image is not None
            and self._result_image is not None
        ):
            open_inspector_dialog(
                page=page,
                raw_image=self._raw_image,
                gray_image=self._gray_image,
                quantized_image=self._result_image,
                bits=self._bits,
                technique=self._technique,
                method=self._gray_method,
            )

    def _render_panel(self) -> None:
        """Monta o layout, utilizando ExpansionTile no mobile e Card direto no desktop."""
        self._content_column.controls.clear()

        header_title = ft.Row(
            controls=[
                ft.Icon(ft.Icons.QUERY_STATS, size=20, color=theme.PRIMARY_LIGHT),
                ft.Text(
                    "Telemetria e Auditoria",
                    size=theme.FONT_SUBTITLE,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.ON_SURFACE,
                ),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        inner_body = ft.Column(
            controls=[
                self._badges_row,
                self._btn_inspector,
                self._histogram_chart,
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        if self._is_mobile():
            # No Mobile, encapsula em ExpansionTile para não empurrar a imagem para fora
            expansion = ft.ExpansionTile(
                title=header_title,
                controls=[inner_body],
                expanded=False,
            )
            self._content_column.controls.append(expansion)
        else:
            # No Desktop, layout aberto direto
            self._content_column.controls.extend([
                header_title,
                inner_body,
            ])


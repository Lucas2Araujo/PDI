"""
grayscale_module.py — Módulo Didático de Conversão em Tons de Cinza e Isolamento de Canais.

Implementa o módulo visual e funcional para:
- Conversão por Luminância Fisiológica Ponderada (Padrão ITU-R BT.601).
- Conversão por Média Aritmética Simples (R+G+B)/3.
- Isolamento dos Canais Primários (Vermelho R, Verde G, Azul B).
- Exibição de cards didáticos contendo as equações matemáticas e fundamentação biológica.
"""

import time
from typing import Any
import flet as ft
import numpy as np

from src.core.grayscale import (
    GrayscaleMethod,
    isolate_channel_rgb,
    method_label,
    to_grayscale,
)
from src.ui import theme
from src.ui.common import _GRAYSCALE_DETAILS
from src.ui.modules.base_module import BasePDIModule


class GrayscaleModule(BasePDIModule):
    """
    Módulo didático para estudo de conversão de imagens coloridas para escala de cinza e canais.
    """

    title = "Conversão para Tons de Cinza & Canais"
    description = (
        "Estudo comparativo entre a ponderação espectral fisiológica (ITU-R BT.601, sensibilidade "
        "humana aos cones verde, vermelho e azul), a média aritmética simples e o isolamento puro "
        "dos canais RGB da matriz cromática."
    )
    requires_second_input = False
    supports_scalar_mode = False

    def __init__(self, **kwargs: Any) -> None:
        self._method = GrayscaleMethod.LUMINANCE
        self._isolate_rgb: bool = True

        self._dd_method: ft.Dropdown | None = None
        self._formula_card: ft.Container | None = None
        self._txt_formula_title: ft.Text | None = None
        self._txt_formula_desc: ft.Text | None = None
        self._txt_formula_eq: ft.Text | None = None
        self._switch_isolate: ft.Switch | None = None

        super().__init__(**kwargs)

    def build_controls(self) -> list[ft.Control]:
        options = [
            ft.dropdown.Option(
                key=GrayscaleMethod.LUMINANCE.name,
                text="Luminância Ponderada (ITU-R BT.601)",
            ),
            ft.dropdown.Option(
                key=GrayscaleMethod.AVERAGE.name,
                text="Média Aritmética Simples (R+G+B)/3",
            ),
            ft.dropdown.Option(
                key=GrayscaleMethod.CHANNEL_R.name,
                text="Isolamento Canal Vermelho (R)",
            ),
            ft.dropdown.Option(
                key=GrayscaleMethod.CHANNEL_G.name,
                text="Isolamento Canal Verde (G)",
            ),
            ft.dropdown.Option(
                key=GrayscaleMethod.CHANNEL_B.name,
                text="Isolamento Canal Azul (B)",
            ),
        ]

        self._dd_method = ft.Dropdown(
            label="Método de Conversão / Canal",
            value=self._method.name,
            options=options,
            content_padding=ft.Padding.symmetric(horizontal=14, vertical=12) if hasattr(ft, "Padding") else 12,
            border_radius=8,
            on_select=self._on_method_changed,
        )

        self._switch_isolate = ft.Switch(
            label="Renderizar canais em cor real RGB (ex: [R,0,0])",
            value=self._isolate_rgb,
            on_change=self._on_switch_changed,
            visible=self._method in (GrayscaleMethod.CHANNEL_R, GrayscaleMethod.CHANNEL_G, GrayscaleMethod.CHANNEL_B),
        )

        details = _GRAYSCALE_DETAILS[self._method]
        self._txt_formula_title = ft.Text(
            details["title"],
            weight=ft.FontWeight.BOLD,
            size=theme.FONT_CAPTION,
            color=theme.PRIMARY_LIGHT,
        )
        self._txt_formula_eq = ft.Text(
            details["formula"],
            size=theme.FONT_BODY,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.ON_SURFACE,
        )
        self._txt_formula_desc = ft.Text(
            details["desc"],
            size=theme.FONT_CAPTION,
            color=theme.TEXT_SECONDARY,
        )

        self._formula_card = ft.Container(
            content=ft.Column(
                controls=[
                    self._txt_formula_title,
                    self._txt_formula_eq,
                    self._txt_formula_desc,
                ],
                spacing=4,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=10,
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT) if hasattr(ft, "Border") else None,
        )

        return [
            self._dd_method,
            self._switch_isolate,
            self._formula_card,
        ]

    def _on_method_changed(self, e: ft.ControlEvent) -> None:
        key = e.control.value
        self._method = GrayscaleMethod[key]
        details = _GRAYSCALE_DETAILS[self._method]

        if self._txt_formula_title:
            self._txt_formula_title.value = details["title"]
        if self._txt_formula_eq:
            self._txt_formula_eq.value = details["formula"]
        if self._txt_formula_desc:
            self._txt_formula_desc.value = details["desc"]

        is_channel = self._method in (GrayscaleMethod.CHANNEL_R, GrayscaleMethod.CHANNEL_G, GrayscaleMethod.CHANNEL_B)
        if self._switch_isolate:
            self._switch_isolate.visible = is_channel

        try:
            if self._formula_card:
                self._formula_card.update()
            if self._switch_isolate:
                self._switch_isolate.update()
        except (RuntimeError, AssertionError):
            pass

        self.notify_param_changed()

    def _on_switch_changed(self, e: ft.ControlEvent) -> None:
        self._isolate_rgb = bool(e.control.value)
        self.notify_param_changed()

    def get_params(self) -> dict[str, Any]:
        return {
            "method": self._method,
            "isolate_rgb": self._isolate_rgb,
        }

    def process(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray | None = None,
        **params: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        method = params.get("method", self._method)
        isolate_rgb = params.get("isolate_rgb", self._isolate_rgb)

        t0 = time.perf_counter()
        is_channel = method in (GrayscaleMethod.CHANNEL_R, GrayscaleMethod.CHANNEL_G, GrayscaleMethod.CHANNEL_B)

        if is_channel and isolate_rgb and img_a.ndim == 3:
            res = isolate_channel_rgb(img_a, method)
        else:
            res = to_grayscale(img_a, method=method)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        m_label = method_label(method)
        metrics_dict: dict[str, Any] = {
            "mean_intensity": float(np.mean(res)),
            "std_dev": float(np.std(res)),
            "unique_levels": int(len(np.unique(res))),
            "time_ms": elapsed_ms,
            "method": m_label,
            "algorithm": m_label,
            "algorithm_badge": m_label,
        }

        return res, metrics_dict

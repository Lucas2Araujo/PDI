"""
binary_ops_module.py — Módulo Didático de Operações Aritméticas e Lógicas em Imagens.

Implementa o módulo visual e funcional para:
- Álgebra matricial ponto a ponto: Adição, Subtração, Diferença Absoluta, Multiplicação e Divisão.
- Mistura com Transparência (Alpha Blending) ponderada por parâmetro alpha contínuo.
- Operações Lógicas Booleanas Bitwise: AND, OR, XOR e NOT (Negativo).
- Compatibilização dimensional: Estrito, Redimensionamento Bilinear (B para A) ou Recorte Comum.
- Suporte a segundo operando como Imagem B ou Constante Escalar Numérica.
"""

from enum import Enum, auto
import time
from typing import Any, Callable
import flet as ft
import numpy as np

from src.core.binary_ops import (
    ResizeMode,
    add,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    blend,
    divide,
    multiply,
    subtract,
)
from src.ui import theme
from src.ui.modules.base_module import BasePDIModule


class BinaryOpType(Enum):
    """Tipos de operações suportadas pelo módulo binário."""
    ADD = auto()
    SUBTRACT = auto()
    SUBTRACT_ABS = auto()
    BLEND = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    BITWISE_NOT = auto()


class BinaryOpsModule(BasePDIModule):
    """
    Módulo de operações aritméticas, lógicas e blending entre imagens e constantes.
    """

    title = "Operações Aritméticas & Lógicas"
    description = (
        "Álgebra matricial ponto a ponto entre matrizes de imagem ou com constantes numéricas. "
        "Permite detecção de variações por subtração absoluta, fusão suave de imagens via Alpha Blending, "
        "isolamento de regiões de interesse (ROI) por multiplicação e operações lógicas bitwise."
    )
    requires_second_input = True
    supports_scalar_mode = True

    def __init__(
        self,
        on_scalar_mode_changed: Callable[[bool], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.on_scalar_mode_changed = on_scalar_mode_changed
        self._op_type = BinaryOpType.ADD
        self._resize_mode = ResizeMode.STRICT
        self._alpha: float = 0.5
        self._is_scalar_input: bool = False

        self._dd_op: ft.Dropdown | None = None
        self._dd_resize: ft.Dropdown | None = None
        self._slider_alpha: ft.Slider | None = None
        self._txt_alpha: ft.Text | None = None
        self._alpha_container: ft.Container | None = None
        self._switch_scalar: ft.Switch | None = None

        super().__init__(**kwargs)

    def build_controls(self) -> list[ft.Control]:
        op_options = [
            ft.dropdown.Option(key=BinaryOpType.ADD.name, text="Soma (+) — Adição com Saturação"),
            ft.dropdown.Option(key=BinaryOpType.SUBTRACT.name, text="Subtração (-) — A - B com Corte em Zero"),
            ft.dropdown.Option(key=BinaryOpType.SUBTRACT_ABS.name, text="Diferença Absoluta (|A - B|) — Detecção"),
            ft.dropdown.Option(key=BinaryOpType.BLEND.name, text="Alpha Blending — Fusão Ponderada (α)"),
            ft.dropdown.Option(key=BinaryOpType.MULTIPLY.name, text="Multiplicação (*) — Ganho / Máscara"),
            ft.dropdown.Option(key=BinaryOpType.DIVIDE.name, text="Divisão (/) — Correção de Fundo"),
            ft.dropdown.Option(key=BinaryOpType.BITWISE_AND.name, text="E Lógico (Bitwise AND)"),
            ft.dropdown.Option(key=BinaryOpType.BITWISE_OR.name, text="OU Lógico (Bitwise OR)"),
            ft.dropdown.Option(key=BinaryOpType.BITWISE_XOR.name, text="XOR Lógico (Bitwise XOR)"),
            ft.dropdown.Option(key=BinaryOpType.BITWISE_NOT.name, text="NÃO Lógico (Inversão de A)"),
        ]

        self._dd_op = ft.Dropdown(
            label="Operação Binária / Lógica",
            value=self._op_type.name,
            options=op_options,
            content_padding=ft.Padding.symmetric(horizontal=14, vertical=12) if hasattr(ft, "Padding") else 12,
            border_radius=8,
            on_select=self._on_op_changed,
        )

        resize_options = [
            ft.dropdown.Option(key=ResizeMode.STRICT.name, text="Estrito (Rigor Acadêmico — Mesmo Tamanho)"),
            ft.dropdown.Option(key=ResizeMode.RESIZE_B_TO_A.name, text="Redimensionar B para A (Bilinear)"),
            ft.dropdown.Option(key=ResizeMode.CROP_COMMON.name, text="Recorte Menor Comum"),
        ]

        self._dd_resize = ft.Dropdown(
            label="Compatibilização Dimensional",
            value=self._resize_mode.name,
            options=resize_options,
            content_padding=ft.Padding.symmetric(horizontal=14, vertical=12) if hasattr(ft, "Padding") else 12,
            border_radius=8,
            on_select=self._on_resize_changed,
        )

        self._txt_alpha = ft.Text(
            f"Alpha: {self._alpha:.2f} (A: {int(self._alpha * 100)}%, B: {int((1 - self._alpha) * 100)}%)",
            size=theme.FONT_CAPTION,
            weight=ft.FontWeight.BOLD,
            color=theme.PRIMARY_LIGHT,
        )

        self._slider_alpha = ft.Slider(
            min=0.0,
            max=1.0,
            value=self._alpha,
            divisions=20,
            label="{value}",
            on_change=self._on_alpha_changed,
            expand=True,
        )

        self._alpha_container = ft.Container(
            content=ft.Column(
                controls=[
                    self._txt_alpha,
                    self._slider_alpha,
                ],
                spacing=2,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=8,
            padding=8,
            visible=bool(self._op_type == BinaryOpType.BLEND),
        )

        self._switch_scalar = ft.Switch(
            label="Operar com Constante Escalar (em vez de Imagem B)",
            value=self._is_scalar_input,
            on_change=self._on_scalar_switched,
        )

        return [
            self._dd_op,
            self._alpha_container,
            self._dd_resize,
            self._switch_scalar,
        ]

    def _on_op_changed(self, e: ft.ControlEvent) -> None:
        key = e.control.value
        self._op_type = BinaryOpType[key]

        # Ajusta visibilidade do slider de alpha
        if self._alpha_container:
            self._alpha_container.visible = bool(self._op_type == BinaryOpType.BLEND)
            try:
                self._alpha_container.update()
            except (RuntimeError, AssertionError):
                pass

        # Se for NOT, não requer segundo input
        is_unary_not = bool(self._op_type == BinaryOpType.BITWISE_NOT)
        self.requires_second_input = not is_unary_not

        self.notify_param_changed()

    def _on_alpha_changed(self, e: ft.ControlEvent) -> None:
        self._alpha = float(e.control.value)
        if self._txt_alpha:
            self._txt_alpha.value = (
                f"Alpha: {self._alpha:.2f} (A: {int(self._alpha * 100)}%, B: {int((1 - self._alpha) * 100)}%)"
            )
            try:
                self._txt_alpha.update()
            except (RuntimeError, AssertionError):
                pass
        self.notify_param_changed()

    def _on_resize_changed(self, e: ft.ControlEvent) -> None:
        key = e.control.value
        self._resize_mode = ResizeMode[key]
        self.notify_param_changed()

    def _on_scalar_switched(self, e: ft.ControlEvent) -> None:
        self._is_scalar_input = bool(e.control.value)
        if self.on_scalar_mode_changed is not None:
            self.on_scalar_mode_changed(self._is_scalar_input)
        self.notify_param_changed()

    def get_params(self) -> dict[str, Any]:
        return {
            "op_type": self._op_type,
            "resize_mode": self._resize_mode,
            "alpha": self._alpha,
            "is_scalar": self._is_scalar_input,
        }

    def process(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray | None = None,
        **params: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        op_type = params.get("op_type", self._op_type)
        resize_mode = params.get("resize_mode", self._resize_mode)
        alpha = params.get("alpha", self._alpha)

        # Segundo operando pode ser escalar numérico ou imagem
        second_operand = params.get("scalar_val", img_b)

        t0 = time.perf_counter()

        if op_type == BinaryOpType.BITWISE_NOT:
            res = bitwise_not(img_a)
        elif op_type == BinaryOpType.ADD:
            res = add(img_a, second_operand, clip=True, resize_mode=resize_mode)
        elif op_type == BinaryOpType.SUBTRACT:
            res = subtract(img_a, second_operand, absolute=False, clip=True, resize_mode=resize_mode)
        elif op_type == BinaryOpType.SUBTRACT_ABS:
            res = subtract(img_a, second_operand, absolute=True, clip=True, resize_mode=resize_mode)
        elif op_type == BinaryOpType.BLEND:
            if not isinstance(second_operand, np.ndarray):
                raise ValueError("Alpha Blending requer uma segunda imagem (Slot B), não escalar.")
            res = blend(img_a, second_operand, alpha=alpha, resize_mode=resize_mode)
        elif op_type == BinaryOpType.MULTIPLY:
            res = multiply(img_a, second_operand, clip=True, resize_mode=resize_mode)
        elif op_type == BinaryOpType.DIVIDE:
            res = divide(img_a, second_operand, clip=True, resize_mode=resize_mode)
        elif op_type == BinaryOpType.BITWISE_AND:
            res = bitwise_and(img_a, second_operand, resize_mode=resize_mode)
        elif op_type == BinaryOpType.BITWISE_OR:
            res = bitwise_or(img_a, second_operand, resize_mode=resize_mode)
        elif op_type == BinaryOpType.BITWISE_XOR:
            res = bitwise_xor(img_a, second_operand, resize_mode=resize_mode)
        else:
            raise ValueError(f"Operação desconhecida: {op_type}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Métricas de telemetria
        diff_a = np.abs(img_a.astype(np.float64) - res.astype(np.float64))
        mse_a = float(np.mean(diff_a ** 2))
        psnr_a = float(10.0 * np.log10((255.0 ** 2) / mse_a)) if mse_a > 0 else float("inf")

        _OP_LABELS = {
            BinaryOpType.ADD: ("Adição (+)", "Adição (+)"),
            BinaryOpType.SUBTRACT: ("Subtração (-)", "Subtração (-)"),
            BinaryOpType.SUBTRACT_ABS: ("Diferença Absoluta (|A - B|)", "|A - B|"),
            BinaryOpType.BLEND: (f"Alpha Blending (α={alpha:.2f})", f"Blending (α={alpha:.2f})"),
            BinaryOpType.MULTIPLY: ("Multiplicação (*)", "Multiplicação (*)"),
            BinaryOpType.DIVIDE: ("Divisão (/)", "Divisão (/)"),
            BinaryOpType.BITWISE_AND: ("E Lógico (Bitwise AND)", "AND Lógico"),
            BinaryOpType.BITWISE_OR: ("OU Lógico (Bitwise OR)", "OR Lógico"),
            BinaryOpType.BITWISE_XOR: ("XOR Lógico (Bitwise XOR)", "XOR Lógico"),
            BinaryOpType.BITWISE_NOT: ("NÃO Lógico (Inversão NOT)", "NOT Lógico"),
        }
        full_name, badge_name = _OP_LABELS.get(op_type, (op_type.name, op_type.name))

        metrics_dict: dict[str, Any] = {
            "mse": mse_a,
            "psnr": psnr_a,
            "unique_levels": int(len(np.unique(res))),
            "time_ms": elapsed_ms,
            "operation": op_type.name,
            "alpha": alpha,
            "algorithm": full_name,
            "algorithm_badge": badge_name,
        }

        return res, metrics_dict

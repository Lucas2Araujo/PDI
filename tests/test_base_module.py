"""
test_base_module.py — Testes unitários para a classe base abstrata BasePDIModule (src.ui.modules.base_module).
"""

import unittest
from typing import Any
import flet as ft
import numpy as np

from src.ui.modules.base_module import BasePDIModule


class DummyAdditionModule(BasePDIModule):
    """Implementação concreta de teste de BasePDIModule."""

    title = "Soma Aritmética"
    description = "Soma linear entre duas matrizes de imagem com saturação."
    requires_second_input = True
    supports_scalar_mode = True

    def __init__(self, **kwargs: Any) -> None:
        self.slider_bias: ft.Slider | None = None
        super().__init__(**kwargs)

    def build_controls(self) -> list[ft.Control]:
        self.slider_bias = ft.Slider(
            min=0,
            max=100,
            value=10,
            on_change=lambda _: self.notify_param_changed(),
        )
        return [
            ft.Text("Bias Escalar:"),
            self.slider_bias,
        ]

    def get_params(self) -> dict[str, Any]:
        return {
            "bias": self.slider_bias.value if self.slider_bias else 0,
        }

    def process(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray | None = None,
        **params: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        bias = params.get("bias", 0)
        res = np.clip(img_a.astype(np.int32) + int(bias), 0, 255).astype(np.uint8)
        metrics = {"mean_result": float(np.mean(res))}
        return res, metrics


class TestBasePDIModule(unittest.TestCase):
    """Suíte de testes para contrato, interface e layout de BasePDIModule."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """BasePDIModule não pode ser instanciada diretamente sem implementar métodos abstratos."""
        with self.assertRaises(TypeError):
            BasePDIModule()  # type: ignore[abstract]

    def test_concrete_module_instantiation(self) -> None:
        """Módulo concreto herda de ft.Column e BasePDIModule com metadados corretos."""
        module = DummyAdditionModule()
        self.assertIsInstance(module, ft.Column)
        self.assertIsInstance(module, BasePDIModule)
        self.assertEqual(module.title, "Soma Aritmética")
        self.assertTrue(module.requires_second_input)
        self.assertTrue(module.supports_scalar_mode)

    def test_header_rendering_and_badges(self) -> None:
        """render_header deve conter título, descrição e badges condizentes com os metadados."""
        module = DummyAdditionModule()
        header = module.render_header()
        self.assertIsInstance(header, ft.Container)
        self.assertIsInstance(header.content, ft.Column)

        # O primeiro controle do header deve conter o título
        title_row = header.content.controls[0]
        self.assertIsInstance(title_row, ft.Row)
        title_text = title_row.controls[1]
        self.assertEqual(title_text.value, "Soma Aritmética")

    def test_on_param_changed_notification(self) -> None:
        """notify_param_changed deve acionar callback registrado no construtor."""
        changed = False

        def on_change():
            nonlocal changed
            changed = True

        module = DummyAdditionModule(on_param_changed=on_change)
        module.notify_param_changed()
        self.assertTrue(changed)

    def test_get_params_and_process_execution(self) -> None:
        """Execução pura de process() retorna tupla (np.ndarray, dict) com resultados válidos."""
        module = DummyAdditionModule()
        params = module.get_params()
        self.assertEqual(params, {"bias": 10})

        test_img = np.full((10, 10), 100, dtype=np.uint8)
        res, metrics = module.process(test_img, None, **params)

        self.assertEqual(res.shape, (10, 10))
        self.assertEqual(res.dtype, np.uint8)
        self.assertTrue(np.all(res == 110))
        self.assertIn("mean_result", metrics)
        self.assertEqual(metrics["mean_result"], 110.0)

    def test_controls_responsive_constraints(self) -> None:
        """Controles gerados não devem conter larguras fixas rígidas > 300px."""
        module = DummyAdditionModule()
        controls = module.build_controls()
        for ctrl in controls:
            width = getattr(ctrl, "width", None)
            if width is not None:
                self.assertLessEqual(
                    width,
                    300,
                    f"Controle {ctrl} possui largura fixa de {width}px, violando a responsividade mobile.",
                )


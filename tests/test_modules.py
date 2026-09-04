"""
test_modules.py — Testes Unitários para os Módulos Didáticos de PDI.

Testa a inicialização, construção de controles, parametrização e processamento
dos módulos concretos da ementa:
- QuantizeModule
- GrayscaleModule
- BinaryOpsModule
"""

import unittest
from unittest.mock import MagicMock
import flet as ft
import numpy as np

from src.core.binary_ops import ResizeMode
from src.core.grayscale import GrayscaleMethod
from src.core.quantization import QuantizationTechnique
from src.ui.modules import (
    BasePDIModule,
    BinaryOpType,
    BinaryOpsModule,
    GrayscaleModule,
    QuantizeModule,
)


class TestQuantizeModule(unittest.TestCase):
    """Testes unitários para o QuantizeModule."""

    def setUp(self) -> None:
        self.mock_callback = MagicMock()
        self.module = QuantizeModule(on_param_changed=self.mock_callback)
        self.color_img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        self.gray_img = (np.random.rand(32, 32) * 255).astype(np.uint8)

    def test_metadata(self) -> None:
        """Verifica metadados e requisitos de entrada."""
        self.assertEqual(self.module.title, "Quantização & Dithering")
        self.assertFalse(self.module.requires_second_input)
        self.assertFalse(self.module.supports_scalar_mode)
        self.assertIn("bits", self.module.description.lower())

    def test_build_controls(self) -> None:
        """Verifica se os controles foram construídos e adicionados ao layout."""
        controls = self.module.build_controls()
        self.assertIsInstance(controls, list)
        self.assertGreaterEqual(len(controls), 3)
        self.assertIsInstance(self.module._dd_technique, ft.Dropdown)
        self.assertIsInstance(self.module._slider_bits, ft.Slider)
        self.assertIsInstance(self.module._switch_dither, ft.Switch)

    def test_get_params(self) -> None:
        """Verifica extração do dicionário de parâmetros."""
        params = self.module.get_params()
        self.assertIn("technique", params)
        self.assertIn("bits", params)
        self.assertEqual(params["technique"], QuantizationTechnique.UNIFORM)
        self.assertEqual(params["bits"], 4)

    def test_param_changes(self) -> None:
        """Verifica atualização de estado ao disparar eventos dos controles."""
        # Mudança de técnica
        mock_event = MagicMock()
        mock_event.control.value = QuantizationTechnique.KMEANS.name
        self.module._on_technique_changed(mock_event)
        self.assertEqual(self.module._technique, QuantizationTechnique.KMEANS)
        self.mock_callback.assert_called()

        # Mudança de bits
        mock_event.control.value = 6
        self.module._on_bits_changed(mock_event)
        self.assertEqual(self.module._bits, 6)

        # Mudança de dither
        mock_event.control.value = True
        self.module._on_dither_switched(mock_event)
        self.assertTrue(self.module._dither_active)
        self.assertEqual(self.module._technique, QuantizationTechnique.FLOYD_STEINBERG)

    def test_process_color(self) -> None:
        """Testa o processamento de imagem colorida RGB."""
        quantized, metrics = self.module.process(self.color_img, bits=3)
        self.assertEqual(quantized.shape, self.color_img.shape)
        self.assertEqual(quantized.dtype, np.uint8)
        self.assertIn("mse", metrics)
        self.assertIn("psnr", metrics)
        self.assertIn("unique_levels", metrics)
        self.assertIn("time_ms", metrics)
        self.assertGreater(metrics["time_ms"], 0.0)

    def test_process_grayscale(self) -> None:
        """Testa o processamento de imagem em escala de cinza 2D."""
        quantized, metrics = self.module.process(self.gray_img, bits=2)
        self.assertEqual(quantized.shape, self.gray_img.shape)
        self.assertLessEqual(metrics["unique_levels"], 4)

    def test_dropdown_bits_selection(self) -> None:
        """Testa o seletor Dropdown de bits e a atualização do badge."""
        self.module.build_controls()
        self.assertIsInstance(self.module._dd_bits, ft.Dropdown)
        self.assertEqual(len(self.module._dd_bits.options), 8)

        mock_event = MagicMock()
        mock_event.control.value = "5"
        self.module._on_dd_bits_changed(mock_event)
        self.assertEqual(self.module._bits, 5)
        self.assertIn("5 bits", self.module._txt_badge_bits.value)
        self.assertEqual(self.module._segmented_bits.selected, ["5"])

    def test_segmented_bits_and_badge(self) -> None:
        """Testa seletor de bits discreto e atualização de badge textual."""
        self.module.build_controls()
        mock_event = MagicMock()
        mock_event.control.selected = ["3"]
        self.module._on_bits_segmented_changed(mock_event)
        self.assertEqual(self.module._bits, 3)
        self.assertIn("3 bits", self.module._txt_badge_bits.value)
        self.assertEqual(self.module._dd_bits.value, "3")

    def test_input_mode_toggle(self) -> None:
        """Testa alternância de modo de entrada entre tons de cinza e RGB."""
        self.module.build_controls()
        mock_event = MagicMock()
        mock_event.control.selected = ["gray"]
        self.module._on_input_mode_changed(mock_event)
        self.assertEqual(self.module._input_mode, "gray")
        self.assertTrue(self.module._gray_method_container.visible)

        # Processamento com input_mode='gray' reduz imagem 3D a 2D
        quantized, _ = self.module.process(self.color_img, bits=4, input_mode="gray")
        self.assertEqual(quantized.ndim, 2)

    def test_compare_all_algorithms(self) -> None:
        """Testa a comparação quádrupla de algoritmos e seleção do vencedor."""
        from src.ui.modules.quantize_module import COMPARE_ALL_KEY
        quantized, metrics = self.module.process(self.color_img, bits=2, technique=COMPARE_ALL_KEY)
        self.assertIn("comparison_results", metrics)
        self.assertIn("best_algorithm", metrics)
        comp = metrics["comparison_results"]
        self.assertIn("UNIFORM", comp)
        self.assertIn("KMEANS", comp)
        self.assertIn("HISTOGRAM", comp)
        self.assertIn("FLOYD_STEINBERG", comp)
        best = metrics["best_algorithm"]
        best_psnr = comp[best]["psnr"]
        for k, v in comp.items():
            self.assertLessEqual(v["psnr"], best_psnr)

    def test_rgb_mode_preserves_3d_across_algorithms(self) -> None:
        """Garante que todos os 4 algoritmos preservam formato 3D uint8 em modo RGB."""
        algos = [
            QuantizationTechnique.UNIFORM,
            QuantizationTechnique.KMEANS,
            QuantizationTechnique.HISTOGRAM,
            QuantizationTechnique.FLOYD_STEINBERG,
        ]
        for algo in algos:
            res, _ = self.module.process(self.color_img, bits=2, technique=algo, input_mode="rgb")
            self.assertEqual(res.shape, self.color_img.shape)
            self.assertEqual(res.dtype, np.uint8)


class TestGrayscaleModule(unittest.TestCase):
    """Testes unitários para o GrayscaleModule."""

    def setUp(self) -> None:
        self.mock_callback = MagicMock()
        self.module = GrayscaleModule(on_param_changed=self.mock_callback)
        self.color_img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)

    def test_metadata(self) -> None:
        """Verifica metadados e requisitos de entrada."""
        self.assertEqual(self.module.title, "Conversão para Tons de Cinza & Canais")
        self.assertFalse(self.module.requires_second_input)
        self.assertFalse(self.module.supports_scalar_mode)

    def test_build_controls(self) -> None:
        """Verifica se os controles essenciais foram construídos."""
        controls = self.module.build_controls()
        self.assertIsInstance(controls, list)
        self.assertIsInstance(self.module._dd_method, ft.Dropdown)
        self.assertIsInstance(self.module._switch_isolate, ft.Switch)
        self.assertIsInstance(self.module._formula_card, ft.Container)

    def test_param_changes(self) -> None:
        """Testa alternância de métodos de escala de cinza."""
        mock_event = MagicMock()
        mock_event.control.value = GrayscaleMethod.CHANNEL_R.name
        self.module._on_method_changed(mock_event)
        self.assertEqual(self.module._method, GrayscaleMethod.CHANNEL_R)
        self.assertTrue(self.module._switch_isolate.visible)
        self.mock_callback.assert_called()

        mock_event.control.value = False
        self.module._on_switch_changed(mock_event)
        self.assertFalse(self.module._isolate_rgb)

    def test_process_luminance(self) -> None:
        """Testa conversão fisiológica ponderada ITU-R BT.601."""
        res, metrics = self.module.process(self.color_img, method=GrayscaleMethod.LUMINANCE)
        self.assertEqual(res.ndim, 2)
        self.assertEqual(res.shape, (32, 32))
        self.assertIn("mean_intensity", metrics)
        self.assertIn("std_dev", metrics)
        self.assertIn("time_ms", metrics)

    def test_process_channel_isolated_rgb(self) -> None:
        """Testa isolamento de canal preservando formato RGB (ex: [R,0,0])."""
        res, metrics = self.module.process(
            self.color_img,
            method=GrayscaleMethod.CHANNEL_R,
            isolate_rgb=True,
        )
        self.assertEqual(res.ndim, 3)
        self.assertEqual(res.shape, (32, 32, 3))
        # Canais G e B devem ser zero
        self.assertTrue(np.all(res[:, :, 1] == 0))
        self.assertTrue(np.all(res[:, :, 2] == 0))


class TestBinaryOpsModule(unittest.TestCase):
    """Testes unitários para o BinaryOpsModule."""

    def setUp(self) -> None:
        self.mock_callback = MagicMock()
        self.mock_scalar_cb = MagicMock()
        self.module = BinaryOpsModule(
            on_scalar_mode_changed=self.mock_scalar_cb,
            on_param_changed=self.mock_callback,
        )
        self.img_a = (np.ones((20, 20, 3)) * 100).astype(np.uint8)
        self.img_b = (np.ones((20, 20, 3)) * 40).astype(np.uint8)

    def test_metadata(self) -> None:
        """Verifica metadados e requisitos dinâmicos de entrada."""
        self.assertEqual(self.module.title, "Operações Aritméticas & Lógicas")
        self.assertTrue(self.module.requires_second_input)
        self.assertTrue(self.module.supports_scalar_mode)

    def test_build_controls(self) -> None:
        """Verifica a presença de todos os seletores e sliders."""
        controls = self.module.build_controls()
        self.assertIsInstance(controls, list)
        self.assertIsInstance(self.module._dd_op, ft.Dropdown)
        self.assertIsInstance(self.module._dd_resize, ft.Dropdown)
        self.assertIsInstance(self.module._slider_alpha, ft.Slider)
        self.assertIsInstance(self.module._switch_scalar, ft.Switch)

    def test_bitwise_not_toggles_requires_second_input(self) -> None:
        """Verifica se selecionar NOT desativa a exigência de segunda entrada."""
        mock_event = MagicMock()
        mock_event.control.value = BinaryOpType.BITWISE_NOT.name
        self.module._on_op_changed(mock_event)
        self.assertFalse(self.module.requires_second_input)

        mock_event.control.value = BinaryOpType.ADD.name
        self.module._on_op_changed(mock_event)
        self.assertTrue(self.module.requires_second_input)

    def test_scalar_switch_triggers_callback(self) -> None:
        """Verifica se alternar para modo escalar invoca on_scalar_mode_changed."""
        mock_event = MagicMock()
        mock_event.control.value = True
        self.module._on_scalar_switched(mock_event)
        self.assertTrue(self.module._is_scalar_input)
        self.mock_scalar_cb.assert_called_with(True)

    def test_process_add(self) -> None:
        """Testa adição entre imagens A e B."""
        res, metrics = self.module.process(self.img_a, img_b=self.img_b, op_type=BinaryOpType.ADD)
        self.assertEqual(res.shape, self.img_a.shape)
        self.assertTrue(np.all(res == 140))
        self.assertIn("mse", metrics)
        self.assertIn("psnr", metrics)

    def test_process_scalar(self) -> None:
        """Testa operação binária com constante escalar."""
        res, metrics = self.module.process(self.img_a, scalar_val=25.0, op_type=BinaryOpType.ADD)
        self.assertTrue(np.all(res == 125))

    def test_process_blend(self) -> None:
        """Testa mistura Alpha Blending com alpha=0.5."""
        res, metrics = self.module.process(
            self.img_a,
            img_b=self.img_b,
            op_type=BinaryOpType.BLEND,
            alpha=0.5,
        )
        # 0.5 * 100 + 0.5 * 40 = 70
        self.assertTrue(np.all(res == 70))


    def test_dropdowns_breathing_room_and_algorithm_metadata(self) -> None:
        """Garante que dropdowns possuem respiração vertical (sem dense=True) e retornam metadados de algoritmo."""
        q_mod = QuantizeModule()
        q_mod.build_controls()
        self.assertFalse(bool(q_mod._dd_bits.dense))
        self.assertFalse(bool(q_mod._dd_technique.dense))
        self.assertFalse(bool(q_mod._dd_gray_method.dense))
        self.assertIsNotNone(q_mod._dd_bits.content_padding)
        self.assertEqual(q_mod._dd_bits.border_radius, 8)

        # Verifica retorno de metadados de algoritmo
        img = np.full((16, 16), 100, dtype=np.uint8)
        _, metrics = q_mod.process(img, technique=QuantizationTechnique.UNIFORM, bits=4)
        self.assertIn("algorithm", metrics)
        self.assertIn("algorithm_badge", metrics)
        self.assertIn("Uniforme", metrics["algorithm"])


if __name__ == "__main__":
    unittest.main()


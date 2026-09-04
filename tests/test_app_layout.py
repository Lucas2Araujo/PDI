"""
test_app_layout.py — Testes Unitários para o Shell de Aplicação AppLayout.

Verifica:
- Instanciação de componentes (Slots A/B, Canvas, Fila, Telemetria, Módulos).
- Alternância dinâmica de módulos e controle reativo de visibilidade do Slot B.
- Alternância de modo (Individual vs Lote).
- Comportamento de redimensionamento responsivo (Mobile < 768px vs Desktop >= 768px).
- Fluxo completo de processamento individual e em lote.
- Exportação e auditoria de telemetria.
"""

import unittest
from unittest.mock import MagicMock
import flet as ft
import numpy as np

from src.ui.app_layout import AppLayout
from src.ui.components.batch_queue import BatchQueueItem
from src.ui.modules.binary_ops_module import BinaryOpType
from src.ui.state.session_state import SessionState


class TestAppLayout(unittest.TestCase):
    """Suíte de testes para a classe AppLayout."""

    def setUp(self) -> None:
        self.session_state = SessionState()
        self.layout = AppLayout(session_state=self.session_state)
        self.sample_img_a = (np.ones((40, 40, 3)) * 120).astype(np.uint8)
        self.sample_img_b = (np.ones((40, 40, 3)) * 30).astype(np.uint8)

    def test_initialization(self) -> None:
        """Verifica a integridade dos componentes instanciados pelo layout."""
        self.assertIsNotNone(self.layout.slot_a)
        self.assertIsNotNone(self.layout.slot_b)
        self.assertIsNotNone(self.layout.canvas)
        self.assertIsNotNone(self.layout.batch_queue)
        self.assertIsNotNone(self.layout.telemetry)
        self.assertEqual(len(self.layout.modules), 2)

        # Módulo padrão é QuantizeModule (índice 0)
        self.assertEqual(self.layout.active_module_index, 0)
        self.assertFalse(self.layout.slot_b.visible)
        self.assertFalse(self.layout.is_batch_mode)
        self.assertTrue(self.layout.canvas.visible)

    def test_select_module_toggles_slot_b(self) -> None:
        """Valida que alternar módulos altera a visibilidade do Slot B conforme requisitos."""
        # 1. QuantizeModule (não requer segundo input)
        self.layout.select_module(0)
        self.assertEqual(self.layout.active_module_index, 0)
        self.assertFalse(self.layout.slot_b.visible)

        # 2. BinaryOpsModule (requer segundo input)
        self.layout.select_module(1)
        self.assertEqual(self.layout.active_module_index, 1)
        self.assertTrue(self.layout.slot_b.visible)

        # 3. Operação NOT no BinaryOpsModule deve ocultar Slot B
        mock_event = MagicMock()
        mock_event.control.value = BinaryOpType.BITWISE_NOT.name
        self.layout.binary_ops_module._on_op_changed(mock_event)
        self.assertFalse(self.layout.slot_b.visible)

        # Retorna para ADD
        mock_event.control.value = BinaryOpType.ADD.name
        self.layout.binary_ops_module._on_op_changed(mock_event)
        self.assertTrue(self.layout.slot_b.visible)

    def test_scalar_mode_integration(self) -> None:
        """Valida que o switch escalar no BinaryOpsModule sincroniza com o Slot B."""
        self.layout.select_module(1)
        mock_event = MagicMock()
        mock_event.control.value = True
        self.layout.binary_ops_module._on_scalar_switched(mock_event)
        self.assertTrue(self.layout.slot_b.is_scalar_mode)

    def test_mode_toggling(self) -> None:
        """Valida alternância entre modo individual e modo em lote."""
        # Muda para lote
        self.layout.set_mode("batch")
        self.assertTrue(self.layout.is_batch_mode)
        self.assertFalse(self.layout.canvas.visible)
        self.assertTrue(self.layout.batch_queue.visible)

        # Retorna para individual
        self.layout.set_mode("single")
        self.assertFalse(self.layout.is_batch_mode)
        self.assertTrue(self.layout.canvas.visible)

    def test_handle_resize(self) -> None:
        """Testa o comportamento de redimensionamento para resoluções mobile e desktop."""
        # Mobile (< 768px)
        self.layout.handle_resize(480, 800)
        self.assertEqual(self.layout._current_width, 480)
        self.assertIsInstance(self.layout._main_body_layout.content, ft.Column)

        # Desktop (>= 768px)
        self.layout.handle_resize(1200, 900)
        self.assertEqual(self.layout._current_width, 1200)
        self.assertIsInstance(self.layout._main_body_layout.content, ft.Row)

    def test_desktop_layout_strict_two_columns(self) -> None:
        """Garante que o layout desktop possui estritamente 2 colunas: sidebar (360px) e workspace."""
        self.layout.handle_resize(1200, 900)
        row = self.layout._main_body_layout.content
        self.assertIsInstance(row, ft.Row)
        self.assertEqual(len(row.controls), 2)
        self.assertIs(row.controls[0], self.layout._sidebar_container)
        self.assertIs(row.controls[1], self.layout._workspace_area)
        self.assertEqual(self.layout._sidebar_container.width, 360)

    def test_run_processing_single_without_image(self) -> None:
        """Verifica que executar sem imagem de entrada avisa o usuário sem falhar."""
        self.layout.run_processing()
        self.assertIsNone(self.session_state.result_image)

    def test_run_processing_single_quantize(self) -> None:
        """Testa o processamento individual com o módulo de quantização."""
        self.layout.slot_a.set_image(self.sample_img_a, "foto_a.png")
        self.layout.select_module(0)
        self.layout.run_processing()

        self.assertIsNotNone(self.session_state.result_image)
        self.assertEqual(self.session_state.result_image.shape, self.sample_img_a.shape)
        self.assertIn("mse", self.session_state.metrics)

    def test_run_processing_single_grayscale(self) -> None:
        """Testa o processamento individual com o modo tons de cinza do módulo de quantização."""
        self.layout.slot_a.set_image(self.sample_img_a, "foto_a.png")
        self.layout.select_module(0)
        self.layout.quantize_module._input_mode = "gray"
        self.layout.run_processing()

        self.assertIsNotNone(self.session_state.result_image)
        self.assertEqual(self.session_state.result_image.ndim, 2)

    def test_run_processing_single_binary_ops(self) -> None:
        """Testa o processamento individual com o módulo de operações binárias."""
        self.layout.slot_a.set_image(self.sample_img_a, "foto_a.png")
        self.layout.slot_b.set_image(self.sample_img_b, "foto_b.png")
        self.layout.select_module(1)
        self.layout.run_processing()

        self.assertIsNotNone(self.session_state.result_image)
        # 120 + 30 = 150
        self.assertTrue(np.all(self.session_state.result_image == 150))

    def test_run_processing_single_binary_scalar(self) -> None:
        """Testa o processamento binário com constante escalar configurada no Slot B."""
        self.layout.slot_a.set_image(self.sample_img_a, "foto_a.png")
        self.layout.select_module(1)
        self.layout.slot_b.set_scalar_mode(True, default_val=10.0)
        self.layout.run_processing()

        self.assertIsNotNone(self.session_state.result_image)
        # 120 + 10 = 130
        self.assertTrue(np.all(self.session_state.result_image == 130))

    def test_run_processing_batch(self) -> None:
        """Testa o fluxo de execução sequencial no modo em lote."""
        item1 = BatchQueueItem(name="item1.png", array=(np.ones((20, 20, 3)) * 50).astype(np.uint8))
        item2 = BatchQueueItem(name="item2.png", array=(np.ones((20, 20, 3)) * 150).astype(np.uint8))

        self.layout.set_mode("batch")
        self.layout.batch_queue.set_items([item1, item2])
        self.layout.select_module(0)  # Quantize
        self.layout.run_processing()

        items = self.layout.batch_queue.items
        self.assertIn("Concluído", items[0].status)
        self.assertIn("Concluído", items[1].status)

    def test_run_processing_batch_empty(self) -> None:
        """Verifica que executar lote vazio não causa exceção."""
        self.layout.set_mode("batch")
        self.layout.batch_queue.clear()
        self.layout.run_processing()

    def test_export_result_without_image(self) -> None:
        """Verifica que exportar sem imagem gerada trata com segurança."""
        self.layout.export_result()

    def test_on_open_inspector_safety(self) -> None:
        """Verifica que invocar auditoria sem resultado não falha."""
        self.layout._on_open_inspector()


if __name__ == "__main__":
    unittest.main()


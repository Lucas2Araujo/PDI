"""
test_pipeline_history.py — Testes Unitários para o Pipeline de Composição e Histórico (Undo/Redo).

Cobre:
- Promoção de resultado para Entrada A (`promote_result_to_input_a`).
- Operações de desfazer (`undo`) e refazer (`redo`) com restauração de array e nome.
- Limite máximo de retenção (`MAX_HISTORY_STEPS = 5`) e coleta forçada com `gc.collect()`.
- Limpeza de histórico e descarte de futuro ao realizar nova ação após undo.
- Integração reativa com `AppLayout` e badge didático de etapa no `InputSlot`.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.ui.app_layout import AppLayout
from src.ui.components.input_slot import InputSlot
from src.ui.state.session_state import (
    EVENT_IMAGE_A_CHANGED,
    EVENT_RESULT_CHANGED,
    MAX_HISTORY_STEPS,
    SessionState,
)


class TestPipelineHistoryState(unittest.TestCase):
    """Testes unitários focados na lógica de estado e memória do SessionState."""

    def setUp(self) -> None:
        self.state = SessionState()
        self.img_init = np.full((10, 10), 50, dtype=np.uint8)
        self.img_res1 = np.full((10, 10), 100, dtype=np.uint8)
        self.img_res2 = np.full((10, 10), 150, dtype=np.uint8)

    def test_initial_pipeline_state(self) -> None:
        """Verifica que as pilhas de histórico e futuro iniciam vazias."""
        self.assertEqual(self.state._history, [])
        self.assertEqual(self.state._future, [])
        self.assertEqual(self.state.history_count, 0)
        self.assertFalse(self.state.can_undo())
        self.assertFalse(self.state.can_redo())

    def test_promote_without_result_returns_false(self) -> None:
        """promote_result_to_input_a deve retornar False se result_image for None."""
        self.state.set_image_a(self.img_init, "orig.png")
        self.assertIsNone(self.state.result_image)

        res = self.state.promote_result_to_input_a()
        self.assertFalse(res)
        self.assertIs(self.state.image_a, self.img_init)
        self.assertEqual(self.state.history_count, 0)

    def test_promote_result_to_input_a_success(self) -> None:
        """Valida promoção bem-sucedida do resultado para Entrada A e emissão de eventos."""
        mock_a = MagicMock()
        mock_res = MagicMock()
        self.state.subscribe(EVENT_IMAGE_A_CHANGED, mock_a)
        self.state.subscribe(EVENT_RESULT_CHANGED, mock_res)

        self.state.set_image_a(self.img_init, "lena.png")
        self.state.set_result(self.img_res1, metrics={"mse": 4.5}, module_name="Quantização")

        # Promove
        success = self.state.promote_result_to_input_a()
        self.assertTrue(success)

        # Imagem A atualizada com cópia independente
        self.assertTrue(np.array_equal(self.state.image_a, self.img_res1))
        self.assertIsNot(self.state.image_a, self.img_res1)  # Cópia
        self.assertEqual(self.state.image_a_name, "lena.png + Quantização")

        # Resultado limpo
        self.assertIsNone(self.state.result_image)
        self.assertEqual(self.state.result_name, "")
        self.assertEqual(self.state.metrics, {})

        # Histórico contém o estado anterior
        self.assertEqual(self.state.history_count, 1)
        self.assertTrue(self.state.can_undo())
        self.assertFalse(self.state.can_redo())
        hist_img, hist_name = self.state._history[0]
        self.assertTrue(np.array_equal(hist_img, self.img_init))
        self.assertEqual(hist_name, "lena.png")

        # Notificações emitidas
        mock_a.assert_called_with(image=self.state.image_a, name=self.state.image_a_name)
        mock_res.assert_called_with(image=None, metrics={})

    @patch("gc.collect")
    def test_undo_restores_previous_state(self, mock_gc: MagicMock) -> None:
        """undo() deve restaurar o estado anterior de Entrada A e acionar gc.collect()."""
        self.state.set_image_a(self.img_init, "orig.png")
        self.state.set_result(self.img_res1, module_name="Grayscale")
        self.state.promote_result_to_input_a()

        mock_cb = MagicMock()
        self.state.subscribe(EVENT_IMAGE_A_CHANGED, mock_cb)

        # Executa Desfazer
        res = self.state.undo()
        self.assertTrue(res)
        self.assertTrue(np.array_equal(self.state.image_a, self.img_init))
        self.assertEqual(self.state.image_a_name, "orig.png")

        self.assertFalse(self.state.can_undo())
        self.assertTrue(self.state.can_redo())
        mock_gc.assert_called()
        mock_cb.assert_called_with(image=self.state.image_a, name="orig.png")

    def test_undo_on_empty_history_returns_false(self) -> None:
        """undo() sem histórico deve retornar False sem lançar erro."""
        self.assertFalse(self.state.can_undo())
        self.assertFalse(self.state.undo())

    def test_redo_restores_future_state(self) -> None:
        """redo() deve restaurar o estado futuro após um undo."""
        self.state.set_image_a(self.img_init, "orig.png")
        self.state.set_result(self.img_res1, module_name="Grayscale")
        self.state.promote_result_to_input_a()

        # Desfaz e em seguida Refaz
        self.state.undo()
        self.assertTrue(self.state.can_redo())

        mock_cb = MagicMock()
        self.state.subscribe(EVENT_IMAGE_A_CHANGED, mock_cb)

        res = self.state.redo()
        self.assertTrue(res)
        self.assertTrue(np.array_equal(self.state.image_a, self.img_res1))
        self.assertEqual(self.state.image_a_name, "orig.png + Grayscale")
        self.assertTrue(self.state.can_undo())
        self.assertFalse(self.state.can_redo())
        mock_cb.assert_called_with(image=self.state.image_a, name="orig.png + Grayscale")

    def test_redo_on_empty_future_returns_false(self) -> None:
        """redo() sem futuro deve retornar False."""
        self.assertFalse(self.state.can_redo())
        self.assertFalse(self.state.redo())

    def test_multi_step_undo_redo_pipeline(self) -> None:
        """Testa encadeamento sequencial de 3 etapas com navegação completa de undo e redo."""
        img0 = np.full((5, 5), 10, dtype=np.uint8)
        img1 = np.full((5, 5), 20, dtype=np.uint8)
        img2 = np.full((5, 5), 30, dtype=np.uint8)
        img3 = np.full((5, 5), 40, dtype=np.uint8)

        self.state.set_image_a(img0, "base.png")

        # Etapa 1: Grayscale
        self.state.set_result(img1, module_name="Grayscale")
        self.state.promote_result_to_input_a()
        self.assertEqual(self.state.image_a_name, "base.png + Grayscale")

        # Etapa 2: Quantize
        self.state.set_result(img2, module_name="Quantização")
        self.state.promote_result_to_input_a()
        self.assertEqual(self.state.image_a_name, "base.png + Grayscale + Quantização")

        # Etapa 3: Binary
        self.state.set_result(img3, module_name="Binário")
        self.state.promote_result_to_input_a()
        self.assertEqual(self.state.image_a_name, "base.png + Grayscale + Quantização + Binário")
        self.assertEqual(self.state.history_count, 3)

        # Desfaz Etapa 3 -> volta para Etapa 2
        self.state.undo()
        self.assertTrue(np.array_equal(self.state.image_a, img2))
        self.assertEqual(self.state.image_a_name, "base.png + Grayscale + Quantização")

        # Desfaz Etapa 2 -> volta para Etapa 1
        self.state.undo()
        self.assertTrue(np.array_equal(self.state.image_a, img1))
        self.assertEqual(self.state.image_a_name, "base.png + Grayscale")

        # Desfaz Etapa 1 -> volta para base
        self.state.undo()
        self.assertTrue(np.array_equal(self.state.image_a, img0))
        self.assertEqual(self.state.image_a_name, "base.png")
        self.assertFalse(self.state.can_undo())

        # Refaz para Etapa 1
        self.state.redo()
        self.assertTrue(np.array_equal(self.state.image_a, img1))

        # Refaz para Etapa 2
        self.state.redo()
        self.assertTrue(np.array_equal(self.state.image_a, img2))

        # Refaz para Etapa 3
        self.state.redo()
        self.assertTrue(np.array_equal(self.state.image_a, img3))
        self.assertFalse(self.state.can_redo())

    def test_new_action_after_undo_clears_future(self) -> None:
        """Aplicar uma nova promoção após um undo deve descartar a pilha de refazer."""
        self.state.set_image_a(self.img_init, "orig.png")
        self.state.set_result(self.img_res1, module_name="M1")
        self.state.promote_result_to_input_a()

        self.state.undo()
        self.assertTrue(self.state.can_redo())

        # Nova ação enquanto estava desfeito
        new_res = np.full((10, 10), 222, dtype=np.uint8)
        self.state.set_result(new_res, module_name="NovoRamo")
        self.state.promote_result_to_input_a()

        # Futuro deve ter sido descartado
        self.assertFalse(self.state.can_redo())
        self.assertEqual(len(self.state._future), 0)
        self.assertEqual(self.state.image_a_name, "orig.png + NovoRamo")

    @patch("gc.collect")
    def test_max_history_steps_limit_and_gc(self, mock_gc: MagicMock) -> None:
        """Garante que _history não ultrapassa MAX_HISTORY_STEPS (5) e descarta o mais antigo chamando gc.collect()."""
        self.state.set_image_a(self.img_init, "base.png")

        # Executa 7 promoções consecutivas (limite é 5)
        for i in range(1, 8):
            step_img = np.full((10, 10), i * 20, dtype=np.uint8)
            self.state.set_result(step_img, module_name=f"Etapa{i}")
            self.state.promote_result_to_input_a()

        # Tamanho do histórico não pode ultrapassar 5
        self.assertLessEqual(len(self.state._history), MAX_HISTORY_STEPS)
        self.assertEqual(len(self.state._history), 5)

        # gc.collect deve ter sido chamado pelo menos 2 vezes para descartar os itens excedentes
        self.assertGreaterEqual(mock_gc.call_count, 2)

    def test_clear_all_resets_history(self) -> None:
        """clear_all() deve esvaziar histórico, futuro e resetar módulo anterior."""
        self.state.set_image_a(self.img_init, "a.png")
        self.state.set_result(self.img_res1, module_name="Módulo")
        self.state.promote_result_to_input_a()

        self.assertEqual(self.state.history_count, 1)
        self.state.clear_all()

        self.assertEqual(self.state.history_count, 0)
        self.assertFalse(self.state.can_undo())
        self.assertFalse(self.state.can_redo())
        self.assertEqual(self.state.last_applied_module_name, "")


class TestPipelineLayoutIntegration(unittest.TestCase):
    """Testes de integração entre AppLayout, InputSlot e controles de pipeline."""

    def setUp(self) -> None:
        self.session_state = SessionState()
        self.layout = AppLayout(session_state=self.session_state)
        self.img = np.full((30, 30, 3), 100, dtype=np.uint8)

    def test_initial_button_states(self) -> None:
        """Botões de pipeline devem iniciar desabilitados."""
        self.assertIsNotNone(self.layout._btn_promote)
        self.assertIsNotNone(self.layout._btn_undo)
        self.assertIsNotNone(self.layout._btn_redo)

        self.assertTrue(self.layout._btn_promote.disabled)
        self.assertTrue(self.layout._btn_undo.disabled)
        self.assertTrue(self.layout._btn_redo.disabled)

    def test_promote_enables_and_disables_correctly(self) -> None:
        """Executar processamento habilita botão de promoção; promover habilita desfazer."""
        self.layout.slot_a.set_image(self.img, "foto.png")
        self.layout.select_module(0)  # Grayscale
        self.layout.run_processing()

        # Após resultado, botão de promoção deve estar habilitado
        self.assertFalse(self.layout._btn_promote.disabled)
        self.assertTrue(self.layout._btn_undo.disabled)

        # Clica em Promover
        success = self.layout.promote_result()
        self.assertTrue(success)

        # Botão de promover agora desabilitado (resultado consumido), undo habilitado
        self.assertTrue(self.layout._btn_promote.disabled)
        self.assertFalse(self.layout._btn_undo.disabled)
        self.assertTrue(self.layout._btn_redo.disabled)

        # Slot A agora tem o resultado e o nome atualizado
        self.assertIsNotNone(self.layout.slot_a.image_array)
        self.assertIn("Quantização & Dithering", self.layout.slot_a.image_name)

    def test_slot_a_shows_composition_badge(self) -> None:
        """Slot A deve exibir badge 'Etapa 2' após a primeira promoção e ocultar no undo."""
        self.layout.slot_a.set_image(self.img, "foto.png")
        self.layout.select_module(0)
        self.layout.run_processing()
        self.layout.promote_result()

        # Verifica conteúdo do cabeçalho do Slot A
        header_row = self.layout.slot_a._content_column.controls[0]
        # Deve ter 2 controles: o título do slot e o badge de etapa
        self.assertEqual(len(header_row.controls), 2)
        badge = header_row.controls[1]
        self.assertEqual(badge.tooltip, "Pipeline Composto — Etapa 2")

        # Ao desfazer, badge deve ser removido
        self.layout.undo()
        header_row_after_undo = self.layout.slot_a._content_column.controls[0]
        self.assertEqual(len(header_row_after_undo.controls), 1)

    def test_batch_mode_hides_pipeline_buttons(self) -> None:
        """Ao alternar para modo em lote, os botões de pipeline do cabeçalho devem ficar invisíveis."""
        self.layout.set_mode("batch")
        self.assertFalse(self.layout._btn_promote.visible)
        self.assertFalse(self.layout._btn_undo.visible)
        self.assertFalse(self.layout._btn_redo.visible)

        self.layout.set_mode("single")
        self.assertTrue(self.layout._btn_promote.visible)
        self.assertTrue(self.layout._btn_undo.visible)
        self.assertTrue(self.layout._btn_redo.visible)


if __name__ == "__main__":
    unittest.main()


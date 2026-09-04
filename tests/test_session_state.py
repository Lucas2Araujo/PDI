"""
test_session_state.py — Testes unitários para o gerenciador de estado reativo (src.ui.state.session_state).
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.ui.state.session_state import (
    EVENT_IMAGE_A_CHANGED,
    EVENT_IMAGE_B_CHANGED,
    EVENT_MODE_TOGGLED,
    EVENT_PROCESSING_STATE,
    EVENT_RESULT_CHANGED,
    SessionState,
    get_session_state,
    reset_session_state,
)


class TestSessionState(unittest.TestCase):
    """Suíte de testes para a classe SessionState e barramento de eventos."""

    def setUp(self) -> None:
        self.state = SessionState()
        self.sample_img = np.zeros((10, 10), dtype=np.uint8)

    def test_initial_state(self) -> None:
        """Verifica que o estado inicial possui campos zerados e corretos."""
        self.assertIsNone(self.state.image_a)
        self.assertIsNone(self.state.image_b)
        self.assertIsNone(self.state.result_image)
        self.assertEqual(self.state.image_a_name, "")
        self.assertEqual(self.state.image_b_name, "")
        self.assertEqual(self.state.metrics, {})
        self.assertFalse(self.state.is_processing)
        self.assertFalse(self.state.is_batch_mode)
        self.assertEqual(self.state.batch_queue, [])
        self.assertIsNone(self.state.batch_results)

    def test_subscribe_and_notify(self) -> None:
        """Valida entrega de eventos a observadores inscritos."""
        mock_cb = MagicMock()
        self.state.subscribe(EVENT_IMAGE_A_CHANGED, mock_cb)

        self.state.notify(EVENT_IMAGE_A_CHANGED, image=self.sample_img, name="foto.png")
        mock_cb.assert_called_once_with(image=self.sample_img, name="foto.png")

    def test_unsubscribe(self) -> None:
        """Valida que observador desinscrito não recebe mais notificações."""
        mock_cb = MagicMock()
        self.state.subscribe(EVENT_RESULT_CHANGED, mock_cb)
        self.state.unsubscribe(EVENT_RESULT_CHANGED, mock_cb)

        self.state.notify(EVENT_RESULT_CHANGED, image=self.sample_img, metrics={})
        mock_cb.assert_not_called()

    def test_unsubscribe_callable_return(self) -> None:
        """Valida padrão de retorno de função de desinscrição."""
        mock_cb = MagicMock()
        unsub = self.state.subscribe(EVENT_PROCESSING_STATE, mock_cb)

        unsub()
        self.state.notify(EVENT_PROCESSING_STATE, is_processing=True)
        mock_cb.assert_not_called()

    def test_parameterless_callback(self) -> None:
        """Observadores sem parâmetros devem ser invocados graciosamente."""
        called = False

        def on_change():
            nonlocal called
            called = True

        self.state.subscribe(EVENT_IMAGE_A_CHANGED, on_change)
        self.state.notify(EVENT_IMAGE_A_CHANGED, image=self.sample_img, name="teste.png")
        self.assertTrue(called)

    def test_subscriber_exception_does_not_break_chain(self) -> None:
        """Erro em um callback não deve impedir que os demais observadores sejam chamados."""
        failing_cb = MagicMock(side_effect=RuntimeError("Erro interno no componente"))
        success_cb = MagicMock()

        self.state.subscribe(EVENT_IMAGE_A_CHANGED, failing_cb)
        self.state.subscribe(EVENT_IMAGE_A_CHANGED, success_cb)

        self.state.notify(EVENT_IMAGE_A_CHANGED, image=self.sample_img)
        failing_cb.assert_called_once()
        success_cb.assert_called_once()

    def test_set_image_a_triggers_event(self) -> None:
        """set_image_a deve atualizar o estado e disparar EVENT_IMAGE_A_CHANGED."""
        mock_cb = MagicMock()
        self.state.subscribe(EVENT_IMAGE_A_CHANGED, mock_cb)

        self.state.set_image_a(self.sample_img, "lena.png")
        self.assertIs(self.state.image_a, self.sample_img)
        self.assertEqual(self.state.image_a_name, "lena.png")
        mock_cb.assert_called_once_with(image=self.sample_img, name="lena.png")

    def test_set_image_b_triggers_event(self) -> None:
        """set_image_b deve atualizar o estado e disparar EVENT_IMAGE_B_CHANGED."""
        mock_cb = MagicMock()
        self.state.subscribe(EVENT_IMAGE_B_CHANGED, mock_cb)

        self.state.set_image_b(self.sample_img, "moon.png")
        self.assertIs(self.state.image_b, self.sample_img)
        self.assertEqual(self.state.image_b_name, "moon.png")
        mock_cb.assert_called_once_with(image=self.sample_img, name="moon.png")

    def test_set_result_triggers_event(self) -> None:
        """set_result deve atualizar resultado, métricas e disparar evento."""
        mock_cb = MagicMock()
        self.state.subscribe(EVENT_RESULT_CHANGED, mock_cb)

        metrics = {"psnr": 42.5, "time_ms": 12.3}
        self.state.set_result(self.sample_img, metrics)
        self.assertIs(self.state.result_image, self.sample_img)
        self.assertEqual(self.state.metrics["psnr"], 42.5)
        mock_cb.assert_called_once_with(image=self.sample_img, metrics=metrics)

    def test_set_processing_and_batch_mode(self) -> None:
        """Testa mutações de flag e seus respectivos eventos."""
        proc_cb = MagicMock()
        mode_cb = MagicMock()
        self.state.subscribe(EVENT_PROCESSING_STATE, proc_cb)
        self.state.subscribe(EVENT_MODE_TOGGLED, mode_cb)

        self.state.set_processing(True)
        self.assertTrue(self.state.is_processing)
        proc_cb.assert_called_once_with(is_processing=True)

        new_mode = self.state.toggle_batch_mode()
        self.assertTrue(new_mode)
        self.assertTrue(self.state.is_batch_mode)
        mode_cb.assert_called_once_with(is_batch_mode=True)

    @patch("gc.collect")
    def test_clear_result_invokes_gc(self, mock_gc: MagicMock) -> None:
        """clear_result deve anular resultado e forçar gc.collect()."""
        self.state.result_image = self.sample_img
        self.state.metrics = {"mse": 10.0}

        mock_cb = MagicMock()
        self.state.subscribe(EVENT_RESULT_CHANGED, mock_cb)

        self.state.clear_result()
        self.assertIsNone(self.state.result_image)
        self.assertEqual(self.state.metrics, {})
        mock_gc.assert_called_once()
        mock_cb.assert_called_once_with(image=None, metrics={})

    @patch("gc.collect")
    def test_clear_all_cleans_memory_and_notifies(self, mock_gc: MagicMock) -> None:
        """clear_all deve resetar todos os estados e invocar gc.collect()."""
        self.state.image_a = self.sample_img
        self.state.image_b = self.sample_img
        self.state.result_image = self.sample_img
        self.state.image_a_name = "a.png"
        self.state.is_processing = True

        self.state.clear_all()
        self.assertIsNone(self.state.image_a)
        self.assertIsNone(self.state.image_b)
        self.assertIsNone(self.state.result_image)
        self.assertEqual(self.state.image_a_name, "")
        self.assertFalse(self.state.is_processing)
        mock_gc.assert_called_once()


class TestGlobalSessionState(unittest.TestCase):
    """Suíte de testes para funções de conveniência do singleton global."""

    def setUp(self) -> None:
        reset_session_state()

    def test_singleton_identity(self) -> None:
        """get_session_state deve retornar a mesma instância em chamadas sucessivas."""
        s1 = get_session_state()
        s2 = get_session_state()
        self.assertIs(s1, s2)

    def test_reset_session_state(self) -> None:
        """reset_session_state deve criar uma nova instância limpa."""
        s1 = get_session_state()
        s1.set_image_a(np.ones((5, 5), dtype=np.uint8), "foto.png")

        s2 = reset_session_state()
        self.assertIsNot(s1, s2)
        self.assertIsNone(s2.image_a)


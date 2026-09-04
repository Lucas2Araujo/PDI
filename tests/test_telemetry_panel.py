"""
test_telemetry_panel.py — Testes unitários para o componente TelemetryPanel (src.ui.components.telemetry_panel).
"""

import unittest
from unittest.mock import MagicMock
import flet as ft
import numpy as np

from src.ui.components.telemetry_panel import TelemetryPanel
from src.ui.state.session_state import SessionState


class TestTelemetryPanel(unittest.TestCase):
    """Suíte de testes para o componente TelemetryPanel."""

    def setUp(self) -> None:
        self.state = SessionState()
        self.sample_img = np.full((32, 32), 120, dtype=np.uint8)

    def test_initial_empty_state(self) -> None:
        """Verifica que o painel inicializa sem dados de telemetria."""
        panel = TelemetryPanel(session_state=self.state)
        self.assertEqual(panel._metrics, {})
        self.assertIsNone(panel._result_image)
        self.assertFalse(panel._btn_inspector.visible)

    def test_update_metrics_creates_badges(self) -> None:
        """update_metrics deve preencher os badges para MSE, PSNR, Níveis e Tempo."""
        panel = TelemetryPanel(session_state=self.state)
        metrics = {
            "mse": 12.34,
            "psnr": 37.25,
            "unique_levels": 8,
            "time_ms": 15.6,
        }
        panel.update_metrics(metrics)

        self.assertEqual(panel._metrics, metrics)
        # Deve possuir ao menos 4 badges renderizados
        self.assertGreaterEqual(len(panel._badges_row.controls), 4)

    def test_reactive_result_changed(self) -> None:
        """SessionState.set_result com métricas deve atualizar automaticamente o TelemetryPanel."""
        panel = TelemetryPanel(session_state=self.state)
        metrics = {"mse": 5.0, "psnr": 42.0}

        self.state.set_result(self.sample_img, metrics=metrics)
        self.assertIs(panel._result_image, self.sample_img)
        self.assertEqual(panel._metrics["psnr"], 42.0)
        self.assertTrue(panel._histogram_chart.visible)

    def test_set_pipeline_context_enables_inspector(self) -> None:
        """set_pipeline_context com dados válidos deve tornar visível o botão das Entranhas."""
        panel = TelemetryPanel(session_state=self.state)
        raw = np.full((32, 32, 3), 100, dtype=np.uint8)
        gray = np.full((32, 32), 100, dtype=np.uint8)

        panel.set_pipeline_context(raw_image=raw, gray_image=gray)
        self.assertTrue(panel._btn_inspector.visible)

    def test_inspector_callback_trigger(self) -> None:
        """_trigger_inspector deve invocar o callback on_open_inspector se configurado."""
        mock_inspector = MagicMock()
        panel = TelemetryPanel(session_state=self.state, on_open_inspector=mock_inspector)

        panel._trigger_inspector()
        mock_inspector.assert_called_once()


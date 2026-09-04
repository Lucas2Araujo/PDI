"""
test_image_canvas.py — Testes unitários para o componente ImageCanvas (src.ui.components.image_canvas).
"""

import unittest
from unittest.mock import MagicMock
import flet as ft
import numpy as np

from src.ui.components.image_canvas import DisplayMode, ImageCanvas
from src.ui.state.session_state import SessionState


class TestImageCanvas(unittest.TestCase):
    """Suíte de testes para o componente ImageCanvas."""

    def setUp(self) -> None:
        self.state = SessionState()
        self.sample_img = np.full((64, 64), 180, dtype=np.uint8)

    def test_initialization_and_default_mode(self) -> None:
        """Verifica inicialização padrão do ImageCanvas."""
        canvas = ImageCanvas(session_state=self.state)
        self.assertEqual(canvas.display_mode, DisplayMode.RESULT_ONLY)
        self.assertIsNone(canvas._result_image)

    def test_mode_switching(self) -> None:
        """Valida alternância entre os modos RESULT_ONLY, SIDE_BY_SIDE e TRIPLE."""
        canvas = ImageCanvas(session_state=self.state)

        canvas.set_display_mode(DisplayMode.SIDE_BY_SIDE)
        self.assertEqual(canvas.display_mode, DisplayMode.SIDE_BY_SIDE)

        canvas.set_display_mode(DisplayMode.TRIPLE)
        self.assertEqual(canvas.display_mode, DisplayMode.TRIPLE)

        canvas.set_display_mode(DisplayMode.RESULT_ONLY)
        self.assertEqual(canvas.display_mode, DisplayMode.RESULT_ONLY)

    def test_set_result_and_intermediate(self) -> None:
        """set_result e set_intermediate devem atualizar os arrays internos."""
        canvas = ImageCanvas(session_state=self.state)
        canvas.set_result(self.sample_img, "foto_result.png")
        self.assertIs(canvas._result_image, self.sample_img)
        self.assertEqual(canvas._result_name, "foto_result.png")

        mid_img = np.full((64, 64), 90, dtype=np.uint8)
        canvas.set_intermediate(mid_img, "cinza.png")
        self.assertIs(canvas._intermediate_image, mid_img)

    def test_reactive_result_changed(self) -> None:
        """SessionState.set_result deve atualizar automaticamente o ImageCanvas."""
        canvas = ImageCanvas(session_state=self.state)
        res_img = np.full((32, 32), 220, dtype=np.uint8)

        self.state.set_result(res_img, metrics={"psnr": 38.2})
        self.assertIs(canvas._result_image, res_img)

    def test_download_callback_triggered(self) -> None:
        """_trigger_download deve acionar o callback on_download com os bytes PNG e o nome."""
        mock_download = MagicMock()
        canvas = ImageCanvas(session_state=self.state, on_download=mock_download)
        canvas.set_result(self.sample_img, "output.png")

        canvas._trigger_download()
        mock_download.assert_called_once()
        args = mock_download.call_args[0]
        self.assertIsInstance(args[0], bytes)
        self.assertEqual(args[1], "output.png")

    def test_no_hardcoded_rigid_dimensions(self) -> None:
        """Garante a ausência de alturas fixas como height=520 na área de trabalho."""
        canvas = ImageCanvas(session_state=self.state)
        canvas.set_result(self.sample_img)

        self.assertNotEqual(getattr(canvas, "height", None), 520)
        self.assertNotEqual(getattr(canvas._display_area, "height", None), 520)

    def test_analytics_mode(self) -> None:
        """Verifica que o modo ANALYTICS constrói figura analítica consolidada em alta resolução como arquivo único."""
        mock_download = MagicMock()
        canvas = ImageCanvas(session_state=self.state, on_download=mock_download)
        img_a = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        res_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        self.state.set_image_a(img_a, "input.png")
        self.state.set_result(res_img, metrics={"psnr": 32.5, "mse": 12.0, "bits": 4, "technique": "Uniforme"})

        canvas.set_display_mode(DisplayMode.ANALYTICS)
        self.assertEqual(canvas.display_mode, DisplayMode.ANALYTICS)
        self.assertIsNotNone(canvas._display_area.content)
        self.assertIsNotNone(canvas._figure_bytes)
        self.assertTrue(canvas._figure_bytes.startswith(b"\x89PNG"))
        self.assertTrue(canvas._img_analytics.src.startswith("data:image/png;base64,"))

        # Download no modo analítico baixa o arquivo único da figura consolidada
        canvas._trigger_download()
        mock_download.assert_called_once()
        d_bytes, d_name = mock_download.call_args[0]
        self.assertEqual(d_bytes, canvas._figure_bytes)
        self.assertIn("painel_analitico_histogramas", d_name)

    def test_comparison_mode(self) -> None:
        """Verifica que o modo COMPARISON renderiza grid 2x2 com métricas e vencedor."""
        canvas = ImageCanvas(session_state=self.state)
        res_img = np.full((32, 32), 128, dtype=np.uint8)

        mock_comparison = {
            "UNIFORM": {
                "title": "Uniforme",
                "image": np.full((32, 32), 120, dtype=np.uint8),
                "psnr": 30.0,
                "mse": 10.0,
                "time_ms": 1.2,
            },
            "KMEANS": {
                "title": "K-Means",
                "image": np.full((32, 32), 125, dtype=np.uint8),
                "psnr": 35.5,
                "mse": 5.0,
                "time_ms": 8.4,
            },
        }

        self.state.set_result(
            res_img,
            metrics={"comparison_results": mock_comparison, "best_algorithm": "KMEANS", "psnr": 35.5},
        )

        canvas.set_display_mode(DisplayMode.COMPARISON)
        self.assertEqual(canvas.display_mode, DisplayMode.COMPARISON)
        self.assertIsNotNone(canvas._display_area.content)
        self.assertIsNotNone(canvas._comparison_winner)

    def test_triple_mode_auto_intermediate(self) -> None:
        """Verifica derivação automática de imagem intermediária e Data URI no modo TRIPLE."""
        canvas = ImageCanvas(session_state=self.state)
        img_a = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        res_img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        self.state.set_image_a(img_a, "original_color.png")
        self.state.set_result(res_img, metrics={"psnr": 30.0})

        canvas.set_display_mode(DisplayMode.TRIPLE)
        self.assertIsNotNone(canvas._intermediate_image)
        self.assertEqual(canvas._intermediate_image.ndim, 2)
        self.assertTrue(canvas._img_inter.src.startswith("data:image/png;base64,"))

    def test_algorithm_name_displayed_on_processed_image(self) -> None:
        """Garante que a imagem processada exibe o nome e badge do algoritmo e não apenas 'imagem processada'."""
        canvas = ImageCanvas(session_state=self.state)
        res_img = np.full((32, 32), 150, dtype=np.uint8)

        self.state.set_result(
            res_img,
            metrics={
                "technique": "Floyd-Steinberg (Dithering)",
                "bits": 3,
                "algorithm": "Floyd-Steinberg (Dithering) (3 bits)",
                "algorithm_badge": "Floyd-Steinberg (3b)",
            },
        )

        algo_full, algo_badge = canvas._get_active_algorithm_info()
        self.assertIn("Floyd-Steinberg", algo_full)
        self.assertIn("3b", algo_badge)

        # No modo RESULT_ONLY, o display_area content deve conter o container com o badge do algoritmo
        canvas.set_display_mode(DisplayMode.RESULT_ONLY)
        self.assertIsInstance(canvas._display_area.content, ft.Container)
        col = canvas._display_area.content.content
        self.assertIsInstance(col, ft.Column)
        header_row = col.controls[0]
        self.assertIsInstance(header_row, ft.Row)
        self.assertEqual(header_row.controls[0].value, "Imagem Processada")
        self.assertEqual(len(header_row.controls), 2)  # Texto + Badge de algoritmo

        # No modo ANALYTICS, a barra de badges deve conter o badge de Algoritmo
        canvas.set_display_mode(DisplayMode.ANALYTICS)
        badge_labels = [c.content.controls[0].value for c in canvas._analytics_badges_row.controls if hasattr(c, "content")]
        self.assertIn("Algoritmo", badge_labels)



"""
test_ui.py — Testes unitários para configurações de UI e temas.
"""

import unittest
import flet as ft

from src.ui.app import APP_TITLE, APP_VERSION
from src.ui import theme


class TestUIConfig(unittest.TestCase):
    """Suíte de testes para metadados e configuração da aplicação."""

    def test_app_version_is_0_1(self) -> None:
        self.assertEqual(APP_VERSION, "0.1")

    def test_app_title_present(self) -> None:
        self.assertTrue(len(APP_TITLE) > 0)
        self.assertIn("PDI", APP_TITLE)


class TestThemeBuilders(unittest.TestCase):
    """Suíte de testes para construtores de tema e componentes estilizados."""

    def test_build_light_theme(self) -> None:
        t = theme.build_light_theme()
        self.assertIsInstance(t, ft.Theme)
        self.assertEqual(t.card_bgcolor, theme.LIGHT_SURFACE_CARD)
        self.assertEqual(t.scaffold_bgcolor, theme.LIGHT_SURFACE)

    def test_build_dark_theme(self) -> None:
        t = theme.build_dark_theme()
        self.assertIsInstance(t, ft.Theme)
        self.assertEqual(t.card_bgcolor, theme.SURFACE_CARD)
        self.assertEqual(t.scaffold_bgcolor, theme.SURFACE)

    def test_card_helper(self) -> None:
        text = ft.Text("Conteúdo de teste")
        c = theme.card(text)
        self.assertIsInstance(c, ft.Container)
        self.assertEqual(c.content, text)
        self.assertEqual(c.border_radius, theme.BORDER_RADIUS)

    def test_section_title_helper(self) -> None:
        st = theme.section_title("Configurações")
        self.assertIsInstance(st, ft.Text)
        self.assertEqual(st.value, "Configurações")

        st_icon = theme.section_title("Configurações", icon=ft.Icons.SETTINGS)
        self.assertIsInstance(st_icon, ft.Row)

    def test_metric_badge_helper(self) -> None:
        badge = theme.metric_badge("PSNR", "35.40 dB")
        self.assertIsInstance(badge, ft.Container)


class TestViewsInstantiation(unittest.TestCase):
    """Suíte de testes para criação e serialização das views."""

    def test_single_view_and_batch_view_creation(self) -> None:
        from unittest.mock import MagicMock
        from src.ui.views.single_view import SingleView
        from src.ui.views.batch_view import BatchView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None

        sv = SingleView(mock_page)
        bv = BatchView(mock_page)

        self.assertIsInstance(sv, ft.Column)
        self.assertIsInstance(bv, ft.Column)

        # Garante que nenhum SegmentedButton usa set no atributo selected
        self.assertIsInstance(sv._gray_category_selector.selected, list)
        self.assertIsInstance(sv._gray_options_selector.selected, list)
        self.assertIsInstance(sv._view_mode_buttons.selected, list)

        # Garante que todos os 5 botões de amostra com tooltips e o botão de cinza existem
        self.assertIsNotNone(sv._btn_sample_portrait)
        self.assertIsNotNone(sv._btn_sample_portrait.tooltip)
        self.assertIsNotNone(sv._btn_sample_benchmark)
        self.assertIsNotNone(sv._btn_sample_benchmark.tooltip)
        self.assertIsNotNone(sv._btn_sample_lena)
        self.assertIsNotNone(sv._btn_sample_lena.tooltip)
        self.assertIsNotNone(sv._btn_sample_ayla)
        self.assertIsNotNone(sv._btn_sample_ayla.tooltip)
        self.assertIsNotNone(sv._btn_sample_pentagono)
        self.assertIsNotNone(sv._btn_sample_pentagono.tooltip)
        self.assertIsNotNone(sv._btn_convert_gray_only)

        # Garante que o card de preview imediato da imagem de entrada existe
        self.assertIsNotNone(sv._input_preview_card)
        self.assertIsNotNone(sv._input_thumbnail)
        self.assertIsNotNone(sv._btn_zoom_input)

        # Garante que o método de abertura do diálogo de zoom existe
        self.assertTrue(callable(getattr(sv, "_open_zoom_dialog", None)))
        self.assertIsNotNone(sv._single_display_container)

    def test_single_view_preview_and_zoom_dialog(self) -> None:
        from unittest.mock import MagicMock
        import numpy as np
        from src.ui.views.single_view import SingleView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None
        mock_page.show_dialog = MagicMock()
        mock_page.update = MagicMock()

        sv = SingleView(mock_page)

        # Testa a atualização do preview com array RGB
        test_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        sv._update_input_preview("Teste RGB", test_arr, is_sample=True)

        self.assertTrue(sv._input_preview_card.visible)
        self.assertIn("100×100 px", sv._input_dim_badge.content.controls[1].value)
        self.assertIn("Colorida RGB", sv._input_type_badge.content.controls[1].value)

        # Testa abertura do diálogo de zoom
        sv._open_zoom_dialog("Teste Zoom Dialog", sv._input_image_bytes)
        mock_page.show_dialog.assert_called_once()
        dialog = mock_page.show_dialog.call_args[0][0]
        self.assertIsInstance(dialog, ft.AlertDialog)
        self.assertTrue(dialog.modal)

    def test_header_and_page_config(self) -> None:
        from unittest.mock import MagicMock
        from src.ui.app import _configure_page, _build_header

        mock_page = MagicMock(spec=ft.Page)
        mock_page.window = MagicMock()

        _configure_page(mock_page)
        self.assertIn("PDI", mock_page.title)
        self.assertIsNotNone(mock_page.window.icon)

        header = _build_header(mock_page)
        self.assertIsInstance(header, ft.Container)


if __name__ == "__main__":
    unittest.main()



"""
test_ui.py — Testes unitários para configurações de UI e temas.
"""

import unittest
import flet as ft

from src.ui.app import APP_TITLE, APP_VERSION
from src.ui import theme


class TestUIConfig(unittest.TestCase):
    """Suíte de testes para metadados e configuração da aplicação."""

    def test_app_version(self) -> None:
        self.assertEqual(APP_VERSION, "0.4")

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

        header, update_header = _build_header(mock_page)
        self.assertIsInstance(header, ft.Container)
        self.assertTrue(callable(update_header))

    def test_batch_view_sample_loading_and_gallery(self) -> None:
        from unittest.mock import MagicMock
        import numpy as np
        from src.core.batch import BatchItemResult, BatchResult
        from src.core.histogram import ImageMetrics
        from src.ui.views.batch_view import BatchView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None
        mock_page.show_dialog = MagicMock()
        mock_page.update = MagicMock()

        bv = BatchView(mock_page)

        # Testa carregamento de amostras do app
        bv._on_select_sample_batch(MagicMock())
        self.assertTrue(bv._queue_container.visible)
        self.assertGreater(len(bv._queue_items), 0)
        self.assertFalse(bv._btn_start.disabled)

        # Testa montagem da galeria de resultados
        fake_metrics = ImageMetrics(mse=12.5, psnr=37.16, unique_levels=16, bits=4)
        fake_item = BatchItemResult(
            filename="teste_amostra.png",
            source_bytes=b"fake_source",
            quantized_bytes=b"fake_quant",
            raw_array=np.zeros((10, 10, 3), dtype=np.uint8),
            gray_array=np.zeros((10, 10), dtype=np.uint8),
            quantized_array=np.zeros((10, 10), dtype=np.uint8),
            metrics=fake_metrics,
            elapsed_seconds=0.05,
            success=True,
        )
        fake_result = BatchResult(
            total=1,
            processed=1,
            items=[fake_item],
            total_elapsed_seconds=0.05,
        )

        bv._on_batch_complete(fake_result)
        self.assertTrue(bv._summary_card.visible)
        self.assertTrue(bv._results_gallery_container.visible)
        self.assertEqual(len(bv._results_gallery.controls), 1)

    def test_single_view_web_upload(self) -> None:
        import asyncio
        import io
        from unittest.mock import AsyncMock, MagicMock
        import numpy as np
        from PIL import Image
        from src.ui.views.single_view import SingleView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None
        mock_page.show_dialog = MagicMock()
        mock_page.update = MagicMock()
        mock_page.web = True

        sv = SingleView(mock_page)

        # Gera bytes de teste PNG
        img_arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        fake_bytes = buf.getvalue()

        # Simula arquivo selecionado pelo FilePicker na Web (com bytes e sem path)
        fake_file = MagicMock()
        fake_file.name = "web_photo.png"
        fake_file.path = None
        fake_file.bytes = fake_bytes

        sv._file_picker.pick_files = AsyncMock(return_value=[fake_file])
        asyncio.run(sv._on_select_image(MagicMock()))

        self.assertIsNotNone(sv._loaded_array)
        self.assertTrue(sv._input_preview_card.visible)
        self.assertFalse(sv._btn_process.disabled)

        # Executa processamento
        sv._run_processing()
        self.assertIsNotNone(sv._raw_image)
        self.assertIsNotNone(sv._quantized_image)
        self.assertIsNotNone(sv._quantized_image_bytes)
        self.assertFalse(sv._btn_inspect.disabled)
        self.assertFalse(sv._btn_save.disabled)

    def test_batch_view_zip_download(self) -> None:
        import asyncio
        import io
        import zipfile
        from unittest.mock import AsyncMock, MagicMock
        from src.core.batch import BatchItemResult, BatchResult
        from src.ui.views.batch_view import BatchView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None
        mock_page.show_dialog = MagicMock()
        mock_page.update = MagicMock()
        mock_page.web = True

        bv = BatchView(mock_page)

        fake_item = BatchItemResult(
            filename="foto1.png",
            quantized_bytes=b"fake_png_data",
            success=True,
        )
        fake_result = BatchResult(total=1, items=[fake_item])
        bv._batch_result = fake_result

        saved_payload = {}

        async def fake_save_file(**kwargs):
            saved_payload.update(kwargs)
            return "lote_quantizado.zip"

        bv._save_picker.save_file = AsyncMock(side_effect=fake_save_file)

        asyncio.run(bv._on_download_zip_clicked(MagicMock()))

        self.assertIn("src_bytes", saved_payload)
        zip_bytes = saved_payload["src_bytes"]
        self.assertIsInstance(zip_bytes, bytes)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            self.assertIn("quantizado_foto1.png", zf.namelist())
            self.assertEqual(zf.read("quantizado_foto1.png"), b"fake_png_data")

    def test_open_histogram_zoom_dialog_and_callbacks(self) -> None:
        from unittest.mock import MagicMock
        import numpy as np
        from src.ui.dialogs import open_histogram_zoom_dialog
        from src.ui.views.single_view import SingleView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None
        mock_page.show_dialog = MagicMock()
        mock_page.update = MagicMock()

        test_data = np.arange(256, dtype=np.uint8).reshape((16, 16))
        open_histogram_zoom_dialog(
            page=mock_page,
            title="Teste Histograma Zoom",
            data=test_data,
        )

        mock_page.show_dialog.assert_called_once()
        dialog = mock_page.show_dialog.call_args[0][0]
        self.assertIsInstance(dialog, ft.AlertDialog)

        sv = SingleView(mock_page)
        sv._gray_image = test_data
        sv._quantized_image = test_data
        sv._open_orig_chart_zoom()
        self.assertEqual(mock_page.show_dialog.call_count, 2)
        sv._open_quant_chart_zoom()
        self.assertEqual(mock_page.show_dialog.call_count, 3)

    def test_unified_analytical_figure_display_and_zoom(self) -> None:
        from unittest.mock import MagicMock
        from src.ui.views.single_view import SingleView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None
        mock_page.show_dialog = MagicMock()
        mock_page.update = MagicMock()

        sv = SingleView(mock_page)
        fake_figure_bytes = b"\x89PNG\r\n\x1a\nfake_figure_bytes"

        sv._figure_bytes = fake_figure_bytes
        sv._color_figure_bytes = fake_figure_bytes

        # Teste de exibição do modo Gráfico Cinza
        sv._display_graph_mode("Uniforme", "Luminância")
        self.assertTrue(sv._graph_container.visible)
        self.assertIn("data:image/png;base64,", sv._graph_display_image.src)

        # Teste de abertura de zoom na figura unificada
        sv._open_graph_figure_zoom()
        mock_page.show_dialog.assert_called_once()
        dialog = mock_page.show_dialog.call_args[0][0]
        self.assertIsInstance(dialog, ft.AlertDialog)

    def test_view_mode_buttons_and_triple_grid_color_mode(self) -> None:
        from unittest.mock import MagicMock
        from src.ui.views.single_view import SingleView

        mock_page = MagicMock(spec=ft.Page)
        mock_page.services = None
        mock_page.update = MagicMock()

        sv = SingleView(mock_page)

        # 1. Modo Grayscale padrão
        self.assertTrue(sv._convert_to_gray)
        segment_values = [s.value for s in sv._view_mode_buttons.segments]
        self.assertNotIn("side_by_side", segment_values)
        self.assertIn("color_side", segment_values)
        self.assertIn("triple", segment_values)

        # 2. Alterna para Modo Colorido (RGB)
        sv._convert_to_gray = False
        sv._update_view_mode_buttons()
        color_seg_values = [s.value for s in sv._view_mode_buttons.segments]
        self.assertNotIn("side_by_side", color_seg_values)
        self.assertNotIn("color_graph", color_seg_values)
        self.assertIn("graph", color_seg_values)
        self.assertIn("triple", color_seg_values)

        # Verifica label da Grade Tripla no modo colorido
        triple_seg = next(s for s in sv._view_mode_buttons.segments if s.value == "triple")
        self.assertIn("Original × RGB × Cinza", triple_seg.label.value)

        # 3. Testa Grade Tripla no modo colorido com imagem RGB
        fake_png = b"\x89PNG\r\n\x1a\nfake"
        sv._color_image_bytes = fake_png
        sv._quantized_image_bytes = fake_png
        sv._gray_quantized_bytes = fake_png
        sv._display_triple_mode(t_name="K-Means", m_name="Luminância")

        self.assertTrue(sv._triple_container.visible)
        self.assertEqual(sv._triple_color_label.value, "1. Original Colorida (RGB)")
        self.assertIn("2. Quantizada com Cores", sv._triple_gray_label.value)
        self.assertIn("3. Quantizada em Tons de Cinza", sv._triple_quant_label.value)

def _create_mock_page():
    from unittest.mock import MagicMock
    mock_page = MagicMock(spec=ft.Page)
    mock_page.services = None
    mock_page.show_dialog = MagicMock()
    mock_page.update = MagicMock()
    return mock_page


from src.ui.views.single_view import SingleView


class TestDynamicTabsAndZipExport(unittest.TestCase):
    """Testes para abas dinâmicas, exportação ZIP de comparações e seletor de algoritmo."""

    def test_dynamic_algorithm_tabs_both_technique(self):
        page = _create_mock_page()
        sv = SingleView(page)
        sv._selected_technique_key = "BOTH"
        sv._bits_value = 4
        sv._update_view_mode_buttons()

        segment_values = [s.value for s in sv._view_mode_buttons.segments]
        self.assertIn("side_unif_kmeans", segment_values)
        self.assertIn("orig_uniform", segment_values)
        self.assertIn("orig_kmeans", segment_values)

        seg_side = next(s for s in sv._view_mode_buttons.segments if s.value == "side_unif_kmeans")
        self.assertIn("Uniforme × K-Means (4b)", seg_side.label.value)

    def test_zip_generation_and_report(self):
        import zipfile
        import io
        from pathlib import Path
        page = _create_mock_page()
        sv = SingleView(page)
        sv._source_path = Path("/tmp/foto_teste.png")
        sv._bits_value = 3
        sv._selected_technique_key = "BOTH"

        fake_png = b"\x89PNG\r\n\x1a\nfake"
        sv._color_image_bytes = fake_png
        sv._gray_image_bytes = fake_png
        sv._direct_quantized_bytes = fake_png
        sv._quantized_image_bytes = fake_png
        sv._dither_image_bytes = fake_png
        sv._figure_bytes = fake_png

        from src.core.histogram import ImageMetrics
        sv._direct_metrics = ImageMetrics(mse=12.5, psnr=37.16, unique_levels=8, bits=3)
        sv._kmeans_metrics = ImageMetrics(mse=8.2, psnr=38.99, unique_levels=8, bits=3)

        # Testa banner de execução
        sv._update_execution_summary()
        self.assertTrue(sv._execution_summary_card.visible)
        self.assertFalse(sv._btn_download_all_zip.disabled)
        self.assertTrue(sv._btn_quick_zip.visible)

        # Testa geração do arquivo ZIP
        zip_data, zip_filename = sv._generate_comparison_zip_bytes()
        self.assertEqual(zip_filename, "foto_teste_comparações_3bits.zip")

        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf:
            namelist = zf.namelist()
            self.assertIn("1_original_foto_teste_colorida_rgb.png", namelist)
            self.assertIn("3_quantizada_uniforme_3bits.png", namelist)
            self.assertIn("4_quantizada_kmeans_3bits.png", namelist)
            self.assertIn("5_quantizada_floyd_steinberg_3bits.png", namelist)
            self.assertIn("painel_analitico_histogramas_3bits.png", namelist)
            self.assertIn("relatorio_metricas.txt", namelist)

            report = zf.read("relatorio_metricas.txt").decode("utf-8")
            self.assertIn("RELATÓRIO TÉCNICO DE PROCESSAMENTO", report)
            self.assertIn("37.16 dB", report)
            self.assertIn("38.99 dB", report)

        # Testa chamada assíncrona com array NumPy para prevenir erro de verdade ambígua
        import asyncio
        from unittest.mock import AsyncMock
        import numpy as np
        sv._raw_image = np.zeros((32, 32, 3), dtype=np.uint8)
        saved_zip = {}

        async def fake_save_zip(**kwargs):
            saved_zip.update(kwargs)
            return "teste.zip"

        sv._save_picker.save_file = AsyncMock(side_effect=fake_save_zip)
        asyncio.run(sv._on_download_all_zip(None))
        self.assertIn("src_bytes", saved_zip)
        self.assertIsInstance(saved_zip["src_bytes"], bytes)

    def test_single_algo_switcher(self):
        page = _create_mock_page()
        sv = SingleView(page)
        sv._selected_technique_key = "BOTH"
        sv._bits_value = 2
        fake_png_u = b"\x89PNG\r\n\x1a\nfake_unif"
        fake_png_k = b"\x89PNG\r\n\x1a\nfake_km"
        sv._direct_quantized_bytes = fake_png_u
        sv._quantized_image_bytes = fake_png_k

        # Default é kmeans
        bytes_km, name_km = sv._get_active_single_bytes_and_name()
        self.assertEqual(bytes_km, fake_png_k)
        self.assertIn("kmeans_2bits.png", name_km)

        # Troca para uniforme
        sv._active_single_algo = "uniform"
        bytes_u, name_u = sv._get_active_single_bytes_and_name()
        self.assertEqual(bytes_u, fake_png_u)
        self.assertIn("uniforme_2bits.png", name_u)


if __name__ == "__main__":
    unittest.main()






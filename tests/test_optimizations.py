"""
test_optimizations.py — Testes unitários para as otimizações de performance e arquitetura.

Valida:
1. Downscaling preventivo (máximo 800×800 px) e utilitários de I/O em `src/core/image_io.py`.
2. Carregamento sob demanda (Lazy Loading) do K-Means em `src/core/quantization.py`.
3. Cálculo numérico de histogramas e métricas sem Matplotlib em `src/core/histogram.py`.
4. Geração do mapa térmico de erro residual em puro NumPy em `src/core/inspector.py`.
5. Componente de Histograma Nativo Flet em `src/ui/components/histogram_chart.py`.
"""

import io
import unittest
import numpy as np
from PIL import Image

from src.core.image_io import (
    MAX_IMAGE_DIMENSION,
    array_to_png_bytes,
    make_thumbnail_png,
    open_and_downscale_image,
    preventive_resize,
)
from src.core.grayscale import GrayscaleMethod, to_grayscale
from src.core.histogram import (
    HistogramData,
    calculate_metrics,
    compute_histogram,
    compute_rgb_histogram,
)
from src.core.inspector import extract_pipeline_telemetry
from src.core.quantization import (
    QuantizationTechnique,
    get_kmeans_class,
    is_kmeans_loaded,
    quantize_kmeans,
    quantize_uniform,
)
from src.ui.components.histogram_chart import NativeHistogramChart


class TestOptimizations(unittest.TestCase):
    """Suíte de testes de otimização de memória, I/O e componentes nativos."""

    def test_preventive_resize_large_image(self) -> None:
        """Garante que imagens maiores que max_dim são reduzidas mantendo proporção."""
        large_img = Image.new("RGB", (2000, 1000), color=(128, 64, 32))
        resized = preventive_resize(large_img, max_dim=800)
        self.assertEqual(resized.size, (800, 400))
        large_img.close()
        resized.close()

    def test_preventive_resize_small_image(self) -> None:
        """Garante que imagens dentro do limite não são alteradas."""
        small_img = Image.new("RGB", (400, 300), color=(100, 100, 100))
        resized = preventive_resize(small_img, max_dim=800)
        self.assertEqual(resized.size, (400, 300))
        small_img.close()

    def test_open_and_downscale_image_from_bytes(self) -> None:
        """Testa decodificação e downscaling preventivo de bytes em memória."""
        large_img = Image.new("RGB", (1600, 1200), color=(200, 150, 100))
        buf = io.BytesIO()
        large_img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        arr = open_and_downscale_image(raw_bytes, max_dim=800)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(arr.shape, (600, 800, 3))
        self.assertLessEqual(max(arr.shape[:2]), 800)

    def test_array_to_png_bytes(self) -> None:
        """Testa conversão de array para bytes PNG."""
        arr = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        png_bytes = array_to_png_bytes(arr)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_make_thumbnail_png(self) -> None:
        """Testa geração de miniatura leve em PNG."""
        arr = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        thumb_bytes = make_thumbnail_png(arr, max_size=120)
        self.assertTrue(thumb_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(io.BytesIO(thumb_bytes)) as thumb_img:
            self.assertLessEqual(max(thumb_img.size), 120)

    def test_lazy_kmeans_loading(self) -> None:
        """Testa importação sob demanda de KMeans."""
        KMeansClass = get_kmeans_class()
        self.assertTrue(is_kmeans_loaded())
        self.assertIsNotNone(KMeansClass)

        # Executa quantização KMeans em array pequeno
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        quant = quantize_kmeans(gray, bits=2)
        self.assertEqual(quant.shape, (64, 64))
        self.assertEqual(quant.dtype, np.uint8)

    def test_pure_numpy_histogram_and_metrics(self) -> None:
        """Valida histogramas e métricas computadas via NumPy."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:50, :] = 50
        gray[50:, :] = 200

        h = compute_histogram(gray)
        self.assertIsInstance(h, HistogramData)
        self.assertEqual(len(h.counts), 256)
        self.assertEqual(h.counts[50], 5000)
        self.assertEqual(h.counts[200], 5000)

        rgb = np.stack([gray, gray, gray], axis=2)
        rgb_h = compute_rgb_histogram(rgb)
        self.assertIn("R", rgb_h)
        self.assertIn("G", rgb_h)
        self.assertIn("B", rgb_h)

        quant = quantize_uniform(gray, bits=2)
        metrics = calculate_metrics(gray, quant, bits=2)
        self.assertGreater(metrics.psnr, 0.0)
        self.assertGreaterEqual(metrics.unique_levels, 1)

    def test_inspector_heatmap_pure_numpy(self) -> None:
        """Valida que o heatmap é gerado em PNG usando puro NumPy sem Matplotlib."""
        raw = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        gray = to_grayscale(raw, GrayscaleMethod.LUMINANCE)
        quant = quantize_uniform(gray, bits=3)

        telemetry = extract_pipeline_telemetry(
            raw_image=raw,
            gray_image=gray,
            quantized_image=quant,
            bits=3,
            technique=QuantizationTechnique.UNIFORM,
            method=GrayscaleMethod.LUMINANCE,
        )
        self.assertIsNotNone(telemetry.heatmap_bytes)
        self.assertTrue(telemetry.heatmap_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_native_histogram_chart_creation(self) -> None:
        """Valida a criação e atualização de dados no NativeHistogramChart."""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        chart = NativeHistogramChart(title="Teste Histograma", image_or_data=gray)
        self.assertIsNotNone(chart._canvas)
        self.assertGreater(len(chart._canvas.shapes), 0)

        # Atualiza com dados quantizados
        chart.set_data(gray, is_quantized=True)
        self.assertGreater(len(chart._canvas.shapes), 0)

        # Atualiza com dados RGB
        rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        chart.set_data(rgb, is_rgb=True)
        self.assertGreater(len(chart._canvas.shapes), 0)


if __name__ == "__main__":
    unittest.main()


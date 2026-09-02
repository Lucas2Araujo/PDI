"""
test_histogram.py — Testes unitários do módulo de histogramas e métricas de qualidade.
"""

import math
import unittest
import numpy as np

from src.core.histogram import (
    calculate_metrics,
    compute_histogram,
    generate_color_comparison_figure,
    generate_comparison_figure,
    generate_dither_comparison_figure,
    generate_full_comparison_figure,
)


class TestComputeHistogram(unittest.TestCase):
    """Suíte de testes para cálculo de histograma."""

    def test_counts_sum_equals_total_pixels(self) -> None:
        img = np.random.randint(0, 256, (50, 40), dtype=np.uint8)
        hist = compute_histogram(img)
        self.assertEqual(int(np.sum(hist.counts)), 50 * 40)
        self.assertEqual(len(hist.counts), 256)
        self.assertEqual(len(hist.bin_edges), 257)

    def test_flat_uniform_image(self) -> None:
        img = np.full((10, 10), 120, dtype=np.uint8)
        hist = compute_histogram(img)
        self.assertEqual(hist.counts[120], 100)
        self.assertEqual(int(np.sum(hist.counts)), 100)
        self.assertEqual(int(np.sum(hist.counts == 0)), 255)

    def test_compute_rgb_histogram(self) -> None:
        from src.core.histogram import compute_rgb_histogram
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        hists = compute_rgb_histogram(img)
        self.assertIn("R", hists)
        self.assertIn("G", hists)
        self.assertIn("B", hists)
        self.assertEqual(int(np.sum(hists["R"].counts)), 400)
        self.assertEqual(int(np.sum(hists["G"].counts)), 400)
        self.assertEqual(int(np.sum(hists["B"].counts)), 400)


class TestCalculateMetrics(unittest.TestCase):
    """Suíte de testes para cálculo de MSE, PSNR e níveis únicos."""

    def test_identical_images(self) -> None:
        img = np.arange(100, dtype=np.uint8).reshape(10, 10)
        metrics = calculate_metrics(img, img, bits=8)
        self.assertEqual(metrics.mse, 0.0)
        self.assertEqual(metrics.psnr, float("inf"))
        self.assertEqual(metrics.unique_levels, 100)
        self.assertEqual(metrics.bits, 8)

    def test_known_difference_2d(self) -> None:
        orig = np.zeros((10, 10), dtype=np.uint8)
        quant = np.full((10, 10), 10, dtype=np.uint8)
        metrics = calculate_metrics(orig, quant, bits=4)

        # MSE = 10^2 = 100.0
        self.assertAlmostEqual(metrics.mse, 100.0, places=4)
        # PSNR = 20 * log10(255) - 10 * log10(100) = 48.1308 - 20 = 28.1308 dB
        expected_psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(100.0)
        self.assertAlmostEqual(metrics.psnr, expected_psnr, places=4)
        self.assertEqual(metrics.unique_levels, 1)

    def test_known_difference_3d_rgb(self) -> None:
        orig = np.zeros((10, 10, 3), dtype=np.uint8)
        quant = np.full((10, 10, 3), 10, dtype=np.uint8)
        metrics = calculate_metrics(orig, quant, bits=4)

        self.assertAlmostEqual(metrics.mse, 100.0, places=4)
        expected_psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(100.0)
        self.assertAlmostEqual(metrics.psnr, expected_psnr, places=4)
        self.assertEqual(metrics.unique_levels, 1)


class TestGenerateFigures(unittest.TestCase):
    """Suíte de testes para geração de figuras comparativas em bytes PNG."""

    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

    def setUp(self) -> None:
        self.orig = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        self.quant1 = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        self.quant2 = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        self.color_img = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)

    def test_generate_comparison_figure(self) -> None:
        fig_bytes = generate_comparison_figure(
            original=self.orig,
            quantized=self.quant1,
            bits=4,
            technique_name="Quantização Uniforme",
        )
        self.assertIsInstance(fig_bytes, bytes)
        self.assertTrue(fig_bytes.startswith(self.PNG_MAGIC))

    def test_generate_full_comparison_figure(self) -> None:
        fig_bytes = generate_full_comparison_figure(
            original=self.orig,
            uniform=self.quant1,
            kmeans=self.quant2,
            bits=4,
        )
        self.assertIsInstance(fig_bytes, bytes)
        self.assertTrue(fig_bytes.startswith(self.PNG_MAGIC))

    def test_generate_color_comparison_figure(self) -> None:
        # Com imagem cinza intermediária (grade 2x3)
        fig_bytes_3 = generate_color_comparison_figure(
            color_image=self.color_img,
            quantized=self.quant1,
            bits=4,
            technique_name="K-Means",
            gray_image=self.orig,
        )
        self.assertIsInstance(fig_bytes_3, bytes)
        self.assertTrue(fig_bytes_3.startswith(self.PNG_MAGIC))

        # Sem imagem cinza intermediária (grade 2x2)
        fig_bytes_2 = generate_color_comparison_figure(
            color_image=self.color_img,
            quantized=self.quant1,
            bits=4,
            technique_name="Uniforme",
        )
        self.assertIsInstance(fig_bytes_2, bytes)
        self.assertTrue(fig_bytes_2.startswith(self.PNG_MAGIC))

    def test_generate_dither_comparison_figure(self) -> None:
        fig_bytes = generate_dither_comparison_figure(
            original_gray=self.orig,
            direct_quantized=self.quant1,
            dither_quantized=self.quant2,
            bits=4,
            mse_direct=12.5,
            psnr_direct=37.1,
            mse_dither=10.2,
            psnr_dither=38.0,
        )
        self.assertIsInstance(fig_bytes, bytes)
        self.assertTrue(fig_bytes.startswith(self.PNG_MAGIC))


if __name__ == "__main__":
    unittest.main()


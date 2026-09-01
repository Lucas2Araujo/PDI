"""
test_inspector.py — Testes unitários para o extrator de telemetria e raio-x didático.
"""

import unittest
import numpy as np

from src.core.grayscale import GrayscaleMethod, to_grayscale
from src.core.inspector import extract_pipeline_telemetry, PipelineTelemetry
from src.core.quantization import QuantizationTechnique, quantize


class TestInspector(unittest.TestCase):
    """Suíte de testes para o módulo de inspeção didática e telemetria."""

    def setUp(self) -> None:
        np.random.seed(42)
        # Imagem sintética RGB 32x32
        self.rgb_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        # Imagem sintética monocromática 32x32
        self.gray_img = to_grayscale(self.rgb_img, GrayscaleMethod.LUMINANCE)

    def test_extract_telemetry_rgb_uniform(self) -> None:
        quant = quantize(self.gray_img, bits=2, technique=QuantizationTechnique.UNIFORM)
        telemetry = extract_pipeline_telemetry(
            raw_image=self.rgb_img,
            gray_image=self.gray_img,
            quantized_image=quant,
            bits=2,
            technique=QuantizationTechnique.UNIFORM,
            method=GrayscaleMethod.LUMINANCE,
        )

        self.assertIsInstance(telemetry, PipelineTelemetry)
        self.assertTrue(telemetry.is_color)
        self.assertEqual(telemetry.image_shape, (32, 32, 3))
        self.assertEqual(telemetry.bits, 2)
        self.assertEqual(telemetry.n_levels, 4)
        self.assertEqual(telemetry.sample_raw.shape, (5, 5, 3))
        self.assertEqual(telemetry.sample_gray.shape, (5, 5))
        self.assertEqual(telemetry.sample_quantized.shape, (5, 5))
        self.assertEqual(len(telemetry.pixel_calculations), 25)
        self.assertGreater(len(telemetry.heatmap_bytes), 100)
        self.assertGreaterEqual(telemetry.mse, 0.0)
        self.assertGreater(telemetry.psnr, 0.0)
        self.assertEqual(len(telemetry.quant_info.table_rows), 4)

    def test_extract_telemetry_grayscale_kmeans(self) -> None:
        quant = quantize(self.gray_img, bits=3, technique=QuantizationTechnique.KMEANS)
        telemetry = extract_pipeline_telemetry(
            raw_image=self.gray_img,
            gray_image=self.gray_img,
            quantized_image=quant,
            bits=3,
            technique=QuantizationTechnique.KMEANS,
            method=GrayscaleMethod.LUMINANCE,
        )

        self.assertFalse(telemetry.is_color)
        self.assertEqual(telemetry.bits, 3)
        self.assertEqual(telemetry.n_levels, 8)
        self.assertEqual(len(telemetry.pixel_calculations), 25)
        self.assertIn("K-Means", telemetry.quantization_technique_name)
        self.assertGreater(len(telemetry.heatmap_bytes), 100)

    def test_extract_telemetry_histogram_technique(self) -> None:
        quant = quantize(self.gray_img, bits=1, technique=QuantizationTechnique.HISTOGRAM)
        telemetry = extract_pipeline_telemetry(
            raw_image=self.rgb_img,
            gray_image=self.gray_img,
            quantized_image=quant,
            bits=1,
            technique=QuantizationTechnique.HISTOGRAM,
            method=GrayscaleMethod.AVERAGE,
        )

        self.assertEqual(telemetry.bits, 1)
        self.assertEqual(telemetry.n_levels, 2)
        self.assertEqual(len(telemetry.quant_info.table_rows), 2)
        self.assertIn("Média", telemetry.grayscale_method_name)


if __name__ == "__main__":
    unittest.main()

"""
test_quantization.py — Testes unitários do módulo de quantização uniforme e não-uniforme (K-Means).
"""

import unittest
import numpy as np

from src.core.quantization import (
    QuantizationTechnique,
    quantize,
    quantize_histogram,
    quantize_kmeans,
    quantize_uniform,
    technique_label,
)


class TestQuantizeUniform(unittest.TestCase):
    """Suíte de testes para quantização uniforme."""

    def setUp(self) -> None:
        # Gradiente suave de 0 a 255
        self.gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (10, 1))

    def test_all_bit_depths_1_to_8(self) -> None:
        """Verifica a quantização uniforme para todos os níveis de bits (1 a 8)."""
        for bits in range(1, 9):
            quantized = quantize_uniform(self.gradient, bits=bits)
            self.assertEqual(quantized.shape, self.gradient.shape)
            self.assertEqual(quantized.dtype, np.uint8)
            self.assertTrue(np.all(quantized >= 0))
            self.assertTrue(np.all(quantized <= 255))

            # O número de níveis únicos não pode ultrapassar 2^bits
            max_levels = 2 ** bits
            unique_levels = len(np.unique(quantized))
            self.assertLessEqual(unique_levels, max_levels)

    def test_1_bit_uniform(self) -> None:
        """Quantização de 1 bit deve gerar exatamente 2 níveis (0 e 255)."""
        quantized = quantize_uniform(self.gradient, bits=1)
        unique = np.unique(quantized)
        self.assertLessEqual(len(unique), 2)
        self.assertIn(0, unique)
        self.assertIn(255, unique)

    def test_constant_images(self) -> None:
        """Testa imagens constantes (tudo preto e tudo branco)."""
        black = np.zeros((20, 20), dtype=np.uint8)
        white = np.full((20, 20), 255, dtype=np.uint8)

        for bits in (1, 2, 4, 8):
            q_black = quantize_uniform(black, bits=bits)
            self.assertEqual(len(np.unique(q_black)), 1)

            q_white = quantize_uniform(white, bits=bits)
            self.assertEqual(len(np.unique(q_white)), 1)

    def test_invalid_bits_raise_error(self) -> None:
        """Verifica se bits fora do intervalo [1, 8] geram ValueError."""
        for invalid_bits in (0, -1, 9, 16, 3.5):
            with self.assertRaises(ValueError):
                quantize_uniform(self.gradient, bits=invalid_bits)  # type: ignore

    def test_invalid_image_inputs_raise_error(self) -> None:
        """Verifica se entradas não-2D, não-uint8 ou não-numpy geram erro."""
        with self.assertRaises(ValueError):
            quantize_uniform(np.zeros((10, 10, 3), dtype=np.uint8), bits=4)

        with self.assertRaises(ValueError):
            quantize_uniform(np.zeros((10, 10), dtype=np.float32), bits=4)

        with self.assertRaises(ValueError):
            quantize_uniform([[0, 128], [255, 64]], bits=4)  # type: ignore


class TestQuantizeKMeans(unittest.TestCase):
    """Suíte de testes para quantização K-Means."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.synthetic = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_kmeans_bit_depths(self) -> None:
        """Verifica a quantização K-Means para diferentes profundidades de bits."""
        for bits in (1, 2, 3, 4):
            quantized = quantize_kmeans(self.synthetic, bits=bits, random_state=42)
            self.assertEqual(quantized.shape, self.synthetic.shape)
            self.assertEqual(quantized.dtype, np.uint8)

            max_clusters = 2 ** bits
            unique_levels = len(np.unique(quantized))
            self.assertLessEqual(unique_levels, max_clusters)

    def test_kmeans_reproducibility(self) -> None:
        """Verifica se o mesmo random_state gera resultados idênticos."""
        q1 = quantize_kmeans(self.synthetic, bits=3, random_state=123)
        q2 = quantize_kmeans(self.synthetic, bits=3, random_state=123)
        np.testing.assert_array_equal(q1, q2)

    def test_invalid_bits_kmeans(self) -> None:
        with self.assertRaises(ValueError):
            quantize_kmeans(self.synthetic, bits=0)

        with self.assertRaises(ValueError):
            quantize_kmeans(self.synthetic, bits=10)


class TestQuantizeHistogram(unittest.TestCase):
    """Suíte de testes para quantização baseada em histograma."""

    def setUp(self) -> None:
        self.gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (10, 1))

    def test_histogram_bit_depths_1_to_8(self) -> None:
        for bits in range(1, 9):
            quantized = quantize_histogram(self.gradient, bits=bits)
            self.assertEqual(quantized.shape, self.gradient.shape)
            self.assertEqual(quantized.dtype, np.uint8)

            max_levels = 2 ** bits
            unique_levels = len(np.unique(quantized))
            self.assertLessEqual(unique_levels, max_levels)

    def test_invalid_bits_histogram(self) -> None:
        with self.assertRaises(ValueError):
            quantize_histogram(self.gradient, bits=0)

        with self.assertRaises(ValueError):
            quantize_histogram(self.gradient, bits=9)


class TestQuantizeDispatcher(unittest.TestCase):
    """Testa a função quantize que direciona para a técnica selecionada."""

    def setUp(self) -> None:
        self.img = np.arange(64, dtype=np.uint8).reshape(8, 8)

    def test_dispatch_uniform(self) -> None:
        q1 = quantize(self.img, bits=3, technique=QuantizationTechnique.UNIFORM)
        q2 = quantize_uniform(self.img, bits=3)
        np.testing.assert_array_equal(q1, q2)

    def test_dispatch_kmeans(self) -> None:
        q1 = quantize(self.img, bits=2, technique=QuantizationTechnique.KMEANS)
        self.assertEqual(q1.dtype, np.uint8)
        self.assertEqual(q1.shape, self.img.shape)

    def test_dispatch_histogram(self) -> None:
        q1 = quantize(self.img, bits=2, technique=QuantizationTechnique.HISTOGRAM)
        q2 = quantize_histogram(self.img, bits=2)
        np.testing.assert_array_equal(q1, q2)

    def test_unknown_technique_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            quantize(self.img, bits=4, technique="UNKNOWN")  # type: ignore


class TestTechniqueLabel(unittest.TestCase):
    """Testa as descrições das técnicas de quantização."""

    def test_all_enum_members_have_labels(self) -> None:
        for tech in QuantizationTechnique:
            label = technique_label(tech)
            self.assertIsInstance(label, str)
            self.assertTrue(len(label) > 0)


if __name__ == "__main__":
    unittest.main()

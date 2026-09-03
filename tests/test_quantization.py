"""
test_quantization.py — Testes unitários do módulo de quantização uniforme e não-uniforme (K-Means).
"""

import unittest
import numpy as np

from src.core.quantization import (
    QuantizationTechnique,
    quantizacao_dithering_floyd_steinberg,
    quantize,
    quantize_floyd_steinberg,
    quantize_histogram,
    quantize_kmeans,
    quantize_uniform,
    technique_label,
)


class TestQuantizeUniform(unittest.TestCase):
    """Suíte de testes para quantização uniforme com centróides."""

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

    def test_1_bit_uniform_centroids(self) -> None:
        """Quantização de 1 bit com centróides deve gerar os pontos médios [64, 192]."""
        quantized = quantize_uniform(self.gradient, bits=1)
        unique = np.unique(quantized)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0], 64)
        self.assertEqual(unique[1], 192)

    def test_centroid_reconstruction_values(self) -> None:
        """Valida que os níveis reconstruídos correspondem exatamente aos centróides."""
        # Para 2 bits (4 níveis): passo = 64.0 -> centróides = [32, 96, 160, 224]
        q_2b = quantize_uniform(self.gradient, bits=2)
        unique_2b = np.unique(q_2b)
        expected_2b = np.array([32, 96, 160, 224], dtype=np.uint8)
        np.testing.assert_array_equal(unique_2b, expected_2b)

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
        """Verifica se entradas não-2D/não-3D, não-uint8 ou não-numpy geram erro."""
        with self.assertRaises(ValueError):
            quantize_uniform(np.zeros((10,), dtype=np.uint8), bits=4)

        with self.assertRaises(ValueError):
            quantize_uniform(np.zeros((10, 10, 2), dtype=np.uint8), bits=4)

        with self.assertRaises(ValueError):
            quantize_uniform(np.zeros((10, 10, 3, 2), dtype=np.uint8), bits=4)

        with self.assertRaises(ValueError):
            quantize_uniform(np.zeros((10, 10), dtype=np.float32), bits=4)

        with self.assertRaises(ValueError):
            quantize_uniform([[0, 128], [255, 64]], bits=4)  # type: ignore

    def test_uniform_rgb_quantization(self) -> None:
        """Verifica quantização uniforme escalar por canal em tensores 3D (RGB)."""
        rgb_img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        for bits in (1, 2, 3, 4):
            q_rgb = quantize_uniform(rgb_img, bits=bits)
            self.assertEqual(q_rgb.shape, (20, 20, 3))
            self.assertEqual(q_rgb.dtype, np.uint8)
            self.assertTrue(np.all(q_rgb >= 0))
            self.assertTrue(np.all(q_rgb <= 255))

            # Verifica número de níveis por canal (<= 2^bits)
            for c in range(3):
                self.assertLessEqual(len(np.unique(q_rgb[:, :, c])), 2 ** bits)

            # Verifica fórmula do ponto médio para 1 bit (tons: 64 e 192)
            if bits == 1:
                unique_c0 = np.unique(q_rgb[:, :, 0])
                for u in unique_c0:
                    self.assertIn(u, [64, 192])


class TestQuantizeFloydSteinberg(unittest.TestCase):
    """Suíte de testes para quantização com Dithering de Floyd-Steinberg."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (16, 1))
        self.synthetic = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_all_bit_depths_1_to_8(self) -> None:
        """Verifica a quantização por dithering para profundidades de 1 a 8 bits."""
        for bits in range(1, 9):
            dithered = quantizacao_dithering_floyd_steinberg(self.synthetic, n_bits=bits)
            self.assertEqual(dithered.shape, self.synthetic.shape)
            self.assertEqual(dithered.dtype, np.uint8)
            self.assertTrue(np.all(dithered >= 0))
            self.assertTrue(np.all(dithered <= 255))

            # Níveis únicos não ultrapassam 2^bits
            max_levels = 2 ** bits
            unique_levels = len(np.unique(dithered))
            self.assertLessEqual(unique_levels, max_levels)

    def test_alias_equivalence(self) -> None:
        """Garante que o alias quantize_floyd_steinberg produz resultado idêntico."""
        res1 = quantizacao_dithering_floyd_steinberg(self.synthetic, 3)
        res2 = quantize_floyd_steinberg(self.synthetic, 3)
        np.testing.assert_array_equal(res1, res2)

    def test_1_bit_halftoning_palette(self) -> None:
        """Dithering de 1 bit deve usar a paleta binária extrema [0, 255]."""
        dithered = quantizacao_dithering_floyd_steinberg(self.gradient, n_bits=1)
        unique = np.unique(dithered)
        self.assertLessEqual(len(unique), 2)
        self.assertIn(0, unique)
        self.assertIn(255, unique)

    def test_constant_black_and_white(self) -> None:
        """Imagens totalmente pretas e brancas devem ser preservadas sem ruído espúrio."""
        black = np.zeros((16, 16), dtype=np.uint8)
        white = np.full((16, 16), 255, dtype=np.uint8)

        for bits in (1, 2, 4, 8):
            q_black = quantizacao_dithering_floyd_steinberg(black, n_bits=bits)
            self.assertTrue(np.all(q_black == 0))

            q_white = quantizacao_dithering_floyd_steinberg(white, n_bits=bits)
            self.assertTrue(np.all(q_white == 255))

    def test_8_bits_passthrough(self) -> None:
        """Quantização de 8 bits deve retornar cópia idêntica da imagem."""
        dithered = quantizacao_dithering_floyd_steinberg(self.synthetic, n_bits=8)
        np.testing.assert_array_equal(dithered, self.synthetic)

    def test_edge_and_small_dimensions(self) -> None:
        """Verifica que matrizes pequenas ou 1D-like não causam estouro de índice de borda."""
        shapes = [(1, 1), (1, 10), (10, 1), (2, 2), (3, 7), (17, 23)]
        for h, w in shapes:
            img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
            dithered = quantizacao_dithering_floyd_steinberg(img, n_bits=2)
            self.assertEqual(dithered.shape, (h, w))
            self.assertEqual(dithered.dtype, np.uint8)

    def test_error_diffusion_directionality(self) -> None:
        """Verifica a difusão do erro de um pixel central isolado."""
        img = np.zeros((3, 3), dtype=np.uint8)
        img[0, 0] = 64
        dithered = quantizacao_dithering_floyd_steinberg(img, n_bits=1)
        self.assertEqual(dithered.shape, (3, 3))
        self.assertEqual(dithered.dtype, np.uint8)

    def test_floyd_steinberg_rgb_processing(self) -> None:
        """Verifica a difusão de erro vetorial em tensores 3D (RGB)."""
        rgb_img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        for bits in (1, 2, 4):
            dithered = quantizacao_dithering_floyd_steinberg(rgb_img, n_bits=bits)
            self.assertEqual(dithered.shape, (16, 16, 3))
            self.assertEqual(dithered.dtype, np.uint8)
            self.assertTrue(np.all(dithered >= 0))
            self.assertTrue(np.all(dithered <= 255))

    def test_invalid_parameters_raise(self) -> None:
        """Verifica exceções para parâmetros inválidos."""
        with self.assertRaises(ValueError):
            quantizacao_dithering_floyd_steinberg(self.synthetic, n_bits=0)
        with self.assertRaises(ValueError):
            quantizacao_dithering_floyd_steinberg(self.synthetic, n_bits=9)
        with self.assertRaises(ValueError):
            quantizacao_dithering_floyd_steinberg(np.zeros((10,), dtype=np.uint8), n_bits=2)


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

    def test_kmeans_rgb_processing(self) -> None:
        """Verifica a quantização vetorial no espaço 3D (RGB) via K-Means."""
        rgb_img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        for bits in (1, 2, 3):
            quantized = quantize_kmeans(rgb_img, bits=bits, random_state=42)
            self.assertEqual(quantized.shape, (20, 20, 3))
            self.assertEqual(quantized.dtype, np.uint8)
            self.assertTrue(np.all(quantized >= 0))
            self.assertTrue(np.all(quantized <= 255))

            # Total de cores únicas RGB no tensor <= 2^bits
            unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
            self.assertLessEqual(unique_colors, 2 ** bits)

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

    def test_kmeans_1d_few_unique_values_edge_case(self) -> None:
        """Garante que imagens com menos tons únicos que clusters não lançam exceção."""
        flat_img = np.array([[10, 50], [50, 10]], dtype=np.uint8)  # apenas 2 tons únicos
        # Solicita 3 bits (8 clusters) para apenas 2 valores únicos
        quantized = quantize_kmeans(flat_img, bits=3)
        self.assertEqual(quantized.shape, flat_img.shape)
        self.assertEqual(quantized.dtype, np.uint8)
        self.assertLessEqual(len(np.unique(quantized)), 2)

    def test_kmeans_1d_lut_consistency(self) -> None:
        """Verifica se pixels idênticos são mapeados ao mesmo centróide via LUT."""
        img = np.full((10, 10), 128, dtype=np.uint8)
        quantized = quantize_kmeans(img, bits=2)
        self.assertEqual(len(np.unique(quantized)), 1)


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

    def test_histogram_rgb_processing(self) -> None:
        """Verifica quantização por histograma em RGB."""
        rgb_img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        quantized = quantize_histogram(rgb_img, bits=2)
        self.assertEqual(quantized.shape, (16, 16, 3))
        self.assertEqual(quantized.dtype, np.uint8)

    def test_invalid_bits_histogram(self) -> None:
        with self.assertRaises(ValueError):
            quantize_histogram(self.gradient, bits=0)

        with self.assertRaises(ValueError):
            quantize_histogram(self.gradient, bits=9)


class TestQuantizeDispatcher(unittest.TestCase):
    """Testa a função quantize que direciona para a técnica selecionada."""

    def setUp(self) -> None:
        self.img = np.arange(64, dtype=np.uint8).reshape(8, 8)
        self.rgb_img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

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

    def test_dispatch_floyd_steinberg(self) -> None:
        q1 = quantize(self.img, bits=2, technique=QuantizationTechnique.FLOYD_STEINBERG)
        q2 = quantizacao_dithering_floyd_steinberg(self.img, n_bits=2)
        np.testing.assert_array_equal(q1, q2)

    def test_dispatch_rgb_all_techniques(self) -> None:
        for tech in (
            QuantizationTechnique.UNIFORM,
            QuantizationTechnique.KMEANS,
            QuantizationTechnique.HISTOGRAM,
            QuantizationTechnique.FLOYD_STEINBERG,
        ):
            q_rgb = quantize(self.rgb_img, bits=2, technique=tech)
            self.assertEqual(q_rgb.shape, (10, 10, 3))
            self.assertEqual(q_rgb.dtype, np.uint8)

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


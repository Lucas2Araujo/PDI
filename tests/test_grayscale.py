"""
test_grayscale.py — Testes unitários do módulo de conversão para escala de cinza.
"""

import unittest
import numpy as np

from src.core.grayscale import GrayscaleMethod, method_label, to_grayscale


class TestToGrayscale(unittest.TestCase):
    """Suíte de testes para a função to_grayscale."""

    def setUp(self) -> None:
        # Cria imagens sintéticas padrão
        self.pure_red = np.zeros((10, 10, 3), dtype=np.uint8)
        self.pure_red[:, :, 0] = 255

        self.pure_green = np.zeros((10, 10, 3), dtype=np.uint8)
        self.pure_green[:, :, 1] = 255

        self.pure_blue = np.zeros((10, 10, 3), dtype=np.uint8)
        self.pure_blue[:, :, 2] = 255

        self.pure_white = np.full((10, 10, 3), 255, dtype=np.uint8)
        self.pure_black = np.zeros((10, 10, 3), dtype=np.uint8)

    def test_luminance_known_values(self) -> None:
        """Verifica se a fórmula ITU-R BT.601 é calculada com os pesos corretos."""
        # R=255 -> 0.2989 * 255 = 76.2195 -> 76
        res_r = to_grayscale(self.pure_red, GrayscaleMethod.LUMINANCE)
        self.assertEqual(res_r.shape, (10, 10))
        self.assertEqual(res_r.dtype, np.uint8)
        self.assertTrue(np.all(res_r == 76))

        # G=255 -> 0.5870 * 255 = 149.685 -> 150
        res_g = to_grayscale(self.pure_green, GrayscaleMethod.LUMINANCE)
        self.assertTrue(np.all(res_g == 150))

        # B=255 -> 0.1140 * 255 = 29.07 -> 29
        res_b = to_grayscale(self.pure_blue, GrayscaleMethod.LUMINANCE)
        self.assertTrue(np.all(res_b == 29))

        # Branco = 255, Preto = 0
        res_w = to_grayscale(self.pure_white, GrayscaleMethod.LUMINANCE)
        self.assertTrue(np.all(res_w == 255))
        res_k = to_grayscale(self.pure_black, GrayscaleMethod.LUMINANCE)
        self.assertTrue(np.all(res_k == 0))

    def test_average_known_values(self) -> None:
        """Verifica se a média aritmética simples (R+G+B)/3 é calculada corretamente."""
        # 255 / 3 = 85
        res_r = to_grayscale(self.pure_red, GrayscaleMethod.AVERAGE)
        self.assertEqual(res_r.dtype, np.uint8)
        self.assertTrue(np.all(res_r == 85))

        res_g = to_grayscale(self.pure_green, GrayscaleMethod.AVERAGE)
        self.assertTrue(np.all(res_g == 85))

        res_b = to_grayscale(self.pure_blue, GrayscaleMethod.AVERAGE)
        self.assertTrue(np.all(res_b == 85))

    def test_channel_r_extraction(self) -> None:
        """Verifica o isolamento do canal vermelho."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 180
        img[:, :, 1] = 50
        img[:, :, 2] = 20
        res = to_grayscale(img, GrayscaleMethod.CHANNEL_R)
        self.assertTrue(np.all(res == 180))

    def test_channel_g_extraction(self) -> None:
        """Verifica o isolamento do canal verde."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 180
        img[:, :, 1] = 120
        img[:, :, 2] = 20
        res = to_grayscale(img, GrayscaleMethod.CHANNEL_G)
        self.assertTrue(np.all(res == 120))

    def test_channel_b_extraction(self) -> None:
        """Verifica o isolamento do canal azul."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 180
        img[:, :, 1] = 50
        img[:, :, 2] = 210
        res = to_grayscale(img, GrayscaleMethod.CHANNEL_B)
        self.assertTrue(np.all(res == 210))

    def test_rgba_alpha_channel_discarded(self) -> None:
        """Verifica se o canal alpha (4º canal) é descartado corretamente."""
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255
        rgba[:, :, 3] = 128  # Alpha semi-transparente
        res = to_grayscale(rgba, GrayscaleMethod.LUMINANCE)
        self.assertEqual(res.shape, (8, 8))
        self.assertTrue(np.all(res == 76))

    def test_already_grayscale_input(self) -> None:
        """Verifica se uma imagem 2D já em escala de cinza é tratada corretamente."""
        gray_2d = np.array([[0, 50], [100, 200]], dtype=np.uint8)
        res = to_grayscale(gray_2d)
        np.testing.assert_array_equal(res, gray_2d)

        # Grayscale float [0.0, 1.0]
        gray_float = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
        res_float = to_grayscale(gray_float)
        self.assertEqual(res_float.dtype, np.uint8)
        self.assertEqual(res_float[0, 0], 0)
        self.assertEqual(res_float[1, 0], 255)

    def test_float_rgb_input_normalized(self) -> None:
        """Verifica se entrada float [0.0, 1.0] RGB é tratada corretamente."""
        rgb_float = np.zeros((4, 4, 3), dtype=np.float64)
        rgb_float[:, :, 0] = 1.0  # R = 1.0
        res = to_grayscale(rgb_float, GrayscaleMethod.LUMINANCE)
        self.assertEqual(res.dtype, np.uint8)
        self.assertTrue(np.all(res == 76))

    def test_invalid_dimension_raises_error(self) -> None:
        """Verifica se arrays 1D ou 4D geram ValueError."""
        with self.assertRaises(ValueError):
            to_grayscale(np.zeros((10,)))

        with self.assertRaises(ValueError):
            to_grayscale(np.zeros((2, 2, 2, 2)))

    def test_non_numpy_input_raises_error(self) -> None:
        """Verifica se entrada que não seja array NumPy gera ValueError."""
        with self.assertRaises(ValueError):
            to_grayscale([[255, 0, 0]])  # type: ignore

    def test_invalid_method_raises_error(self) -> None:
        """Verifica se método inválido gera ValueError."""
        with self.assertRaises(ValueError):
            to_grayscale(self.pure_red, method="INVALID_METHOD")  # type: ignore


class TestChannelIsolation(unittest.TestCase):
    """Suíte de testes para isolamento visual e cromático dos canais RGB."""

    def setUp(self) -> None:
        self.img_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        self.img_rgb[:, :, 0] = 200  # R
        self.img_rgb[:, :, 1] = 100  # G
        self.img_rgb[:, :, 2] = 50   # B

    def test_is_channel_isolation(self) -> None:
        from src.core.grayscale import is_channel_isolation
        self.assertTrue(is_channel_isolation(GrayscaleMethod.CHANNEL_R))
        self.assertTrue(is_channel_isolation(GrayscaleMethod.CHANNEL_G))
        self.assertTrue(is_channel_isolation(GrayscaleMethod.CHANNEL_B))
        self.assertFalse(is_channel_isolation(GrayscaleMethod.LUMINANCE))
        self.assertFalse(is_channel_isolation(GrayscaleMethod.AVERAGE))

    def test_isolate_channel_rgb_red(self) -> None:
        from src.core.grayscale import isolate_channel_rgb
        res = isolate_channel_rgb(self.img_rgb, GrayscaleMethod.CHANNEL_R)
        self.assertEqual(res.shape, (10, 10, 3))
        self.assertTrue(np.all(res[:, :, 0] == 200))
        self.assertTrue(np.all(res[:, :, 1] == 0))
        self.assertTrue(np.all(res[:, :, 2] == 0))

    def test_isolate_channel_rgb_green(self) -> None:
        from src.core.grayscale import isolate_channel_rgb
        res = isolate_channel_rgb(self.img_rgb, GrayscaleMethod.CHANNEL_G)
        self.assertEqual(res.shape, (10, 10, 3))
        self.assertTrue(np.all(res[:, :, 0] == 0))
        self.assertTrue(np.all(res[:, :, 1] == 100))
        self.assertTrue(np.all(res[:, :, 2] == 0))

    def test_isolate_channel_rgb_blue(self) -> None:
        from src.core.grayscale import isolate_channel_rgb
        res = isolate_channel_rgb(self.img_rgb, GrayscaleMethod.CHANNEL_B)
        self.assertEqual(res.shape, (10, 10, 3))
        self.assertTrue(np.all(res[:, :, 0] == 0))
        self.assertTrue(np.all(res[:, :, 1] == 0))
        self.assertTrue(np.all(res[:, :, 2] == 50))

    def test_colorize_channel(self) -> None:
        from src.core.grayscale import colorize_channel
        gray_2d = np.full((8, 8), 150, dtype=np.uint8)

        red_colored = colorize_channel(gray_2d, GrayscaleMethod.CHANNEL_R)
        self.assertEqual(red_colored.shape, (8, 8, 3))
        self.assertTrue(np.all(red_colored[:, :, 0] == 150))
        self.assertTrue(np.all(red_colored[:, :, 1] == 0))
        self.assertTrue(np.all(red_colored[:, :, 2] == 0))

        # Luminance should leave 2D array unchanged
        lum = colorize_channel(gray_2d, GrayscaleMethod.LUMINANCE)
        self.assertEqual(lum.shape, (8, 8))


if __name__ == "__main__":
    unittest.main()


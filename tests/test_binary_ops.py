"""
test_binary_ops.py — Testes unitários para o módulo de operações binárias (src.core.binary_ops).
"""

import unittest
import numpy as np

from src.core.binary_ops import (
    ResizeMode,
    add,
    align_images,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    blend,
    divide,
    multiply,
    subtract,
)


class TestAlignImages(unittest.TestCase):
    """Suíte de testes para alinhamento e compatibilização de imagens (ResizeMode)."""

    def setUp(self) -> None:
        self.gray_a = np.full((100, 100), 100, dtype=np.uint8)
        self.gray_b_same = np.full((100, 100), 200, dtype=np.uint8)
        self.gray_b_diff = np.full((60, 140), 150, dtype=np.uint8)

        self.rgb_a = np.zeros((100, 100, 3), dtype=np.uint8)
        self.rgb_b_diff = np.full((80, 120, 3), 180, dtype=np.uint8)

    def test_strict_mode_matching_shapes(self) -> None:
        """Modo STRICT deve ter sucesso quando shapes forem idênticos."""
        a, b = align_images(self.gray_a, self.gray_b_same, mode=ResizeMode.STRICT)
        self.assertEqual(a.shape, (100, 100))
        self.assertEqual(b.shape, (100, 100))
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(b.dtype, np.uint8)

    def test_strict_mode_divergent_shapes_raises_value_error(self) -> None:
        """Modo STRICT deve lançar ValueError explicativo quando dimensões divergirem."""
        with self.assertRaises(ValueError) as ctx:
            align_images(self.gray_a, self.gray_b_diff, mode=ResizeMode.STRICT)
        self.assertIn("STRICT", str(ctx.exception))
        self.assertIn("RESIZE_B_TO_A", str(ctx.exception))

    def test_strict_mode_divergent_channels_raises_value_error(self) -> None:
        """Modo STRICT deve lançar ValueError se canais divergirem (Grayscale x RGB)."""
        with self.assertRaises(ValueError):
            align_images(self.gray_a, self.rgb_a, mode=ResizeMode.STRICT)

    def test_resize_b_to_a_spatial(self) -> None:
        """Modo RESIZE_B_TO_A deve redimensionar B para exatamente o shape espacial de A."""
        a, b = align_images(self.gray_a, self.gray_b_diff, mode=ResizeMode.RESIZE_B_TO_A)
        self.assertEqual(a.shape, (100, 100))
        self.assertEqual(b.shape, (100, 100))

    def test_resize_b_to_a_channels_expansion(self) -> None:
        """Modo RESIZE_B_TO_A com A (RGB) e B (Grayscale) deve expandir B para 3 canais."""
        a, b = align_images(self.rgb_a, self.gray_b_diff, mode=ResizeMode.RESIZE_B_TO_A)
        self.assertEqual(a.shape, (100, 100, 3))
        self.assertEqual(b.shape, (100, 100, 3))

    def test_resize_b_to_a_channels_reduction(self) -> None:
        """Modo RESIZE_B_TO_A com A (Grayscale) e B (RGB) deve converter B para Grayscale."""
        a, b = align_images(self.gray_a, self.rgb_b_diff, mode=ResizeMode.RESIZE_B_TO_A)
        self.assertEqual(a.shape, (100, 100))
        self.assertEqual(b.shape, (100, 100))

    def test_crop_common_spatial(self) -> None:
        """Modo CROP_COMMON deve recortar ambas as imagens para min(H_a, H_b) e min(W_a, W_b)."""
        # A: (100, 100), B: (60, 140) -> corte comum: (60, 100)
        a, b = align_images(self.gray_a, self.gray_b_diff, mode=ResizeMode.CROP_COMMON)
        self.assertEqual(a.shape, (60, 100))
        self.assertEqual(b.shape, (60, 100))

    def test_rgba_alpha_strip(self) -> None:
        """Imagens RGBA devem ter o canal Alpha descartado automaticamente."""
        rgba = np.zeros((50, 50, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        target = np.zeros((50, 50, 3), dtype=np.uint8)
        a, b = align_images(target, rgba, mode=ResizeMode.STRICT)
        self.assertEqual(b.shape, (50, 50, 3))

    def test_float_input_conversion(self) -> None:
        """Imagens float [0.0, 1.0] devem ser convertidas para uint8 [0, 255]."""
        float_img = np.ones((50, 50), dtype=np.float32)
        target = np.zeros((50, 50), dtype=np.uint8)
        a, b = align_images(target, float_img, mode=ResizeMode.STRICT)
        self.assertEqual(b.dtype, np.uint8)
        self.assertEqual(b[0, 0], 255)


class TestArithmeticOperations(unittest.TestCase):
    """Suíte de testes para operações aritméticas em binary_ops."""

    def setUp(self) -> None:
        self.img_150 = np.full((10, 10), 150, dtype=np.uint8)
        self.img_120 = np.full((10, 10), 120, dtype=np.uint8)
        self.rgb_img = np.full((10, 10, 3), 100, dtype=np.uint8)

    def test_add_saturation(self) -> None:
        """Adição com clip=True deve saturar em 255."""
        res = add(self.img_150, self.img_120, clip=True)
        self.assertEqual(res.dtype, np.uint8)
        self.assertTrue(np.all(res == 255))

    def test_add_wrap_around(self) -> None:
        """Adição com clip=False deve aplicar wrap-around uint8 (150 + 120 = 270 -> 14)."""
        res = add(self.img_150, self.img_120, clip=False)
        self.assertEqual(res.dtype, np.uint8)
        self.assertTrue(np.all(res == 14))

    def test_add_scalar(self) -> None:
        """Adição de imagem com escalar numérico."""
        res = add(self.img_150, 50, clip=True)
        self.assertTrue(np.all(res == 200))

        res_sat = add(self.img_150, 120, clip=True)
        self.assertTrue(np.all(res_sat == 255))

    def test_subtract_normal(self) -> None:
        """Subtração com clip=True satura em 0 para resultados negativos."""
        res = subtract(self.img_120, self.img_150, absolute=False, clip=True)
        self.assertTrue(np.all(res == 0))

        res_pos = subtract(self.img_150, self.img_120, absolute=False, clip=True)
        self.assertTrue(np.all(res_pos == 30))

    def test_subtract_absolute(self) -> None:
        """Subtração com absolute=True deve retornar magnitude |A - B|."""
        res = subtract(self.img_120, self.img_150, absolute=True)
        self.assertTrue(np.all(res == 30))

    def test_subtract_scalar(self) -> None:
        """Subtração de imagem com escalar."""
        res = subtract(self.img_150, 50)
        self.assertTrue(np.all(res == 100))

        res_neg = subtract(self.img_150, 200, absolute=False, clip=True)
        self.assertTrue(np.all(res_neg == 0))

    def test_blend(self) -> None:
        """Mistura Alpha Blending (alpha * A + (1 - alpha) * B)."""
        img_0 = np.full((10, 10), 0, dtype=np.uint8)
        img_200 = np.full((10, 10), 200, dtype=np.uint8)

        # alpha = 0.5 -> 0.5*0 + 0.5*200 = 100
        res = blend(img_0, img_200, alpha=0.5)
        self.assertTrue(np.all(res == 100))

        # alpha = 1.0 -> img_0 (0)
        res_1 = blend(img_0, img_200, alpha=1.0)
        self.assertTrue(np.all(res_1 == 0))

        # alpha = 0.0 -> img_200 (200)
        res_0 = blend(img_0, img_200, alpha=0.0)
        self.assertTrue(np.all(res_0 == 200))

    def test_blend_invalid_alpha(self) -> None:
        """Blend deve lançar ValueError para alpha fora de [0, 1]."""
        with self.assertRaises(ValueError):
            blend(self.img_150, self.img_120, alpha=1.5)
        with self.assertRaises(ValueError):
            blend(self.img_150, self.img_120, alpha=-0.1)

    def test_multiply_scalar(self) -> None:
        """Multiplicação por escalar com saturação em 255."""
        img = np.full((5, 5), 100, dtype=np.uint8)
        res = multiply(img, 1.5, clip=True)
        self.assertTrue(np.all(res == 150))

        res_sat = multiply(img, 3.0, clip=True)
        self.assertTrue(np.all(res_sat == 255))

    def test_multiply_mask(self) -> None:
        """Multiplicação por máscara binária (0 e 1)."""
        img = np.full((4, 4), 200, dtype=np.uint8)
        mask = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        res = multiply(img, mask)
        expected = mask * 200
        np.testing.assert_array_equal(res, expected)

    def test_divide_scalar(self) -> None:
        """Divisão por escalar numérico."""
        img = np.full((4, 4), 200, dtype=np.uint8)
        res = divide(img, 2.0)
        self.assertTrue(np.all(res == 100))

    def test_divide_by_zero_prevention(self) -> None:
        """Divisão por zeros não deve levantar exceção devido à proteção por eps."""
        img = np.full((4, 4), 200, dtype=np.uint8)
        zeros = np.zeros((4, 4), dtype=np.uint8)
        res = divide(img, zeros, eps=1e-5, clip=True)
        self.assertEqual(res.dtype, np.uint8)
        # Valores muito altos saturam em 255
        self.assertTrue(np.all(res == 255))

    def test_rgb_support(self) -> None:
        """Operações devem funcionar perfeitamente com 3 canais RGB."""
        img_rgb_a = np.full((10, 10, 3), 100, dtype=np.uint8)
        img_rgb_b = np.full((10, 10, 3), 80, dtype=np.uint8)

        res_add = add(img_rgb_a, img_rgb_b)
        self.assertEqual(res_add.shape, (10, 10, 3))
        self.assertTrue(np.all(res_add == 180))

        res_sub = subtract(img_rgb_a, img_rgb_b)
        self.assertEqual(res_sub.shape, (10, 10, 3))
        self.assertTrue(np.all(res_sub == 20))


class TestLogicalOperations(unittest.TestCase):
    """Suíte de testes para operações lógicas bitwise em binary_ops."""

    def setUp(self) -> None:
        self.img_a = np.array([[0b11110000, 0b10101010]], dtype=np.uint8)
        self.img_b = np.array([[0b11001100, 0b01010101]], dtype=np.uint8)

    def test_bitwise_and(self) -> None:
        """Bitwise AND entre imagens e com máscara escalar."""
        # 11110000 & 11001100 = 11000000 (192)
        # 10101010 & 01010101 = 00000000 (0)
        res = bitwise_and(self.img_a, self.img_b)
        self.assertEqual(res[0, 0], 192)
        self.assertEqual(res[0, 1], 0)

        # Com escalar: 0b11110000 & 0b00001111 = 0
        res_scalar = bitwise_and(self.img_a, 0b00001111)
        self.assertEqual(res_scalar[0, 0], 0)

    def test_bitwise_or(self) -> None:
        """Bitwise OR entre imagens."""
        # 11110000 | 11001100 = 11111100 (252)
        # 10101010 | 01010101 = 11111111 (255)
        res = bitwise_or(self.img_a, self.img_b)
        self.assertEqual(res[0, 0], 252)
        self.assertEqual(res[0, 1], 255)

    def test_bitwise_xor(self) -> None:
        """Bitwise XOR entre imagens."""
        # 11110000 ^ 11001100 = 00111100 (60)
        # 10101010 ^ 01010101 = 11111111 (255)
        res = bitwise_xor(self.img_a, self.img_b)
        self.assertEqual(res[0, 0], 60)
        self.assertEqual(res[0, 1], 255)

        # XOR com a própria imagem zera todos os bits
        res_self = bitwise_xor(self.img_a, self.img_a)
        self.assertTrue(np.all(res_self == 0))

    def test_bitwise_not(self) -> None:
        """Bitwise NOT (inversão de todos os bits / negativo digital)."""
        # ~11110000 (240) = 00001111 (15)
        # ~10101010 (170) = 01010101 (85)
        res = bitwise_not(self.img_a)
        self.assertEqual(res[0, 0], 15)
        self.assertEqual(res[0, 1], 85)


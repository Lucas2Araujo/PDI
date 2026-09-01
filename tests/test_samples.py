"""
test_samples.py — Testes unitários do módulo de imagens de exemplo embutidas.
"""

import unittest
import numpy as np

from src.core.samples import (
    SAMPLE_AYLA_NAME,
    SAMPLE_BENCHMARK_NAME,
    SAMPLE_LENA_NAME,
    SAMPLE_OPTIONS,
    SAMPLE_PENTAGONO_NAME,
    SAMPLE_PORTRAIT_NAME,
    get_sample_path,
    load_sample_array,
    load_sample_bytes,
)


class TestSamples(unittest.TestCase):
    """Suíte de testes para as imagens de exemplo embutidas no aplicativo."""

    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

    def test_sample_options_list(self) -> None:
        self.assertEqual(len(SAMPLE_OPTIONS), 5)
        names = [s["name"] for s in SAMPLE_OPTIONS]
        self.assertIn(SAMPLE_PORTRAIT_NAME, names)
        self.assertIn(SAMPLE_BENCHMARK_NAME, names)
        self.assertIn(SAMPLE_LENA_NAME, names)
        self.assertIn(SAMPLE_AYLA_NAME, names)
        self.assertIn(SAMPLE_PENTAGONO_NAME, names)

    def test_get_sample_path(self) -> None:
        for opt in SAMPLE_OPTIONS:
            p = get_sample_path(opt["name"])
            self.assertTrue(p.exists(), f"Path não existe: {p}")

    def test_get_invalid_sample_path_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            get_sample_path("non_existent_image.png")

    def test_load_sample_array(self) -> None:
        for opt in SAMPLE_OPTIONS:
            arr = load_sample_array(opt["name"])
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.dtype, np.uint8)
            self.assertTrue(arr.ndim in (2, 3))

        # Testes específicos
        arr_portrait = load_sample_array(SAMPLE_PORTRAIT_NAME)
        self.assertEqual(arr_portrait.shape, (512, 512, 3))

        arr_lena = load_sample_array(SAMPLE_LENA_NAME)
        self.assertEqual(arr_lena.shape, (512, 512, 3))

        arr_ayla = load_sample_array(SAMPLE_AYLA_NAME)
        self.assertEqual(arr_ayla.shape, (1600, 1494, 3))

        arr_pent = load_sample_array(SAMPLE_PENTAGONO_NAME)
        self.assertEqual(arr_pent.shape, (1024, 1024))

    def test_load_sample_bytes(self) -> None:
        for opt in SAMPLE_OPTIONS:
            b = load_sample_bytes(opt["name"])
            self.assertTrue(b.startswith(self.PNG_MAGIC), f"Header PNG inválido para {opt['name']}")


if __name__ == "__main__":
    unittest.main()

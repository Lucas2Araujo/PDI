"""
test_batch.py — Testes unitários do motor de processamento em lote.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.batch import (
    BatchResult,
    discover_images,
    process_batch,
    process_bytes_list,
)
from src.core.grayscale import GrayscaleMethod
from src.core.quantization import QuantizationTechnique


class TestDiscoverImages(unittest.TestCase):
    """Suíte de testes para a detecção de imagens em diretório."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Cria arquivos com extensões variadas
        (self.root / "img1.png").touch()
        (self.root / "img2.jpg").touch()
        (self.root / "img3.WEBP").touch()
        (self.root / "document.txt").touch()
        (self.root / "script.py").touch()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_discover_supported_extensions(self) -> None:
        found = discover_images(self.root)
        names = [f.name for f in found]
        self.assertEqual(len(found), 3)
        self.assertIn("img1.png", names)
        self.assertIn("img2.jpg", names)
        self.assertIn("img3.WEBP", names)
        self.assertNotIn("document.txt", names)
        self.assertNotIn("script.py", names)

    def test_invalid_directory_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            discover_images(self.root / "non_existent_dir")


class TestProcessBatch(unittest.TestCase):
    """Suíte de testes para execução de processamento em lote."""

    def setUp(self) -> None:
        self.temp_in = tempfile.TemporaryDirectory()
        self.temp_out = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_in.name)
        self.output_dir = Path(self.temp_out.name)

        # Salva 2 imagens válidas
        img_arr = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(self.input_dir / "valid1.png")
        Image.fromarray(img_arr).save(self.input_dir / "valid2.jpg")

        # Cria 1 arquivo corrompido com extensão válida
        (self.input_dir / "corrupted.png").write_bytes(b"NOT_A_VALID_IMAGE_BYTES")

    def tearDown(self) -> None:
        self.temp_in.cleanup()
        self.temp_out.cleanup()

    def test_process_batch_execution_and_metrics(self) -> None:
        progress_calls = []

        def _callback(curr: int, total: int, filename: str) -> None:
            progress_calls.append((curr, total, filename))

        result = process_batch(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            technique=QuantizationTechnique.UNIFORM,
            bits=4,
            grayscale_method=GrayscaleMethod.LUMINANCE,
            progress_callback=_callback,
        )

        self.assertIsInstance(result, BatchResult)
        self.assertEqual(result.total, 3)
        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.failure_count, 1)
        self.assertEqual(len(progress_calls), 3)

        # Verifica se os arquivos de saída foram criados
        outputs = list(self.output_dir.glob("*.png"))
        self.assertEqual(len(outputs), 2)


class TestProcessBytesList(unittest.TestCase):
    """Suíte de testes para processamento em lote puramente em memória (Web mode)."""

    def test_process_bytes_list_success_and_failures(self) -> None:
        import io

        # Cria imagem válida em memória
        img_arr = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        valid_bytes = buf.getvalue()

        # Lista de teste com 1 válida e 1 inválida
        items = [
            ("foto1.png", valid_bytes),
            ("invalido.png", b"BYTES_INVALIDOS"),
        ]

        progress_calls = []

        def _callback(curr: int, total: int, filename: str) -> None:
            progress_calls.append((curr, total, filename))

        results, failures = process_bytes_list(
            images=items,
            technique=QuantizationTechnique.UNIFORM,
            bits=4,
            grayscale_method=GrayscaleMethod.LUMINANCE,
            progress_callback=_callback,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(progress_calls), 2)
        self.assertTrue(results[0][0].endswith(".png"))
        self.assertGreater(len(results[0][1]), 0)
        self.assertEqual(failures[0][0], "invalido.png")

    def test_process_bytes_batch_rich_results(self) -> None:
        import io
        from src.core.batch import process_bytes_batch, BatchItemResult

        img_arr = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        valid_bytes = buf.getvalue()

        items = [
            ("amostra.png", valid_bytes),
            ("corrompida.png", b"CORROMPIDO"),
        ]

        batch_res = process_bytes_batch(
            images=items,
            technique=QuantizationTechnique.UNIFORM,
            bits=4,
            grayscale_method=GrayscaleMethod.LUMINANCE,
        )

        self.assertEqual(batch_res.total, 2)
        self.assertEqual(batch_res.success_count, 1)
        self.assertEqual(batch_res.failure_count, 1)
        self.assertEqual(len(batch_res.items), 2)
        self.assertIsInstance(batch_res.items[0], BatchItemResult)
        self.assertTrue(batch_res.items[0].success)
        self.assertIsNotNone(batch_res.items[0].metrics)
        self.assertGreater(batch_res.items[0].metrics.psnr, 0.0)
        self.assertFalse(batch_res.items[1].success)
        self.assertGreater(batch_res.avg_psnr, 0.0)
        self.assertEqual(batch_res.avg_savings_pct, 50.0)
        self.assertIsNotNone(batch_res.items[0].source_thumb_bytes)
        self.assertIsNotNone(batch_res.items[0].quantized_thumb_bytes)

    def test_make_thumbnail_png(self) -> None:
        from src.core.batch import make_thumbnail_png
        arr_rgb = np.random.randint(0, 256, (500, 400, 3), dtype=np.uint8)
        thumb = make_thumbnail_png(arr_rgb, max_size=100)
        self.assertIsInstance(thumb, bytes)
        self.assertGreater(len(thumb), 0)

        # Verifica dimensões da miniatura gerada
        import io
        pil_thumb = Image.open(io.BytesIO(thumb))
        self.assertLessEqual(max(pil_thumb.size), 100)


if __name__ == "__main__":
    unittest.main()


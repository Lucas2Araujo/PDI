"""
test_batch.py — Testes unitários do motor de processamento em lote.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.batch import BatchResult, discover_images, process_batch
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


if __name__ == "__main__":
    unittest.main()

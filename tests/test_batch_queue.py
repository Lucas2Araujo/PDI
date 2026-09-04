"""
test_batch_queue.py — Testes unitários para o componente BatchQueue (src.ui.components.batch_queue).
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.ui import theme
from src.ui.components.batch_queue import BatchQueue, BatchQueueItem


class TestBatchQueue(unittest.TestCase):
    """Suíte de testes para BatchQueue e BatchQueueItem."""

    def setUp(self) -> None:
        self.sample_array = np.full((50, 50, 3), 150, dtype=np.uint8)

    def test_batch_queue_item_inspection(self) -> None:
        """BatchQueueItem deve inspecionar e extrair dimensões, canais e thumbnail."""
        item = BatchQueueItem(name="teste.png", array=self.sample_array)
        self.assertEqual(item.name, "teste.png")
        self.assertEqual(item.dimensions, "50×50 px")
        self.assertIn("RGB", item.color_type)
        self.assertIsNotNone(item.thumb_bytes)

        full_png = item.get_full_png_bytes()
        self.assertIsInstance(full_png, bytes)

    def test_queue_initial_state(self) -> None:
        """BatchQueue inicializa invisível e com lista de itens vazia."""
        queue = BatchQueue()
        self.assertEqual(queue.items, [])
        self.assertFalse(queue.visible)

    def test_add_and_set_items(self) -> None:
        """add_item e set_items devem popular a fila e torná-la visível."""
        queue = BatchQueue()
        item1 = BatchQueueItem(name="img1.png", array=self.sample_array)
        item2 = BatchQueueItem(name="img2.png", array=self.sample_array)

        queue.add_item(item1)
        self.assertEqual(len(queue.items), 1)
        self.assertTrue(queue.visible)

        queue.set_items([item1, item2])
        self.assertEqual(len(queue.items), 2)
        self.assertEqual(len(queue._grid_container.controls), 2)

    def test_update_item_status(self) -> None:
        """update_item_status deve modificar o status e a cor do item."""
        queue = BatchQueue()
        item = BatchQueueItem(name="img.png", array=self.sample_array)
        queue.add_item(item)

        queue.update_item_status(0, "✅ Concluído", color=theme.SUCCESS)
        self.assertEqual(queue.items[0].status, "✅ Concluído")
        self.assertEqual(queue.items[0].status_color, theme.SUCCESS)

    def test_update_progress(self) -> None:
        """update_progress deve atualizar o valor da barra de progresso."""
        queue = BatchQueue()
        queue.update_progress(current=4, total=10)
        self.assertTrue(queue._progress_bar.visible)
        self.assertAlmostEqual(queue._progress_bar.value, 0.4)

        queue.update_progress(current=0, total=0)
        self.assertFalse(queue._progress_bar.visible)

    def test_clear_queue(self) -> None:
        """clear deve esvaziar os itens, ocultar a fila e acionar callback on_clear."""
        mock_clear = MagicMock()
        queue = BatchQueue(on_clear=mock_clear)
        queue.add_item(BatchQueueItem(name="img.png", array=self.sample_array))

        queue.clear()
        self.assertEqual(queue.items, [])
        self.assertFalse(queue.visible)
        mock_clear.assert_called_once()

    def test_responsive_cards_no_rigid_width(self) -> None:
        """Cards da fila não devem ter largura fixa rígida maior que 300px."""
        queue = BatchQueue()
        queue.add_item(BatchQueueItem(name="img.png", array=self.sample_array))

        card = queue._grid_container.controls[0]
        card_width = getattr(card, "width", None)
        self.assertNotEqual(card_width, 260)  # Eliminado width=260 rígido
        if card_width is not None:
            self.assertLessEqual(card_width, 300)


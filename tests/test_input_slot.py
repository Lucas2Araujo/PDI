"""
test_input_slot.py — Testes unitários para o componente InputSlot (src.ui.components.input_slot).
"""

import unittest
from unittest.mock import MagicMock
import flet as ft
import numpy as np

from src.ui.components.input_slot import InputSlot
from src.ui.state.session_state import SessionState


class TestInputSlot(unittest.TestCase):
    """Suíte de testes para o componente InputSlot."""

    def setUp(self) -> None:
        self.state = SessionState()
        self.sample_img = np.full((50, 50), 128, dtype=np.uint8)
        self.mock_page = MagicMock(spec=ft.Page)

    def test_initial_empty_state(self) -> None:
        """Verifica inicialização do slot em estado vazio."""
        slot = InputSlot(slot_id="A", label="Slot Primário", session_state=self.state)
        self.assertEqual(slot.slot_id, "A")
        self.assertEqual(slot.label, "Slot Primário")
        self.assertIsNone(slot.image_array)
        self.assertEqual(slot.image_name, "")
        self.assertFalse(slot.is_scalar_mode)

    def test_set_image_and_session_sync(self) -> None:
        """set_image deve atualizar o slot e sincronizar com o SessionState."""
        slot = InputSlot(slot_id="A", label="Slot A", session_state=self.state)
        slot.set_image(self.sample_img, "teste.png", sync_session=True)

        self.assertIs(slot.image_array, self.sample_img)
        self.assertEqual(slot.image_name, "teste.png")
        self.assertIs(self.state.image_a, self.sample_img)
        self.assertEqual(self.state.image_a_name, "teste.png")

    def test_slot_b_sync(self) -> None:
        """Slot B deve sincronizar com image_b do SessionState."""
        slot_b = InputSlot(slot_id="B", label="Slot B", session_state=self.state)
        slot_b.set_image(self.sample_img, "mascara.png", sync_session=True)

        self.assertIs(self.state.image_b, self.sample_img)
        self.assertEqual(self.state.image_b_name, "mascara.png")

    def test_clear_slot(self) -> None:
        """clear() deve anular a imagem no slot e no SessionState."""
        slot = InputSlot(slot_id="A", session_state=self.state)
        slot.set_image(self.sample_img, "foto.png")
        slot.clear(sync_session=True)

        self.assertIsNone(slot.image_array)
        self.assertIsNone(self.state.image_a)

    def test_external_session_change_updates_slot(self) -> None:
        """Modificação externa no SessionState deve atualizar a visualização do InputSlot."""
        slot = InputSlot(slot_id="A", session_state=self.state)
        new_img = np.full((20, 20), 200, dtype=np.uint8)

        self.state.set_image_a(new_img, "externa.png")
        self.assertIs(slot.image_array, new_img)
        self.assertEqual(slot.image_name, "externa.png")

    def test_scalar_mode_toggle(self) -> None:
        """Valida alternância para modo escalar numérico."""
        slot = InputSlot(slot_id="B", session_state=self.state, supports_scalar=True)
        self.assertFalse(slot.is_scalar_mode)

        slot.set_scalar_mode(True, default_val=75.0)
        self.assertTrue(slot.is_scalar_mode)
        self.assertEqual(slot.scalar_value, 75.0)

    def test_sample_selection(self) -> None:
        """_on_select_sample deve carregar com sucesso uma das amostras embutidas."""
        slot = InputSlot(slot_id="A", session_state=self.state)
        slot._on_select_sample("portrait")

        self.assertIsNotNone(slot.image_array)
        self.assertEqual(slot.image_array.ndim, 3)
        self.assertEqual(slot.image_array.shape[:2], (512, 512))
        self.assertIs(self.state.image_a, slot.image_array)

    def test_save_loaded_as_grayscale(self) -> None:
        """Verifica a ação rápida de salvar a imagem carregada em tons de cinza de 8 bits."""
        slot = InputSlot(slot_id="A", session_state=self.state)
        color_img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        slot.set_image(color_img, "foto_colorida.png")

        # Chama a ação rápida de salvar como tons de cinza
        # Não deve lançar exceção mesmo sem página Flet ativa
        slot._save_loaded_as_grayscale()

        # Slot já em cinza 2D também deve ser aceito
        gray_img = np.full((30, 30), 100, dtype=np.uint8)
        slot.set_image(gray_img, "foto_cinza.png")
        slot._save_loaded_as_grayscale()

    def test_loaded_state_card_structure(self) -> None:
        """Garante que o card com imagem carregada possui estrutura com ações e botão sem overflow."""
        slot = InputSlot(slot_id="A", session_state=self.state)
        color_img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        slot.set_image(color_img, "teste.png")

        # Verifica que o container de conteúdo possui o card montado
        self.assertGreater(len(slot._content_column.controls), 1)
        loaded_card = slot._content_column.controls[1]
        self.assertIsInstance(loaded_card, ft.Container)
        self.assertIsInstance(loaded_card.content, ft.Column)
        # Deve ter a linha superior (thumb + info + actions) e o botão inferior
        self.assertEqual(len(loaded_card.content.controls), 2)
        bottom_btn = loaded_card.content.controls[1]
        self.assertIsInstance(bottom_btn, ft.Button)




"""
Pacote src.ui.state — Gerenciamento reativo de estado da aplicação.
"""

from src.ui.state.session_state import (
    EVENT_IMAGE_A_CHANGED,
    EVENT_IMAGE_B_CHANGED,
    EVENT_MODE_TOGGLED,
    EVENT_PROCESSING_STATE,
    EVENT_RESULT_CHANGED,
    SUPPORTED_EVENTS,
    SessionState,
    get_session_state,
    reset_session_state,
)

__all__ = [
    "SessionState",
    "get_session_state",
    "reset_session_state",
    "EVENT_IMAGE_A_CHANGED",
    "EVENT_IMAGE_B_CHANGED",
    "EVENT_RESULT_CHANGED",
    "EVENT_PROCESSING_STATE",
    "EVENT_MODE_TOGGLED",
    "SUPPORTED_EVENTS",
]


"""
session_state.py — Gerenciador de Estado Reativo e Ciclo de Vida de Memória.

Centraliza dados ativos da aplicação PDI (imagens em memória, métricas, estado de processamento),
fornecendo um barramento de eventos (Pub/Sub leve) para desacoplamento da interface e
gestão explícita de memória com gc.collect() para execução estável no WebAssembly (Pyodide).
"""

from collections import defaultdict
import gc
import logging
from typing import Any, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Eventos canônicos suportados pelo barramento de eventos
EVENT_IMAGE_A_CHANGED = "image_a_changed"
EVENT_IMAGE_B_CHANGED = "image_b_changed"
EVENT_RESULT_CHANGED = "result_changed"
EVENT_PROCESSING_STATE = "processing_state"
EVENT_MODE_TOGGLED = "mode_toggled"

SUPPORTED_EVENTS = frozenset({
    EVENT_IMAGE_A_CHANGED,
    EVENT_IMAGE_B_CHANGED,
    EVENT_RESULT_CHANGED,
    EVENT_PROCESSING_STATE,
    EVENT_MODE_TOGGLED,
})

# Limite máximo de estados armazenados para proteção de memória em WebAssembly (Pyodide)
MAX_HISTORY_STEPS: int = 5


class SessionState:
    """
    Gerenciador reativo de estado de sessão do aplicativo PDI.

    Mantém referências às imagens ativas nos slots A, B e Resultado,
    métricas de qualidade e status de processamento, provendo notificações
    a observadores registrados e coleta forçada de lixo para WebAssembly.
    """

    def __init__(self) -> None:
        self.image_a: np.ndarray | None = None
        self.image_b: np.ndarray | None = None
        self.result_image: np.ndarray | None = None

        self.image_a_name: str = ""
        self.image_b_name: str = ""
        self.result_name: str = ""
        self.last_applied_module_name: str = ""

        # Pilhas de histórico para composição e Undo/Redo (Pipeline)
        self._history: list[tuple[np.ndarray | None, str]] = []
        self._future: list[tuple[np.ndarray | None, str]] = []

        self.metrics: dict[str, Any] = {}
        self.is_processing: bool = False
        self.is_batch_mode: bool = False
        self.batch_queue: list[Any] = []
        self.batch_results: Any | None = None

        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    @property
    def history_count(self) -> int:
        """Retorna o número de passos acumulados no histórico de composição."""
        return len(self._history)

    # ---------------------------------------------------------------------------
    # Sistema de Observadores (Pub/Sub leve)
    # ---------------------------------------------------------------------------

    def subscribe(self, event_name: str, callback: Callable[..., Any]) -> Callable[[], None]:
        """
        Inscreve um callback para ser executado quando o evento especificado ocorrer.

        Args:
            event_name: Nome do evento a observar (ex: "image_a_changed").
            callback: Função executada ao disparar o evento.

        Returns:
            Função sem argumentos que desinscreve o callback (unsubscribe pattern).
        """
        if callback not in self._subscribers[event_name]:
            self._subscribers[event_name].append(callback)

        def _unsubscribe() -> None:
            self.unsubscribe(event_name, callback)

        return _unsubscribe

    def unsubscribe(self, event_name: str, callback: Callable[..., Any]) -> None:
        """
        Remove a inscrição de um callback para determinado evento.

        Args:
            event_name: Nome do evento registrado.
            callback: Função callback a ser removida.
        """
        if event_name in self._subscribers and callback in self._subscribers[event_name]:
            self._subscribers[event_name].remove(callback)

    def notify(self, event_name: str, **kwargs: Any) -> None:
        """
        Dispara um evento para todos os callbacks inscritos.

        Falhas individuais de callbacks são capturadas e registradas em log
        para impedir que um erro em um componente interrompa os demais observadores.

        Args:
            event_name: Nome do evento a disparar.
            **kwargs: Parâmetros contextuais repassados aos callbacks.
        """
        subscribers = list(self._subscribers.get(event_name, []))
        for cb in subscribers:
            try:
                cb(**kwargs)
            except TypeError:
                # Tenta chamar sem argumentos se o callback não aceitar kwargs
                try:
                    cb()
                except Exception as ex:
                    logger.exception("Erro ao invocar observer sem argumentos para evento %s: %s", event_name, ex)
            except Exception as ex:
                logger.exception("Erro ao invocar observer com kwargs para evento %s: %s", event_name, ex)

    # ---------------------------------------------------------------------------
    # Setters Reativos com Notificação Automática
    # ---------------------------------------------------------------------------

    def set_image_a(self, image: np.ndarray | None, name: str = "") -> None:
        """Define a imagem do Slot A e notifica observadores."""
        self.image_a = image
        self.image_a_name = name
        self.notify(EVENT_IMAGE_A_CHANGED, image=image, name=name)

    def set_image_b(self, image: np.ndarray | None, name: str = "") -> None:
        """Define a imagem do Slot B e notifica observadores."""
        self.image_b = image
        self.image_b_name = name
        self.notify(EVENT_IMAGE_B_CHANGED, image=image, name=name)

    def set_result(
        self,
        image: np.ndarray | None,
        metrics: dict[str, Any] | None = None,
        name: str = "resultado.png",
        module_name: str = "",
    ) -> None:
        """Define a imagem resultante, atualiza métricas e notifica observadores."""
        self.result_image = image
        self.result_name = name
        self.metrics = dict(metrics) if metrics is not None else {}
        if module_name:
            self.last_applied_module_name = module_name
        self.notify(EVENT_RESULT_CHANGED, image=image, metrics=self.metrics)

    def set_processing(self, is_processing: bool) -> None:
        """Define o estado de processamento ativo e notifica observadores."""
        self.is_processing = bool(is_processing)
        self.notify(EVENT_PROCESSING_STATE, is_processing=self.is_processing)

    def set_batch_mode(self, is_batch: bool) -> None:
        """Define o modo de lote (True) ou individual (False) e notifica observadores."""
        self.is_batch_mode = bool(is_batch)
        self.notify(EVENT_MODE_TOGGLED, is_batch_mode=self.is_batch_mode)

    def toggle_batch_mode(self) -> bool:
        """Alterna entre modo individual e modo em lote, notificando observadores."""
        self.set_batch_mode(not self.is_batch_mode)
        return self.is_batch_mode

    # ---------------------------------------------------------------------------
    # Pipeline de Composição e Histórico de Operações (Undo/Redo)
    # ---------------------------------------------------------------------------

    def promote_result_to_input_a(self) -> bool:
        """
        Promove a imagem resultante para Entrada A, salvando o estado atual no histórico
        para permitir o encadeamento sequencial de transformações e navegação de histórico.

        Returns:
            True se a promoção ocorreu com sucesso, False se result_image for None.
        """
        if self.result_image is None:
            return False

        # Garante respeito ao limite de passos para proteção de memória WebAssembly
        while len(self._history) >= MAX_HISTORY_STEPS:
            old_item = self._history.pop(0)
            del old_item
            gc.collect()

        # Salva o estado atual na pilha de histórico
        self._history.append((self.image_a, self.image_a_name))

        # Novo ramo de ações invalida a pilha de refazer
        if self._future:
            self._future.clear()
            gc.collect()

        # Promove a imagem resultante como nova entrada A
        self.image_a = self.result_image.copy()

        # Atualiza o rótulo de pipeline identificando etapas encadeadas
        mod_name = self.last_applied_module_name or "Composição"
        if self.image_a_name:
            self.image_a_name = f"{self.image_a_name} + {mod_name}"
        else:
            self.image_a_name = f"Etapa ({mod_name})"

        # Limpa o resultado ativo (marcado como consumido)
        self.result_image = None
        self.result_name = ""
        self.metrics.clear()

        # Notifica observadores da alteração em Entrada A e no Resultado
        self.notify(EVENT_IMAGE_A_CHANGED, image=self.image_a, name=self.image_a_name)
        self.notify(EVENT_RESULT_CHANGED, image=None, metrics={})
        return True

    def can_undo(self) -> bool:
        """Indica se há estados anteriores disponíveis para desfazer."""
        return len(self._history) > 0

    def can_redo(self) -> bool:
        """Indica se há estados futuros disponíveis para refazer."""
        return len(self._future) > 0

    def undo(self) -> bool:
        """
        Desfaz o último passo de composição, restaurando o estado anterior de Entrada A.
        Move o estado atual para a pilha de refazer.

        Returns:
            True se a operação foi desfeita com sucesso, False se não houver histórico.
        """
        if not self.can_undo():
            return False

        # Salva o estado atual na pilha de refazer
        self._future.append((self.image_a, self.image_a_name))

        # Restaura o último estado da pilha de histórico
        prev_image, prev_name = self._history.pop()
        self.image_a = prev_image
        self.image_a_name = prev_name

        self.notify(EVENT_IMAGE_A_CHANGED, image=self.image_a, name=self.image_a_name)
        gc.collect()
        return True

    def redo(self) -> bool:
        """
        Refaz o passo desfeito, restaurando o estado posterior de Entrada A.
        Move o estado atual de volta para a pilha de histórico.

        Returns:
            True se a operação foi refeita com sucesso, False se não houver passos futuros.
        """
        if not self.can_redo():
            return False

        # Respeita o teto de passos ao reincorporar ao histórico
        while len(self._history) >= MAX_HISTORY_STEPS:
            old_item = self._history.pop(0)
            del old_item
            gc.collect()

        self._history.append((self.image_a, self.image_a_name))

        # Restaura da pilha futura
        next_image, next_name = self._future.pop()
        self.image_a = next_image
        self.image_a_name = next_name

        self.notify(EVENT_IMAGE_A_CHANGED, image=self.image_a, name=self.image_a_name)
        return True

    def clear_history(self) -> None:
        """Descarta todo o histórico de desfazer/refazer e coleta lixo de memória."""
        self._history.clear()
        self._future.clear()
        gc.collect()

    # ---------------------------------------------------------------------------
    # Gestão de Memória (WebAssembly / Heap Cleanup)
    # ---------------------------------------------------------------------------

    def clear_result(self) -> None:
        """
        Descarta a imagem resultante e métricas, forçando coleta de lixo.
        Essencial para liberar buffers no heap linear do WebAssembly.
        """
        self.result_image = None
        self.result_name = ""
        self.metrics.clear()
        gc.collect()
        self.notify(EVENT_RESULT_CHANGED, image=None, metrics={})

    def clear_all(self) -> None:
        """
        Descarta todas as imagens ativas (A, B e Resultado) e filas de lote,
        invocando gc.collect() para reciclagem integral de memória.
        """
        self.image_a = None
        self.image_b = None
        self.result_image = None

        self.image_a_name = ""
        self.image_b_name = ""
        self.result_name = ""
        self.last_applied_module_name = ""

        self._history.clear()
        self._future.clear()

        self.metrics.clear()
        self.batch_queue.clear()
        self.batch_results = None
        self.is_processing = False

        gc.collect()

        self.notify(EVENT_IMAGE_A_CHANGED, image=None, name="")
        self.notify(EVENT_IMAGE_B_CHANGED, image=None, name="")
        self.notify(EVENT_RESULT_CHANGED, image=None, metrics={})
        self.notify(EVENT_PROCESSING_STATE, is_processing=False)


# ---------------------------------------------------------------------------
# Instância Global de Sessão (Singleton)
# ---------------------------------------------------------------------------

_global_session_state: SessionState | None = None


def get_session_state() -> SessionState:
    """Retorna a instância singleton compartilhada de SessionState."""
    global _global_session_state
    if _global_session_state is None:
        _global_session_state = SessionState()
    return _global_session_state


def reset_session_state() -> SessionState:
    """
    Reinicializa a instância global de SessionState.
    Útil em testes unitários e reinicializações de aplicação.
    """
    global _global_session_state
    if _global_session_state is not None:
        _global_session_state.clear_all()
    _global_session_state = SessionState()
    return _global_session_state


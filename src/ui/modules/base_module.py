"""
base_module.py — Contrato Base e Interface Abstrata para Módulos Didáticos da Ementa.

Define a classe base `BasePDIModule` para componentes modulares do aplicativo PDI.
Cada módulo encapsula:
  - Metadados didáticos (título, descrição, requisitos de entrada).
  - Construção de controles reativos específicos (sliders, dropdowns, switches).
  - Execução pura e síncrona do algoritmo de processamento de imagem com retorno de métricas.
  - Design 100% responsivo para visualização fluida em dispositivos móveis e desktop.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import flet as ft
import numpy as np

from src.ui import theme


class BasePDIModule(ft.Column, ABC):
    """
    Componente visual base e contrato para módulos de processamento digital de imagens.

    Subclasses devem definir os metadados de classe e implementar `build_controls()`
    e `process()`. O componente gerencia automaticamente seu layout responsivo,
    cabeçalho explicativo e integração com o ciclo de reatividade.
    """

    # Metadados que devem ser declarados pelas subclasses
    title: str = "Módulo PDI"
    description: str = "Descrição do algoritmo e relevância didática."
    requires_second_input: bool = False
    supports_scalar_mode: bool = False

    def __init__(
        self,
        on_param_changed: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o módulo PDI.

        Args:
            on_param_changed: Callback acionado quando algum parâmetro dos controles é alterado,
                              permitindo re-execução ou preview reativo pela View controladora.
            **kwargs: Parâmetros adicionais repassados para ft.Column.
        """
        on_param_changed = kwargs.pop("on_change", on_param_changed)
        # Configuração de layout responsivo padrão: sem largura fixa, alinhamento estendido
        column_kwargs: dict[str, Any] = {
            "spacing": 12,
            "tight": True,
            "horizontal_alignment": ft.CrossAxisAlignment.STRETCH,
        }
        column_kwargs.update(kwargs)
        super().__init__(**column_kwargs)

        self.on_param_changed = on_param_changed
        self._controls_container = ft.Column(
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        # Montagem visual inicial
        self._setup_module_ui()

    def _setup_module_ui(self) -> None:
        """Monta a estrutura inicial do módulo (cabeçalho didático + controles)."""
        header = self.render_header()
        controls = self.build_controls()
        self._controls_container.controls = controls

        self.controls = [
            header,
            self._controls_container,
        ]

    def render_header(self) -> ft.Container:
        """
        Gera o container do cabeçalho didático responsivo com título, badges e descrição.

        Returns:
            Container estético contendo a apresentação didática do módulo.
        """
        badges: list[ft.Control] = []

        if self.requires_second_input:
            badges.append(
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.LAYERS, size=14, color=theme.PRIMARY_LIGHT),
                            ft.Text(
                                "Requer 2 Imagens (Slot A e B)",
                                size=theme.FONT_CAPTION,
                                color=theme.PRIMARY_LIGHT,
                                weight=ft.FontWeight.W_500,
                            ),
                        ],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.CENTER,
                        tight=True,
                    ),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=6,
                    padding=ft.Padding.symmetric(horizontal=8, vertical=4) if hasattr(ft, "Padding") else 6,
                )
            )

        if self.supports_scalar_mode:
            badges.append(
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.NUMBERS, size=14, color=theme.INFO),
                            ft.Text(
                                "Suporta Escalar / Constante",
                                size=theme.FONT_CAPTION,
                                color=theme.INFO,
                                weight=ft.FontWeight.W_500,
                            ),
                        ],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.CENTER,
                        tight=True,
                    ),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=6,
                    padding=ft.Padding.symmetric(horizontal=8, vertical=4) if hasattr(ft, "Padding") else 6,
                )
            )

        header_content: list[ft.Control] = [
            ft.Row(
                controls=[
                    ft.Icon(ft.Icons.PSYCHOLOGY_ALT, size=20, color=theme.PRIMARY_LIGHT),
                    ft.Text(
                        self.title,
                        size=theme.FONT_TITLE,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.ON_SURFACE,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ]

        if badges:
            header_content.append(
                ft.Row(
                    controls=badges,
                    spacing=6,
                    wrap=True,
                )
            )

        if self.description:
            header_content.append(
                ft.Text(
                    self.description,
                    size=theme.FONT_BODY,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                )
            )

        return ft.Container(
            content=ft.Column(
                controls=header_content,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=theme.BORDER_RADIUS,
            padding=theme.PADDING_CARD,
        )

    def notify_param_changed(self) -> None:
        """
        Dispara a notificação de alteração de parâmetros para o observador/view pai.
        Deve ser associada aos eventos on_change de sliders, dropdowns e inputs.
        """
        if self.on_param_changed is not None:
            self.on_param_changed()

    def refresh_controls(self) -> None:
        """Reconstrói a lista de controles do módulo e atualiza a interface."""
        self._controls_container.controls = self.build_controls()
        if hasattr(self, "update") and self.page is not None:
            self.update()

    def get_params(self) -> dict[str, Any]:
        """
        Retorna o dicionário com os valores atuais dos parâmetros configurados nos controles.
        Pode ser sobrescrito por subclasses para facilidade de desestruturação em process().
        """
        return {}

    @abstractmethod
    def build_controls(self) -> list[ft.Control]:
        """
        Constrói e retorna a lista de controles específicos (Sliders, Dropdowns, etc.) do módulo.

        Garantias de implementação:
          - Não utilizar larguras fixas maiores que 300px (usar expand=True ou Responsividade).
          - Ligar os manipuladores de evento de alteração ao método `self.notify_param_changed()`.

        Returns:
            Lista de controles Flet contextuais.
        """
        pass

    @abstractmethod
    def process(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray | None = None,
        **params: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Executa a transformação de processamento de imagens de forma síncrona e pura.

        Args:
            img_a: Imagem primária (Slot A), formato uint8 (H, W) ou (H, W, 3).
            img_b: Imagem secundária (Slot B) opcional para operações binárias.
            **params: Parâmetros numéricos/configurações da operação.

        Returns:
            Tupla contendo:
              - result_image: np.ndarray uint8 com o resultado da operação.
              - metrics: dict com métricas computadas (ex: MSE, PSNR, tempo, parâmetros didáticos).
        """
        pass


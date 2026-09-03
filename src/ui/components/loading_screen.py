"""
loading_screen.py — Tela de Carregamento Inicial Síncrona com Porcentagem Real.

Exibe uma interface de splash/loading determinística durante a inicialização do app:
  - Logotipo e título do aplicativo PDI.
  - Barra de progresso responsiva acompanhada do percentual exato (0% a 100%).
  - Mensagem explicativa da etapa corrente para transparência ao usuário.
"""

from typing import Any
import flet as ft


class LoadingScreen(ft.Container):
    """Componente de tela cheia para carregamento inicial com feedback visual imediato."""

    def __init__(self, page: ft.Page) -> None:
        self._page = page

        self._icon = ft.Icon(
            icon=ft.Icons.AUTO_AWESOME_MOSAIC,
            size=54,
            color=ft.Colors.PRIMARY,
        )

        self._title = ft.Text(
            value="PDI — Quantização de Imagens",
            size=22,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
        )

        self._subtitle = ft.Text(
            value="Ambiente de Processamento e Análise de Imagens",
            size=13,
            color=ft.Colors.ON_SURFACE_VARIANT,
            text_align=ft.TextAlign.CENTER,
        )

        self._progress_bar = ft.ProgressBar(
            value=0.0,
            width=320,
            color=ft.Colors.PRIMARY,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=ft.BorderRadius.all(4),
        )

        self._percent_text = ft.Text(
            value="0%",
            size=13,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.PRIMARY,
        )

        self._status_text = ft.Text(
            value="Iniciando aplicação...",
            size=12,
            color=ft.Colors.OUTLINE,
            text_align=ft.TextAlign.CENTER,
        )

        progress_row = ft.Row(
            controls=[
                self._progress_bar,
                self._percent_text,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        )

        card_content = ft.Column(
            controls=[
                self._icon,
                ft.Container(height=6),
                self._title,
                self._subtitle,
                ft.Container(height=20),
                progress_row,
                ft.Container(height=4),
                self._status_text,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=4,
        )

        super().__init__(
            content=ft.Container(
                content=card_content,
                padding=32,
                border_radius=ft.BorderRadius.all(16),
                bgcolor=ft.Colors.SURFACE_CONTAINER,
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
                width=460,
                alignment=ft.Alignment.CENTER,
            ),
            expand=True,
            alignment=ft.Alignment.CENTER,
        )

    def set_progress(self, percent: float, message: str) -> None:
        """
        Atualiza o percentual da barra de progresso e a mensagem de status,
        forçando a atualização síncrona da página Flet.

        Args:
            percent: Valor numérico entre 0.0 e 100.0.
            message: Descrição da etapa sendo executada.
        """
        clamped = max(0.0, min(100.0, float(percent)))
        self._progress_bar.value = clamped / 100.0
        self._percent_text.value = f"{int(clamped)}%"
        self._status_text.value = message
        self._page.update()

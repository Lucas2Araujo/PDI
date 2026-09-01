"""
batch_view.py — Aba de Processamento de Imagens em Lote.

Permite ao usuário:
  - Selecionar uma pasta de entrada com múltiplas imagens.
  - Selecionar uma pasta de saída para os resultados.
  - Configurar a técnica de quantização e o nível de bits.
  - Acompanhar o processamento em tempo real via barra de progresso e log.
  - Visualizar um resumo ao final (processadas, falhas).
"""

from pathlib import Path

import flet as ft

from src.core.batch import SUPPORTED_EXTENSIONS, BatchResult, discover_images, process_batch_async
from src.core.grayscale import GrayscaleMethod
from src.core.quantization import QuantizationTechnique
from src.ui import theme


# ---------------------------------------------------------------------------
# Helpers de Compatibilidade
# ---------------------------------------------------------------------------


def _register_file_pickers(page: ft.Page, *pickers: ft.FilePicker) -> None:
    """Registra os FilePickers como serviços na página (Flet 0.86+)."""
    if hasattr(page, "services") and hasattr(page.services, "register_service"):
        for picker in pickers:
            page.services.register_service(picker)


class BatchView(ft.Column):
    """
    View de processamento em lote de um diretório inteiro de imagens.

    Herda ft.Column para ser inserida diretamente como conteúdo de aba.
    """

    def __init__(self, page: ft.Page) -> None:
        super().__init__(
            scroll=ft.ScrollMode.AUTO,
            spacing=16,
            expand=True,
        )
        self._page = page
        self._input_dir: Path | None = None
        self._output_dir: Path | None = None
        self._is_processing = False

        self._build_controls()
        self._assemble_layout()

    # -----------------------------------------------------------------------
    # Construção dos Controles
    # -----------------------------------------------------------------------

    def _build_controls(self) -> None:
        """Inicializa todos os controles da view."""
        # FilePickers registrados como serviços (Flet 0.86+)
        self._input_picker = ft.FilePicker()
        self._output_picker = ft.FilePicker()
        _register_file_pickers(self._page, self._input_picker, self._output_picker)

        # Rótulos dos diretórios selecionados
        self._input_label = ft.Text(
            "Nenhuma pasta selecionada",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
            italic=True,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        self._output_label = ft.Text(
            "Nenhuma pasta selecionada",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
            italic=True,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )

        # Contador de imagens encontradas
        self._image_count_text = ft.Text(
            "",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
        )

        # Dropdown: técnica de quantização
        self._technique_dropdown = ft.Dropdown(
            label="Técnica de Quantização",
            options=[
                ft.dropdown.Option(key=str(QuantizationTechnique.UNIFORM.value), text="Quantização Uniforme"),
                ft.dropdown.Option(key=str(QuantizationTechnique.KMEANS.value), text="Quantização Não-Uniforme (K-Means)"),
                ft.dropdown.Option(key=str(QuantizationTechnique.HISTOGRAM.value), text="Quantização por Histograma (Frequência)"),
            ],
            value=str(QuantizationTechnique.UNIFORM.value),
            color=ft.Colors.ON_SURFACE,
        )

        # Dropdown: método de conversão para cinza (organizado e descritivo)
        self._gray_dropdown = ft.Dropdown(
            label="Método de Conversão para Cinza",
            options=[
                ft.dropdown.Option(key=str(GrayscaleMethod.LUMINANCE.value), text="Luminância ITU-R BT.601 (Padrão: 0.299R + 0.587G + 0.114B)"),
                ft.dropdown.Option(key=str(GrayscaleMethod.AVERAGE.value), text="Média Aritmética ((R+G+B)/3)"),
                ft.dropdown.Option(key=str(GrayscaleMethod.CHANNEL_R.value), text="Isolamento de Canal: Vermelho (R)"),
                ft.dropdown.Option(key=str(GrayscaleMethod.CHANNEL_G.value), text="Isolamento de Canal: Verde (G)"),
                ft.dropdown.Option(key=str(GrayscaleMethod.CHANNEL_B.value), text="Isolamento de Canal: Azul (B)"),
            ],
            value=str(GrayscaleMethod.LUMINANCE.value),
            color=ft.Colors.ON_SURFACE,
        )

        # Slider de bits
        self._bits_value = 4
        self._bits_label = ft.Text(
            f"{self._bits_value} bits  —  {2 ** self._bits_value} tons de cinza",
            size=theme.FONT_BODY,
            color=ft.Colors.ON_SURFACE,
            weight=ft.FontWeight.BOLD,
        )
        self._bits_slider = ft.Slider(
            min=1,
            max=8,
            divisions=7,
            value=self._bits_value,
            label="{value} bits",
            active_color=theme.PRIMARY,
            thumb_color=theme.PRIMARY_LIGHT,
            on_change=self._on_bits_changed,
        )

        # Barra de progresso e status
        self._progress_bar = ft.ProgressBar(
            value=0.0,
            color=theme.PRIMARY,
            height=8,
            border_radius=4,
        )
        self._progress_text = ft.Text(
            "Aguardando início...",
            color=ft.Colors.ON_SURFACE_VARIANT,
            size=theme.FONT_CAPTION,
        )
        self._progress_percent = ft.Text(
            "0%",
            color=ft.Colors.ON_SURFACE,
            size=theme.FONT_BODY,
            weight=ft.FontWeight.BOLD,
        )

        # Log de processamento
        self._log_list = ft.ListView(
            height=200,
            spacing=2,
            auto_scroll=True,
        )

        # Resumo final
        self._summary_card = ft.Container(visible=False)

        # Botões de ação
        self._btn_input = ft.Button(
            content="Pasta de Entrada",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_select_input_dir,
            bgcolor=theme.PRIMARY,
            color="#FFFFFF",
        )
        self._btn_output = ft.Button(
            content="Pasta de Saída",
            icon=ft.Icons.DRIVE_FILE_MOVE,
            on_click=self._on_select_output_dir,
            bgcolor=theme.PRIMARY_DARK,
            color="#FFFFFF",
        )
        self._btn_start = ft.Button(
            content="Iniciar Processamento",
            icon=ft.Icons.PLAY_CIRCLE,
            on_click=self._on_start,
            disabled=True,
            bgcolor=theme.SUCCESS,
            color="#FFFFFF",
        )

    def _assemble_layout(self) -> None:
        """Monta o layout completo da aba de lote."""
        self.controls = [
            # Card 1: Seleção de pastas
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("📁  Seleção de Diretórios"),
                        ft.Divider(height=1),
                        ft.Row(
                            controls=[self._btn_input, self._input_label],
                            spacing=12,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Row(
                            controls=[self._btn_output, self._output_label],
                            spacing=12,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        self._image_count_text,
                    ],
                    spacing=12,
                )
            ),
            # Card 2: Configurações
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("⚙️  Configurações"),
                        ft.Divider(height=1),
                        self._technique_dropdown,
                        self._gray_dropdown,
                        ft.Column(
                            controls=[
                                ft.Row(
                                    controls=[
                                        ft.Text("Nível de Bits:", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_BODY),
                                        self._bits_label,
                                    ],
                                    spacing=8,
                                ),
                                ft.Row(
                                    controls=[
                                        ft.Text("1 bit", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION),
                                        self._bits_slider,
                                        ft.Text("8 bits", color=ft.Colors.ON_SURFACE_VARIANT, size=theme.FONT_CAPTION),
                                    ],
                                    spacing=4,
                                ),
                            ],
                            spacing=6,
                        ),
                        ft.Divider(height=1),
                        self._btn_start,
                    ],
                    spacing=12,
                )
            ),
            # Card 3: Progresso
            theme.card(
                ft.Column(
                    controls=[
                        theme.section_title("⏳  Progresso"),
                        ft.Divider(height=1),
                        ft.Row(
                            controls=[self._progress_text, self._progress_percent],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        self._progress_bar,
                        ft.Container(
                            content=self._log_list,
                            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            border_radius=8,
                            padding=8,
                            height=200,
                        ),
                    ],
                    spacing=10,
                )
            ),
            # Card 4: Resumo (visível apenas ao concluir)
            self._summary_card,
        ]

    # -----------------------------------------------------------------------
    # Handlers de Eventos
    # -----------------------------------------------------------------------

    async def _on_select_input_dir(self, _: ft.ControlEvent) -> None:
        """Abre o seletor para o diretório de entrada."""
        path_str = await self._input_picker.get_directory_path(dialog_title="Selecionar Pasta de Entrada")
        if not path_str:
            return
        self._input_dir = Path(path_str)
        self._input_label.value = str(self._input_dir)
        self._input_label.italic = False
        self._input_label.color = theme.TEXT_PRIMARY

        try:
            images = discover_images(self._input_dir)
            exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            self._image_count_text.value = (
                f"✅  {len(images)} imagem(ns) encontrada(s) — formatos suportados: {exts}"
            )
            self._image_count_text.color = theme.SUCCESS
        except ValueError:
            self._image_count_text.value = "⚠️  Nenhuma imagem suportada encontrada nesta pasta."
            self._image_count_text.color = theme.WARNING

        self._update_start_button_state()
        self._page.update()

    async def _on_select_output_dir(self, _: ft.ControlEvent) -> None:
        """Abre o seletor para o diretório de saída."""
        path_str = await self._output_picker.get_directory_path(dialog_title="Selecionar Pasta de Saída")
        if not path_str:
            return
        self._output_dir = Path(path_str)
        self._output_label.value = str(self._output_dir)
        self._output_label.italic = False
        self._output_label.color = theme.TEXT_PRIMARY
        self._update_start_button_state()
        self._page.update()

    def _on_bits_changed(self, event: ft.ControlEvent) -> None:
        """Atualiza o rótulo de bits ao mover o slider."""
        self._bits_value = int(event.control.value)
        self._bits_label.value = f"{self._bits_value} bits  —  {2 ** self._bits_value} tons de cinza"
        self._page.update()

    def _on_start(self, _: ft.ControlEvent) -> None:
        """Inicia o processamento em lote em thread sincronizada com o Flet."""
        if self._is_processing or not self._input_dir or not self._output_dir:
            return

        technique = self._get_selected_technique()
        gray_method = self._get_selected_gray_method()

        self._is_processing = True
        self._btn_start.disabled = True
        self._btn_input.disabled = True
        self._btn_output.disabled = True
        self._log_list.controls.clear()
        self._summary_card.visible = False
        self._progress_bar.value = 0.0
        self._progress_percent.value = "0%"
        self._progress_text.value = "Iniciando..."
        self._page.update()

        if hasattr(self._page, "run_thread"):
            self._page.run_thread(
                self._run_batch_worker,
                self._input_dir,
                self._output_dir,
                technique,
                self._bits_value,
                gray_method,
            )
        else:
            process_batch_async(
                input_dir=self._input_dir,
                output_dir=self._output_dir,
                technique=technique,
                bits=self._bits_value,
                grayscale_method=gray_method,
                progress_callback=self._on_progress,
                done_callback=self._on_batch_done,
            )

    def _run_batch_worker(
        self,
        input_dir: Path,
        output_dir: Path,
        technique: QuantizationTechnique,
        bits: int,
        grayscale_method: GrayscaleMethod,
    ) -> None:
        """Executa o processamento do lote dentro da thread gerenciada pelo Flet."""
        from src.core.batch import process_batch
        try:
            result = process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                technique=technique,
                bits=bits,
                grayscale_method=grayscale_method,
                progress_callback=self._on_progress,
            )
            self._on_batch_done(result)
        except Exception as error:
            self._progress_text.value = f"Erro: {error}"
            self._is_processing = False
            self._btn_start.disabled = False
            self._btn_input.disabled = False
            self._btn_output.disabled = False
            self._page.update()

    # -----------------------------------------------------------------------
    # Callbacks de Processamento
    # -----------------------------------------------------------------------

    def _on_progress(self, processed: int, total: int, filename: str) -> None:
        """Atualiza a UI de progresso ao concluir cada imagem."""
        ratio = processed / total if total > 0 else 0.0
        self._progress_bar.value = ratio
        self._progress_percent.value = f"{int(ratio * 100)}%"
        self._progress_text.value = f"[{processed}/{total}]  {filename}"

        status_icon = "✅"
        log_entry = ft.Text(
            f"{status_icon}  {filename}",
            color=theme.TEXT_SECONDARY,
            size=theme.FONT_CAPTION,
        )
        self._log_list.controls.append(log_entry)
        self._page.update()

    def _on_batch_done(self, result: BatchResult) -> None:
        """Exibe o resumo final ao concluir o processamento em lote."""
        self._is_processing = False
        self._btn_start.disabled = False
        self._btn_input.disabled = False
        self._btn_output.disabled = False

        for failed_path, error_msg in result.failed:
            log_entry = ft.Text(
                f"❌  {failed_path.name}: {error_msg}",
                color=theme.ACCENT,
                size=theme.FONT_CAPTION,
            )
            self._log_list.controls.append(log_entry)

        self._summary_card = theme.card(
            ft.Column(
                controls=[
                    theme.section_title("✅  Processamento Concluído"),
                    ft.Divider(height=1),
                    ft.Row(
                        controls=[
                            theme.metric_badge("Total", str(result.total)),
                            theme.metric_badge("Sucesso", str(result.success_count), color=theme.SUCCESS),
                            theme.metric_badge("Falhas", str(result.failure_count), color=theme.ACCENT),
                        ],
                        spacing=12,
                    ),
                    ft.Text(
                        f"Resultados salvos em: {result.output_dir}",
                        color=ft.Colors.ON_SURFACE_VARIANT,
                        size=theme.FONT_CAPTION,
                    ),
                ],
                spacing=10,
            )
        )
        self._summary_card.visible = True
        self.controls[-1] = self._summary_card
        self._page.update()

    # -----------------------------------------------------------------------
    # Helpers Privados
    # -----------------------------------------------------------------------

    def _update_start_button_state(self) -> None:
        """Habilita o botão de início apenas quando ambas as pastas estão selecionadas."""
        self._btn_start.disabled = not (self._input_dir and self._output_dir)

    def _get_selected_technique(self) -> QuantizationTechnique:
        """Retorna a técnica de quantização selecionada no dropdown."""
        value = int(self._technique_dropdown.value)
        for technique in QuantizationTechnique:
            if technique.value == value:
                return technique
        return QuantizationTechnique.UNIFORM

    def _get_selected_gray_method(self) -> GrayscaleMethod:
        """Retorna o método de conversão para cinza selecionado no dropdown."""
        value = int(self._gray_dropdown.value)
        for method in GrayscaleMethod:
            if method.value == value:
                return method
        return GrayscaleMethod.LUMINANCE

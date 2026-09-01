"""
batch.py — Motor de Processamento de Imagens em Lote.

Responsável por varrer um diretório de entrada, aplicar a conversão
para escala de cinza e a técnica de quantização desejada a todas as
imagens suportadas, e salvar os resultados em um diretório de saída.

O processamento é executado em uma thread separada para não bloquear
a interface gráfica, reportando o progresso via callback.
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from skimage import io

from src.core.grayscale import (
    GrayscaleMethod,
    colorize_channel,
    is_channel_isolation,
    to_grayscale,
)
from src.core.quantization import QuantizationTechnique, quantize, technique_label

# Formatos de arquivo de imagem suportados pelo módulo
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)


# ---------------------------------------------------------------------------
# Estruturas de Dados Públicas
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    """
    Resultado do processamento em lote de um diretório.

    Attributes:
        total: Número total de imagens encontradas na pasta de entrada.
        processed: Número de imagens processadas com sucesso.
        failed: Lista de tuplas (caminho_do_arquivo, mensagem_de_erro).
        output_dir: Diretório onde os resultados foram salvos.
    """

    total: int = 0
    processed: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)
    output_dir: Path = field(default_factory=Path)

    @property
    def success_count(self) -> int:
        """Número de imagens processadas com sucesso."""
        return self.processed

    @property
    def failure_count(self) -> int:
        """Número de imagens que falharam no processamento."""
        return len(self.failed)


# Tipo do callback de progresso: recebe (imagens_processadas, total, caminho_atual)
ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# API Pública
# ---------------------------------------------------------------------------


def discover_images(directory: Path) -> list[Path]:
    """
    Retorna a lista de arquivos de imagem suportados em um diretório.

    Busca apenas no nível raiz do diretório (sem recursão).

    Args:
        directory: Caminho para o diretório de entrada.

    Returns:
        Lista de objetos Path para os arquivos de imagem encontrados,
        ordenados pelo nome do arquivo.

    Raises:
        ValueError: Se `directory` não existir ou não for um diretório.
    """
    if not directory.is_dir():
        raise ValueError(f"Diretório de entrada inválido: {directory}")

    found = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return found


def process_batch(
    input_dir: Path,
    output_dir: Path,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    progress_callback: ProgressCallback | None = None,
) -> BatchResult:
    """
    Processa todas as imagens de `input_dir` e salva os resultados em `output_dir`.

    Para cada imagem:
      1. Carrega o arquivo de imagem.
      2. Converte para escala de cinza com o método especificado.
      3. Aplica a técnica de quantização.
      4. Salva o resultado em `output_dir` com sufixo padronizado.

    O callback `progress_callback(processadas, total, nome_arquivo)` é chamado
    após cada imagem processada (com sucesso ou falha), permitindo atualização
    da barra de progresso na interface gráfica.

    Args:
        input_dir: Diretório com as imagens de entrada.
        output_dir: Diretório onde os resultados serão salvos (criado se não existir).
        technique: Técnica de quantização a ser aplicada.
        bits: Número de bits de quantização (1 a 8).
        grayscale_method: Método de conversão para escala de cinza.
        progress_callback: Função opcional chamada ao completar cada imagem.

    Returns:
        BatchResult com as estatísticas do processamento.

    Raises:
        ValueError: Se o diretório de entrada for inválido.
    """
    images = discover_images(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = BatchResult(total=len(images), output_dir=output_dir)

    for index, image_path in enumerate(images):
        try:
            quantized = _process_single_image(
                image_path=image_path,
                technique=technique,
                bits=bits,
                grayscale_method=grayscale_method,
            )
            output_path = _build_output_path(output_dir, image_path, technique, bits)
            io.imsave(str(output_path), quantized)
            result.processed += 1

        except Exception as error:  # noqa: BLE001
            result.failed.append((image_path, str(error)))

        if progress_callback is not None:
            progress_callback(index + 1, result.total, image_path.name)

    return result


def process_batch_async(
    input_dir: Path,
    output_dir: Path,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE,
    progress_callback: ProgressCallback | None = None,
    done_callback: Callable[[BatchResult], None] | None = None,
) -> threading.Thread:
    """
    Executa o processamento em lote em uma thread separada (não bloqueante).

    Ideal para uso com interfaces gráficas, evitando o congelamento da tela.
    O `done_callback` é chamado na thread de background ao finalizar.

    Args:
        input_dir: Diretório com as imagens de entrada.
        output_dir: Diretório onde os resultados serão salvos.
        technique: Técnica de quantização a ser aplicada.
        bits: Número de bits de quantização (1 a 8).
        grayscale_method: Método de conversão para escala de cinza.
        progress_callback: Chamado após cada imagem processada.
        done_callback: Chamado ao finalizar todo o lote, recebendo o BatchResult.

    Returns:
        Thread iniciada com o processamento em background.
    """
    def _worker() -> None:
        result = process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            technique=technique,
            bits=bits,
            grayscale_method=grayscale_method,
            progress_callback=progress_callback,
        )
        if done_callback is not None:
            done_callback(result)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Funções Privadas de Apoio
# ---------------------------------------------------------------------------


def _process_single_image(
    image_path: Path,
    technique: QuantizationTechnique,
    bits: int,
    grayscale_method: GrayscaleMethod,
) -> np.ndarray:
    """
    Carrega, converte para cinza/canal e quantiza uma única imagem.

    Args:
        image_path: Caminho para o arquivo de imagem.
        technique: Técnica de quantização.
        bits: Número de bits de quantização.
        grayscale_method: Método de conversão para escala de cinza ou isolamento de canal.

    Returns:
        Array NumPy (H, W) ou (H, W, 3) uint8 com a imagem quantizada.
    """
    image_array = io.imread(str(image_path))
    gray = to_grayscale(image_array, method=grayscale_method)
    quantized = quantize(gray, bits=bits, technique=technique)
    if is_channel_isolation(grayscale_method):
        return colorize_channel(quantized, grayscale_method)
    return quantized



def _build_output_path(
    output_dir: Path,
    source_path: Path,
    technique: QuantizationTechnique,
    bits: int,
) -> Path:
    """
    Constrói o caminho do arquivo de saída com sufixo padronizado.

    Padrão: {nome_original}_{técnica}_{bits}bits.png

    Exemplo: foto.jpg → foto_uniforme_4bits.png
    """
    technique_slug = technique_label(technique).lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    suffix = f"_{technique_slug}_{bits}bits"
    return output_dir / f"{source_path.stem}{suffix}.png"


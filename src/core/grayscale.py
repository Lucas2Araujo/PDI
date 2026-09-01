"""
grayscale.py — Módulo de Conversão para Tons de Cinza.

Responsável por converter imagens coloridas (RGB) para escala de cinza,
suportando múltiplos métodos de conversão para fins didáticos e de pesquisa.

Referências:
    - ITU-R BT.601: https://www.itu.int/rec/R-REC-BT.601/
    - Gonzalez & Woods, "Digital Image Processing", Cap. 6.
"""

from enum import Enum, auto

import numpy as np



class GrayscaleMethod(Enum):
    """Métodos disponíveis para conversão de imagem colorida para escala de cinza."""

    LUMINANCE = auto()   # Padrão ITU-R BT.601 — pesos perceptuais por faixa de comprimento de onda
    AVERAGE = auto()     # Média aritmética simples dos 3 canais (R+G+B)/3
    CHANNEL_R = auto()   # Isolamento do canal vermelho
    CHANNEL_G = auto()   # Isolamento do canal verde
    CHANNEL_B = auto()   # Isolamento do canal azul


# Pesos de luminância perceptual definidos pelo padrão ITU-R BT.601 (NTSC)
# O olho humano é mais sensível ao verde (~58.7%), depois vermelho (~29.9%) e azul (~11.4%)
_LUMINANCE_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)


def to_grayscale(image: np.ndarray, method: GrayscaleMethod = GrayscaleMethod.LUMINANCE) -> np.ndarray:
    """
    Converte uma imagem colorida (RGB ou RGBA) para tons de cinza (uint8).

    A imagem resultante possui valores no intervalo [0, 255] e dtype uint8.
    O canal Alpha é descartado automaticamente em imagens RGBA.

    Args:
        image: Array NumPy da imagem de entrada. Aceita:
               - RGB  (H, W, 3), dtype uint8 ou float [0.0–1.0]
               - RGBA (H, W, 4), dtype uint8 ou float [0.0–1.0]
               - Já em escala de cinza (H, W), retornada sem alteração.
        method: Método de conversão. Padrão: GrayscaleMethod.LUMINANCE.

    Returns:
        Array NumPy (H, W) dtype uint8 com valores em [0, 255].

    Raises:
        ValueError: Se o array de entrada não for uma imagem válida (2D ou 3D).
    """
    _validate_image(image)

    # Imagem já está em escala de cinza — garante apenas o dtype correto
    if image.ndim == 2:
        return _ensure_uint8(image)

    # Remove canal Alpha (RGBA → RGB) antes de processar
    rgb = _strip_alpha(image)

    # Normaliza para float64 em [0.0, 1.0] para os cálculos intermediários
    rgb_float = rgb.astype(np.float64) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float64)

    if method == GrayscaleMethod.LUMINANCE:
        return _convert_luminance(rgb_float)
    if method == GrayscaleMethod.AVERAGE:
        return _convert_average(rgb_float)
    if method == GrayscaleMethod.CHANNEL_R:
        return _extract_channel(rgb_float, channel=0)
    if method == GrayscaleMethod.CHANNEL_G:
        return _extract_channel(rgb_float, channel=1)
    if method == GrayscaleMethod.CHANNEL_B:
        return _extract_channel(rgb_float, channel=2)

    raise ValueError(f"Método de conversão desconhecido: {method}")


def is_channel_isolation(method: GrayscaleMethod) -> bool:
    """Verifica se o método selecionado corresponde ao isolamento de um canal RGB específico."""
    return method in (
        GrayscaleMethod.CHANNEL_R,
        GrayscaleMethod.CHANNEL_G,
        GrayscaleMethod.CHANNEL_B,
    )


def get_channel_index(method: GrayscaleMethod) -> int | None:
    """Retorna o índice do canal RGB (0=R, 1=G, 2=B) ou None se for método combinado."""
    if method == GrayscaleMethod.CHANNEL_R:
        return 0
    if method == GrayscaleMethod.CHANNEL_G:
        return 1
    if method == GrayscaleMethod.CHANNEL_B:
        return 2
    return None


def get_channel_color_name(method: GrayscaleMethod) -> str:
    """Retorna o nome da cor do canal ou 'Cinza'."""
    if method == GrayscaleMethod.CHANNEL_R:
        return "Vermelho"
    if method == GrayscaleMethod.CHANNEL_G:
        return "Verde"
    if method == GrayscaleMethod.CHANNEL_B:
        return "Azul"
    return "Cinza"


def get_channel_color_hex(method: GrayscaleMethod) -> str:
    """Retorna o código hexadecimal correspondente à cor temática do canal."""
    if method == GrayscaleMethod.CHANNEL_R:
        return "#e53935"
    if method == GrayscaleMethod.CHANNEL_G:
        return "#43a047"
    if method == GrayscaleMethod.CHANNEL_B:
        return "#1e88e5"
    return "#4a90d9"


def isolate_channel_rgb(image: np.ndarray, method: GrayscaleMethod) -> np.ndarray:
    """
    Retorna a representação visual cromática do canal isolado em espaço RGB (H, W, 3).

    - Para CHANNEL_R: [R, 0, 0] (tons puros de vermelho)
    - Para CHANNEL_G: [0, G, 0] (tons puros de verde)
    - Para CHANNEL_B: [0, 0, B] (tons puros de azul)
    - Para outros métodos: imagem em escala de cinza (H, W) uint8.

    Args:
        image: Imagem de entrada (H, W, 3 ou 4) ou (H, W).
        method: Método de conversão selecionado.

    Returns:
        Array NumPy (H, W, 3) uint8 se for canal isolado, ou (H, W) se cinza.
    """
    _validate_image(image)
    if image.ndim == 2:
        return _ensure_uint8(image)

    rgb = _strip_alpha(image)
    rgb_uint8 = _ensure_uint8(rgb)

    ch_idx = get_channel_index(method)
    if ch_idx is not None:
        isolated = np.zeros_like(rgb_uint8)
        isolated[:, :, ch_idx] = rgb_uint8[:, :, ch_idx]
        return isolated

    return to_grayscale(image, method=method)


def colorize_channel(gray_or_quantized: np.ndarray, method: GrayscaleMethod) -> np.ndarray:
    """
    Converte um array 2D quantizado ou em escala de cinza no canal de cor temático correspondente.

    - Se method for CHANNEL_R: produz array (H, W, 3) com canal Vermelho ativo.
    - Se method for CHANNEL_G: produz array (H, W, 3) com canal Verde ativo.
    - Se method for CHANNEL_B: produz array (H, W, 3) com canal Azul ativo.
    - Caso contrário: retorna o próprio array 2D uint8 sem alteração.
    """
    ch_idx = get_channel_index(method)
    if ch_idx is not None and gray_or_quantized.ndim == 2:
        h, w = gray_or_quantized.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[:, :, ch_idx] = _ensure_uint8(gray_or_quantized)
        return colored
    return gray_or_quantized


def method_label(method: GrayscaleMethod) -> str:
    """
    Retorna uma string legível descrevendo o método de conversão.

    Args:
        method: Instância de GrayscaleMethod.

    Returns:
        Nome formatado do método para exibição na interface ou logs.
    """
    labels = {
        GrayscaleMethod.LUMINANCE: "Luminância ITU-R BT.601",
        GrayscaleMethod.AVERAGE: "Média Aritmética (R+G+B)/3",
        GrayscaleMethod.CHANNEL_R: "Canal Vermelho (R)",
        GrayscaleMethod.CHANNEL_G: "Canal Verde (G)",
        GrayscaleMethod.CHANNEL_B: "Canal Azul (B)",
    }
    return labels[method]


# ---------------------------------------------------------------------------
# Funções Privadas de Conversão
# ---------------------------------------------------------------------------


def _validate_image(image: np.ndarray) -> None:
    """Valida que o array de entrada é uma imagem com 2 ou 3 dimensões."""
    if not isinstance(image, np.ndarray):
        raise ValueError("A imagem deve ser um array NumPy.")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Array com {image.ndim} dimensões não é uma imagem válida. "
            "Esperado: 2D (escala de cinza) ou 3D (colorida com canais)."
        )


def _strip_alpha(image: np.ndarray) -> np.ndarray:
    """Remove o canal Alpha de imagens RGBA, retornando apenas os canais RGB."""
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Garante que a imagem está em dtype uint8 com valores em [0, 255]."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)
    return image.astype(np.uint8)


def _convert_luminance(rgb_float: np.ndarray) -> np.ndarray:
    """
    Aplica a fórmula de luminância perceptual ITU-R BT.601 de forma vetorizada.

    Fórmula: Y = 0.2989·R + 0.5870·G + 0.1140·B
    """
    gray_float = np.dot(rgb_float, _LUMINANCE_WEIGHTS)
    return (np.clip(gray_float, 0.0, 1.0) * 255).round().astype(np.uint8)


def _convert_average(rgb_float: np.ndarray) -> np.ndarray:
    """
    Converte para escala de cinza calculando a média aritmética dos canais.

    Fórmula: Y = (R + G + B) / 3
    """
    gray_float = rgb_float.mean(axis=2)
    return (np.clip(gray_float, 0.0, 1.0) * 255).round().astype(np.uint8)


def _extract_channel(rgb_float: np.ndarray, channel: int) -> np.ndarray:
    """
    Extrai e retorna um único canal de cor como imagem em escala de cinza.

    Args:
        rgb_float: Array float64 (H, W, 3) normalizado em [0.0, 1.0].
        channel: Índice do canal (0=R, 1=G, 2=B).
    """
    return (rgb_float[:, :, channel] * 255).round().astype(np.uint8)


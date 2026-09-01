"""
quantization.py — Módulo de Quantização de Imagens em Tons de Cinza.

Implementa duas técnicas de quantização digital, reduzindo o número de níveis
de intensidade (bits de quantização) de uma imagem em escala de cinza:

1. **Quantização Uniforme**: Divide o intervalo [0, 255] em intervalos iguais.
   Simples, rápida e determinística.

2. **Quantização Não-Uniforme (K-Means)**: Algoritmo de aprendizado de máquina
   que encontra os k centróides ótimos baseados na distribuição real dos pixels.
   Preserva melhor os detalhes visuais que a quantização uniforme.

Referências:
    - Gonzalez & Woods, "Digital Image Processing", Cap. 8.
    - Lloyd, S.P. (1982). "Least squares quantization in PCM". IEEE Transactions.
"""

from enum import Enum, auto

import numpy as np
from sklearn.cluster import KMeans


class QuantizationTechnique(Enum):
    """Técnicas de quantização disponíveis."""

    UNIFORM = auto()    # Quantização Uniforme — intervalos iguais
    KMEANS = auto()     # Quantização Não-Uniforme — K-Means adaptativo
    HISTOGRAM = auto()  # Quantização por Histograma — particionamento adaptativo por quantis


def quantize(
    image_gray: np.ndarray,
    bits: int,
    technique: QuantizationTechnique,
) -> np.ndarray:
    """
    Aplica a técnica de quantização especificada a uma imagem em escala de cinza.

    Dispatches para `quantize_uniform`, `quantize_kmeans` ou `quantize_histogram`.
    Args:
        image_gray: Array NumPy (H, W) dtype uint8 em escala de cinza.
        bits: Número de bits de quantização. Deve estar no intervalo [1, 8].
        technique: Técnica de quantização a ser utilizada.

    Returns:
        Array NumPy (H, W) dtype uint8 com a imagem quantizada.

    Raises:
        ValueError: Se `bits` estiver fora do intervalo [1, 8] ou
                    `technique` for desconhecida.
    """
    if technique == QuantizationTechnique.UNIFORM:
        return quantize_uniform(image_gray, bits)
    if technique == QuantizationTechnique.KMEANS:
        return quantize_kmeans(image_gray, bits)
    if technique == QuantizationTechnique.HISTOGRAM:
        return quantize_histogram(image_gray, bits)

    raise ValueError(f"Técnica desconhecida: {technique}")


def quantize_uniform(image_gray: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantização Uniforme: divide o espaço de intensidades [0, 255] em
    intervalos de tamanho igual e remapeia cada pixel ao nível representativo
    de seu intervalo.

    Fórmula:
        passo = 256 / n_tons
        índice = pixel // passo
        saída  = índice * (255 / (n_tons - 1))

    Complexidade: O(H·W) — linear no número de pixels.

    Args:
        image_gray: Array NumPy (H, W) dtype uint8.
        bits: Nível de quantização em bits (1 a 8), definindo n_tons = 2^bits.

    Returns:
        Array NumPy (H, W) dtype uint8 com a imagem quantizada uniformemente.

    Raises:
        ValueError: Se `bits` estiver fora do intervalo [1, 8].
    """
    _validate_bits(bits)
    _validate_grayscale_image(image_gray)

    n_tons = 2 ** bits
    passo = 256 // n_tons
    indices = image_gray // passo
    fator_escala = 255 // (n_tons - 1)
    quantizada = (indices * fator_escala).astype(np.uint8)

    return quantizada


def quantize_kmeans(
    image_gray: np.ndarray,
    bits: int,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """
    Quantização Não-Uniforme via K-Means: encontra os k = 2^bits centróides
    que minimizam a distância intra-cluster no espaço de intensidades dos pixels.

    Os centróides resultantes representam os níveis de cinza que melhor
    preservam a distribuição original de intensidades.

    Complexidade: O(H·W · k · iterações) — mais lento que a quantização uniforme
    para imagens grandes, mas com maior qualidade visual.

    Args:
        image_gray: Array NumPy (H, W) dtype uint8.
        bits: Nível de quantização em bits (1 a 8), definindo k = 2^bits clusters.
        random_state: Semente aleatória para reprodutibilidade dos resultados.
        n_init: Número de inicializações do K-Means (maior = mais estável, mais lento).

    Returns:
        Array NumPy (H, W) dtype uint8 com a imagem quantizada pelo K-Means.

    Raises:
        ValueError: Se `bits` estiver fora do intervalo [1, 8].
    """
    _validate_bits(bits)
    _validate_grayscale_image(image_gray)

    n_tons = 2 ** bits

    # Achata a imagem em um vetor coluna (N_pixels, 1) — formato esperado pelo scikit-learn
    pixels = image_gray.reshape(-1, 1).astype(np.float32)

    kmeans = KMeans(n_clusters=n_tons, random_state=random_state, n_init=n_init)
    kmeans.fit(pixels)

    # Mapeia cada centróide para uint8 e reconstrói a imagem na forma original
    centroides = np.uint8(np.round(kmeans.cluster_centers_))
    quantizada = centroides[kmeans.labels_].reshape(image_gray.shape)

    return quantizada


def quantize_histogram(image_gray: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantização Baseada em Histograma: divide o espaço de intensidades em
    intervalos baseados na distribuição de frequência acumulada (quantis/percentis)
    dos pixels.

    Faixas de intensidade com maior densidade de pixels recebem maior precisão
    (mais níveis), enquanto regiões com poucos pixels são agrupadas em faixas mais largas.

    Args:
        image_gray: Array NumPy (H, W) dtype uint8 em escala de cinza.
        bits: Número de bits de quantização (1 a 8).

    Returns:
        Array NumPy (H, W) dtype uint8 quantizado por histograma.
    """
    _validate_bits(bits)
    _validate_grayscale_image(image_gray)

    n_tons = 2 ** bits
    if n_tons >= 256:
        return image_gray.copy()

    flat = image_gray.ravel()
    # Percentis equiprováveis
    percentiles = np.linspace(0, 100, n_tons + 1)
    bins = np.percentile(flat, percentiles)
    bins[0] = 0.0
    bins[-1] = 256.0
    bins = np.unique(bins)

    if len(bins) <= 1:
        return image_gray.copy()

    # Mapeia cada pixel para sua faixa
    digitized = np.digitize(flat, bins[1:-1])

    # Calcula o centroide / média real dos pixels em cada faixa
    out_levels = np.zeros(len(bins), dtype=np.uint8)
    for i in range(len(bins)):
        mask = (digitized == i)
        if np.any(mask):
            out_levels[i] = np.uint8(np.round(np.mean(flat[mask])))
        else:
            idx = min(i, len(bins) - 1)
            out_levels[i] = np.uint8(np.clip(np.round(bins[idx]), 0, 255))

    quantizada = out_levels[digitized].reshape(image_gray.shape)
    return quantizada


def technique_label(technique: QuantizationTechnique) -> str:
    """
    Retorna uma string legível descrevendo a técnica de quantização.

    Args:
        technique: Instância de QuantizationTechnique.

    Returns:
        Nome formatado da técnica para exibição na interface ou logs.
    """
    labels = {
        QuantizationTechnique.UNIFORM: "Quantização Uniforme",
        QuantizationTechnique.KMEANS: "Quantização Não-Uniforme (K-Means)",
        QuantizationTechnique.HISTOGRAM: "Quantização por Histograma (Frequência)",
    }
    return labels[technique]


# ---------------------------------------------------------------------------
# Funções Privadas de Validação
# ---------------------------------------------------------------------------


def _validate_bits(bits: int) -> None:
    """Valida que o número de bits está no intervalo permitido [1, 8]."""
    if not isinstance(bits, int) or not (1 <= bits <= 8):
        raise ValueError(
            f"O número de bits deve ser um inteiro entre 1 e 8. Recebido: {bits!r}"
        )


def _validate_grayscale_image(image: np.ndarray) -> None:
    """Valida que o array de entrada é uma imagem 2D em escala de cinza."""
    if not isinstance(image, np.ndarray):
        raise ValueError("A imagem deve ser um array NumPy.")
    if image.ndim != 2:
        raise ValueError(
            f"Esperado array 2D (H, W) em escala de cinza. "
            f"Recebido array com {image.ndim} dimensões."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"Esperado dtype uint8. Recebido: {image.dtype}. "
            "Use src.core.grayscale.to_grayscale() para preparar a imagem."
        )


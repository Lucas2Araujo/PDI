"""
quantization.py — Módulo de Quantização de Imagens em Tons de Cinza e Coloridas (RGB).

Implementa técnicas de quantização digital, reduzindo o número de níveis
de intensidade (bits de quantização / tamanho de paleta de cores) de imagens:

1. **Quantização Uniforme**: Divide o intervalo [0, 255] em intervalos iguais e
   reconstrói no centróide (ponto médio). Em RGB, aplica mapeamento escalar por canal,
   resultando em (2^b)^3 cores no total.

2. **Quantização Não-Uniforme (K-Means / MiniBatchKMeans)**: Algoritmo adaptativo
   que encontra os k centróides ótimos no espaço 1D (cinza) ou espaço 3D de cores (RGB).
   Carregado sob demanda (Lazy Loading) para acelerar a inicialização Web.

3. **Quantização por Histograma**: Particionamento adaptativo por quantis/percentis.

4. **Dithering por Difusão de Erro (Floyd-Steinberg)**: Difusão de resíduos de quantização
   para os 4 vizinhos imediatos (7/16, 3/16, 5/16, 1/16) em matrizes 2D e tensores 3D RGB.

Referências:
    - Gonzalez & Woods, "Digital Image Processing", Cap. 8.
    - Lloyd, S.P. (1982). "Least squares quantization in PCM". IEEE Transactions.
    - Floyd, R.W. & Steinberg, L. (1976). "An Adaptive Algorithm for Spatial Grey Scale".
"""

from enum import Enum, auto
import gc
from typing import Any, Callable
import numpy as np

# Controle de estado de carregamento do K-Means em NumPy puro
_KMEANS_CLASS = None
_MINIBATCH_KMEANS_CLASS = None


def _quantize_kmeans_1d(
    image: np.ndarray,
    bits: int = 4,
    n_clusters: int | None = None,
    random_state: int = 42,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Quantização K-Means 1D ultra-rápida baseada na distribuição de frequências (histograma).

    Em imagens em escala de cinza de 8 bits, existem no máximo 256 níveis únicos de intensidade.
    Esta função executa o K-Means ponderado diretamente sobre os valores únicos e suas contagens,
    convergindo em milissegundos e mapeando a imagem completa via Look-Up Table (LUT).
    """
    k = 2 ** bits if n_clusters is None else n_clusters
    unique_vals, weights = np.unique(image, return_counts=True)
    n_unique = len(unique_vals)

    if n_unique <= k:
        all_vals = np.arange(256, dtype=np.float32)
        dists = np.abs(all_vals[:, None] - unique_vals.astype(np.float32)[None, :])
        lut = unique_vals[np.argmin(dists, axis=1)].astype(np.uint8)
        return lut[image]

    effective_k = min(k, n_unique)
    unique_vals_f = unique_vals.astype(np.float32)
    weights_f = weights.astype(np.float32)

    # Inicialização K-Means++ ponderada determinística
    rng = np.random.default_rng(random_state)
    probs = weights_f / np.sum(weights_f)
    centers = np.empty(effective_k, dtype=np.float32)
    first_idx = rng.choice(n_unique, p=probs)
    centers[0] = unique_vals_f[first_idx]
    closest_dist_sq = (unique_vals_f - centers[0]) ** 2

    for c in range(1, effective_k):
        combined = closest_dist_sq * weights_f
        s = float(np.sum(combined))
        p_c = (combined / s) if s > 0 else probs
        chosen = rng.choice(n_unique, p=p_c)
        centers[c] = unique_vals_f[chosen]
        closest_dist_sq = np.minimum(closest_dist_sq, (unique_vals_f - centers[c]) ** 2)

    # Iterações de Lloyd ponderadas vetorizadas
    for _ in range(max_iter):
        dists = np.abs(unique_vals_f[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)

        b_weights = np.bincount(labels, weights=weights_f, minlength=effective_k)
        b_sums = np.bincount(labels, weights=unique_vals_f * weights_f, minlength=effective_k)

        active = b_weights > 0
        new_centers = centers.copy()
        new_centers[active] = b_sums[active] / b_weights[active]

        if float(np.max(np.abs(new_centers - centers))) < tol:
            centers = new_centers
            break
        centers = new_centers

    # Mapeamento instantâneo via LUT de 256 bytes
    all_vals = np.arange(256, dtype=np.float32)
    dists_all = np.abs(all_vals[:, None] - centers[None, :])
    nearest = np.argmin(dists_all, axis=1)
    lut = np.uint8(np.clip(np.round(centers[nearest]), 0, 255))
    return lut[image]


class NumPyKMeans:
    """Implementação vetorizada em NumPy puro do algoritmo K-Means (Lloyd + K-Means++)."""

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: int | None = 42,
        n_init: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NumPyKMeans":
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        effective_k = min(self.n_clusters, n_samples)

        unique_samples = np.unique(X, axis=0)
        if len(unique_samples) <= effective_k:
            self.cluster_centers_ = unique_samples.astype(np.float32)
            dists = np.sum((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2)
            self.labels_ = np.argmin(dists, axis=1)
            return self

        best_inertia = float("inf")
        best_centers = None
        best_labels = None

        for init_idx in range(self.n_init):
            seed = None if self.random_state is None else (self.random_state + init_idx * 1000)
            init_rng = np.random.default_rng(seed)
            centers = np.empty((effective_k, n_features), dtype=np.float32)

            # K-Means++ em subamostra se o conjunto for grande
            sub_size = min(10000, len(unique_samples))
            if len(unique_samples) > sub_size:
                sub_idx = init_rng.choice(len(unique_samples), size=sub_size, replace=False)
                sample_pool = unique_samples[sub_idx]
            else:
                sample_pool = unique_samples

            centers[0] = sample_pool[init_rng.integers(0, len(sample_pool))]
            closest_dist_sq = np.sum((sample_pool - centers[0]) ** 2, axis=1)

            for c_idx in range(1, effective_k):
                sum_sq = float(np.sum(closest_dist_sq))
                if sum_sq > 0:
                    probs = closest_dist_sq / sum_sq
                    centers[c_idx] = sample_pool[init_rng.choice(len(sample_pool), p=probs)]
                else:
                    centers[c_idx] = sample_pool[init_rng.integers(0, len(sample_pool))]
                new_dist = np.sum((sample_pool - centers[c_idx]) ** 2, axis=1)
                closest_dist_sq = np.minimum(closest_dist_sq, new_dist)

            # Iterações de Lloyd totalmente vetorizadas com np.bincount e np.add.at
            x_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
            for _ in range(self.max_iter):
                c_norm_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
                dists = x_norm_sq - 2.0 * np.dot(X, centers.T) + c_norm_sq
                labels = np.argmin(dists, axis=1)

                counts = np.bincount(labels, minlength=effective_k)
                centers_sum = np.zeros_like(centers)
                np.add.at(centers_sum, labels, X)

                active = counts > 0
                new_centers = centers.copy()
                new_centers[active] = centers_sum[active] / counts[active, None]

                diff = float(np.max(np.abs(new_centers - centers)))
                centers = new_centers
                if diff < self.tol:
                    break

            c_norm_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
            dists = x_norm_sq - 2.0 * np.dot(X, centers.T) + c_norm_sq
            labels = np.argmin(dists, axis=1)
            inertia = float(np.sum(np.min(dists, axis=1)))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        return self


class NumPyMiniBatchKMeans:
    """Implementação vetorizada em NumPy puro de MiniBatch K-Means para alta velocidade."""

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: int | None = 42,
        n_init: int = 2,
        max_iter: int = 40,
        batch_size: int = 2048,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NumPyMiniBatchKMeans":
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        effective_k = min(self.n_clusters, n_samples)

        if n_samples <= effective_k:
            self.cluster_centers_ = X.copy()
            self.labels_ = np.arange(n_samples, dtype=np.int32)
            return self

        best_inertia = float("inf")
        best_centers = None
        actual_batch_size = min(self.batch_size, n_samples)

        for init_idx in range(self.n_init):
            seed = None if self.random_state is None else (self.random_state + init_idx * 1000)
            init_rng = np.random.default_rng(seed)

            # Subamostragem rápida para K-Means++
            sub_size = min(10000, n_samples)
            sub_idx = init_rng.choice(n_samples, size=sub_size, replace=False)
            sub_X = X[sub_idx]

            centers = np.empty((effective_k, n_features), dtype=np.float32)
            c0 = init_rng.integers(0, sub_size)
            centers[0] = sub_X[c0]
            closest_dist_sq = np.sum((sub_X - centers[0]) ** 2, axis=1)

            for c in range(1, effective_k):
                sum_sq = float(np.sum(closest_dist_sq))
                probs = (closest_dist_sq / sum_sq) if sum_sq > 0 else None
                chosen = init_rng.choice(sub_size, p=probs)
                centers[c] = sub_X[chosen]
                closest_dist_sq = np.minimum(closest_dist_sq, np.sum((sub_X - centers[c]) ** 2, axis=1))

            counts = np.zeros(effective_k, dtype=np.int32)

            for _ in range(self.max_iter):
                batch_idx = init_rng.integers(0, n_samples, size=actual_batch_size)
                batch_X = X[batch_idx]

                b_norm_sq = np.sum(batch_X ** 2, axis=1, keepdims=True)
                c_norm_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
                b_dists = b_norm_sq - 2.0 * np.dot(batch_X, centers.T) + c_norm_sq
                b_labels = np.argmin(b_dists, axis=1)

                old_centers = centers.copy()

                # Atualização 100% vetorizada com np.bincount e np.add.at
                b_counts = np.bincount(b_labels, minlength=effective_k)
                centers_sum = np.zeros_like(centers)
                np.add.at(centers_sum, b_labels, batch_X)

                active = b_counts > 0
                counts[active] += b_counts[active]
                eta = 1.0 / counts[active, None]
                centers[active] = (1.0 - eta) * centers[active] + eta * (centers_sum[active] / b_counts[active, None])

                if float(np.max(np.abs(centers - old_centers))) < self.tol:
                    break

            # Avaliação de inércia em amostra representativa
            eval_size = min(4000, n_samples)
            eval_idx = init_rng.choice(n_samples, size=eval_size, replace=False)
            eval_X = X[eval_idx]
            eval_d = (
                np.sum(eval_X ** 2, axis=1, keepdims=True)
                - 2.0 * np.dot(eval_X, centers.T)
                + np.sum(centers ** 2, axis=1, keepdims=True).T
            )
            inertia = float(np.sum(np.min(eval_d, axis=1)))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers

        self.cluster_centers_ = best_centers

        # Atribuição final em chunks para otimização de cache e memória
        chunk_size = 32768
        self.labels_ = np.empty(n_samples, dtype=np.int32)
        c_norm_sq = np.sum(self.cluster_centers_ ** 2, axis=1, keepdims=True).T
        for i in range(0, n_samples, chunk_size):
            chunk_X = X[i : i + chunk_size]
            chunk_norm = np.sum(chunk_X ** 2, axis=1, keepdims=True)
            dists = chunk_norm - 2.0 * np.dot(chunk_X, self.cluster_centers_.T) + c_norm_sq
            self.labels_[i : i + chunk_size] = np.argmin(dists, axis=1)

        return self


class QuantizationTechnique(Enum):
    """Técnicas de quantização disponíveis."""

    UNIFORM = auto()          # Quantização Uniforme — intervalos iguais (centróides)
    KMEANS = auto()           # Quantização Não-Uniforme — K-Means adaptativo
    HISTOGRAM = auto()        # Quantização por Histograma — particionamento adaptativo por quantis
    FLOYD_STEINBERG = auto()  # Quantização com Dithering — Difusão de Erro (Floyd-Steinberg)


def is_kmeans_loaded() -> bool:
    """Verifica se a classe KMeans já foi inicializada em memória."""
    return _KMEANS_CLASS is not None


def get_kmeans_class(
    on_start_load: Callable[[], None] | None = None,
    on_done_load: Callable[[], None] | None = None,
):
    """Retorna a classe NumPyKMeans compatível."""
    global _KMEANS_CLASS
    if _KMEANS_CLASS is None:
        if on_start_load is not None:
            on_start_load()
        _KMEANS_CLASS = NumPyKMeans
        if on_done_load is not None:
            on_done_load()
    return _KMEANS_CLASS


def get_minibatch_kmeans_class(
    on_start_load: Callable[[], None] | None = None,
    on_done_load: Callable[[], None] | None = None,
):
    """Retorna a classe NumPyMiniBatchKMeans compatível."""
    global _MINIBATCH_KMEANS_CLASS
    if _MINIBATCH_KMEANS_CLASS is None:
        if on_start_load is not None:
            on_start_load()
        _MINIBATCH_KMEANS_CLASS = NumPyMiniBatchKMeans
        if on_done_load is not None:
            on_done_load()
    return _MINIBATCH_KMEANS_CLASS


def quantize(
    image: np.ndarray,
    bits: int,
    technique: QuantizationTechnique,
    n_clusters: int | None = None,
) -> np.ndarray:
    """
    Aplica a técnica de quantização especificada a uma imagem em tons de cinza (2D) ou colorida RGB (3D).

    Dispatches para `quantize_uniform`, `quantize_kmeans`, `quantize_histogram`
    ou `quantizacao_dithering_floyd_steinberg`.

    Args:
        image: Array NumPy (H, W) ou (H, W, 3) dtype uint8.
        bits: Número de bits de quantização (1 a 8).
        technique: Técnica de quantização a ser utilizada.
        n_clusters: Opcional, número explícito de clusters/cores na paleta para K-Means.

    Returns:
        Array NumPy (H, W) ou (H, W, 3) dtype uint8 com a imagem quantizada.

    Raises:
        ValueError: Se `bits` estiver fora do intervalo [1, 8] ou `technique` for desconhecida.
    """
    if technique == QuantizationTechnique.UNIFORM:
        return quantize_uniform(image, bits)
    if technique == QuantizationTechnique.KMEANS:
        return quantize_kmeans(image, bits=bits, n_clusters=n_clusters)
    if technique == QuantizationTechnique.HISTOGRAM:
        return quantize_histogram(image, bits)
    if technique == QuantizationTechnique.FLOYD_STEINBERG:
        return quantizacao_dithering_floyd_steinberg(image, bits)

    raise ValueError(f"Técnica desconhecida: {technique}")


def quantize_uniform(image: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantização Uniforme com Reconstrução por Centróides: divide o espaço de intensidades
    [0, 255] em 2^bits intervalos de tamanho igual e remapeia cada pixel ao ponto médio
    (centróide) do seu respectivo intervalo.

    Fórmula (aplicada por canal em 2D ou 3D RGB):
        n_tons = 2 ** bits
        passo = 256.0 / n_tons
        índice = np.clip(np.floor(imagem / passo), 0, n_tons - 1)
        reconstrucao[i] = np.clip((i + 0.5) * passo, 0, 255).astype(np.uint8)

    Em imagens coloridas RGB (H, W, 3), a quantização produz (2^bits)^3 cores no total.

    Complexidade: O(H·W·C) — linear no número de pixels, acelerado via vetorização NumPy.

    Args:
        image: Array NumPy (H, W) ou (H, W, 3) dtype uint8.
        bits: Nível de quantização em bits (1 a 8), definindo n_tons = 2^bits por canal.

    Returns:
        Array NumPy (H, W) ou (H, W, 3) dtype uint8 com a imagem quantizada uniformemente.

    Raises:
        ValueError: Se `bits` estiver fora do intervalo [1, 8] ou imagem for inválida.
    """
    _validate_bits(bits)
    _validate_image(image)

    n_tons = 2 ** bits
    passo = 256.0 / n_tons

    # Mapeamento do índice de partição para cada pixel/canal em 2^bits intervalos iguais
    indices = np.clip(np.floor(image.astype(np.float32) / passo), 0, n_tons - 1).astype(np.intp)

    # Reconstrução baseada no CENTRÓIDE (ponto médio de cada partição)
    reconstrucao = np.clip((np.arange(n_tons, dtype=np.float32) + 0.5) * passo, 0, 255).astype(np.uint8)

    quantizada = reconstrucao[indices]
    del indices, reconstrucao
    return quantizada


def quantize_kmeans(
    image: np.ndarray,
    bits: int = 4,
    n_clusters: int | None = None,
    random_state: int = 42,
    n_init: int = 10,
    use_minibatch: bool = True,
    on_start_load: Callable[[], None] | None = None,
    on_done_load: Callable[[], None] | None = None,
) -> np.ndarray:
    """
    Quantização Não-Uniforme via K-Means: encontra os k centróides ótimos
    que minimizam a distância intra-cluster no espaço de intensidades (1D para cinza, 3D para RGB).

    Em imagens RGB, executa quantização vetorial no espaço tridimensional de cores agrupando
    (H*W, 3) pixels em k cores de paleta representativas.

    Args:
        image: Array NumPy (H, W) ou (H, W, 3) dtype uint8.
        bits: Nível de quantização em bits (1 a 8), definindo k = 2^bits quando n_clusters não for informado.
        n_clusters: Opcional, quantidade explícita de cores/clusters na paleta final.
        random_state: Semente aleatória para reprodutibilidade dos resultados.
        n_init: Número de inicializações do K-Means.
        use_minibatch: Se True, utiliza MiniBatchKMeans para velocidade máxima em imagens grandes.
        on_start_load: Callback opcional executado caso scikit-learn precise ser importado.
        on_done_load: Callback opcional executado após a importação do scikit-learn.

    Returns:
        Array NumPy (H, W) ou (H, W, 3) dtype uint8 com a imagem quantizada pelo K-Means.

    Raises:
        ValueError: Se `bits` ou `n_clusters` forem inválidos.
    """
    if n_clusters is None:
        _validate_bits(bits)
        k = 2 ** bits
    else:
        if not isinstance(n_clusters, int) or n_clusters < 1 or n_clusters > 1024:
            raise ValueError(f"n_clusters deve ser um inteiro entre 1 e 1024. Recebido: {n_clusters!r}")
        k = n_clusters

    _validate_image(image)

    # Carrega classe de K-Means sob demanda para acionar callbacks caso registrados
    if use_minibatch:
        ClusterClass = get_minibatch_kmeans_class(on_start_load=on_start_load, on_done_load=on_done_load)
    else:
        ClusterClass = get_kmeans_class(on_start_load=on_start_load, on_done_load=on_done_load)

    # Otimização específica para 1D (escala de cinza): K-Means ponderado por histograma via LUT
    if image.ndim == 2:
        return _quantize_kmeans_1d(
            image=image,
            bits=bits,
            n_clusters=n_clusters,
            random_state=random_state,
        )

    # Imagens 3D (RGB): quantização vetorial no cubo de cores RGB
    pixels = image.reshape(-1, 3).astype(np.float32)
    num_samples = pixels.shape[0]
    effective_k = min(k, num_samples)

    try:
        if use_minibatch:
            batch_size = min(2048, max(256, num_samples // 10))
            cluster_model = ClusterClass(
                n_clusters=effective_k,
                random_state=random_state,
                n_init=min(n_init, 3),
                batch_size=batch_size,
            )
        else:
            cluster_model = ClusterClass(
                n_clusters=effective_k,
                random_state=random_state,
                n_init=n_init,
            )

        cluster_model.fit(pixels)

        # Mapeia cada centróide para uint8 [0, 255] e reconstrói a imagem na forma original
        centroides = np.uint8(np.clip(np.round(cluster_model.cluster_centers_), 0, 255))
        quantizada = centroides[cluster_model.labels_].reshape(image.shape)
        return quantizada
    finally:
        del pixels
        gc.collect()


def quantize_histogram(image: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantização Baseada em Histograma: divide o espaço de intensidades em
    intervalos baseados na distribuição de frequência acumulada (quantis/percentis)
    dos pixels.

    Para imagens coloridas RGB (H, W, 3), a quantização é aplicada canal por canal.

    Args:
        image: Array NumPy (H, W) ou (H, W, 3) dtype uint8.
        bits: Número de bits de quantização (1 a 8).

    Returns:
        Array NumPy (H, W) ou (H, W, 3) dtype uint8 quantizado por histograma.
    """
    _validate_bits(bits)
    _validate_image(image)

    if image.ndim == 3:
        # Aplica quantização por quantis em cada canal RGB individualmente
        channels = [quantize_histogram(image[:, :, c], bits) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)

    n_tons = 2 ** bits
    if n_tons >= 256:
        return image.copy()

    flat = image.ravel()
    percentiles = np.linspace(0, 100, n_tons + 1)
    bins = np.percentile(flat, percentiles)
    bins[0] = 0.0
    bins[-1] = 256.0
    bins = np.unique(bins)

    if len(bins) <= 1:
        return image.copy()

    # Mapeia cada pixel para sua faixa
    digitized = np.digitize(flat, bins[1:-1])

    # Calcula o centróide / média real dos pixels em cada faixa
    out_levels = np.zeros(len(bins), dtype=np.uint8)
    for i in range(len(bins)):
        mask = (digitized == i)
        if np.any(mask):
            out_levels[i] = np.uint8(np.round(np.mean(flat[mask])))
        else:
            idx = min(i, len(bins) - 1)
            out_levels[i] = np.uint8(np.clip(np.round(bins[idx]), 0, 255))

    quantizada = out_levels[digitized].reshape(image.shape)
    del flat, digitized, bins, percentiles, out_levels
    return quantizada


_FS_W7: float = 7.0 / 16.0
_FS_W3: float = 3.0 / 16.0
_FS_W5: float = 5.0 / 16.0
_FS_W1: float = 1.0 / 16.0


def _diffuse_floyd_steinberg_2d(
    flat: np.ndarray,
    idx: int,
    next_y_offset: int,
    x: int,
    w: int,
    has_next_row: bool,
    err: float,
) -> None:
    """Difunde o erro residual 2D para os 4 vizinhos imediatos não processados."""
    if x + 1 < w:
        flat[idx + 1] += err * _FS_W7
    if has_next_row:
        n_idx = next_y_offset + x
        if x > 0:
            flat[n_idx - 1] += err * _FS_W3
        flat[n_idx] += err * _FS_W5
        if x + 1 < w:
            flat[n_idx + 1] += err * _FS_W1


def quantizacao_dithering_floyd_steinberg(
    imagem_uint8: np.ndarray,
    n_bits: int,
) -> np.ndarray:
    """
    Quantização com Difusão de Erro Residual (Dithering de Floyd-Steinberg).

    Percorre os pixels da imagem em ordem raster (linha por linha, esquerda para direita),
    quantiza o valor float do pixel atual para o nível representativo mais próximo da paleta
    de 2^n_bits tons e difunde o erro residual de quantização para os 4 vizinhos imediatos
    não processados de acordo com os pesos clássicos de Floyd-Steinberg (1976):
        - Direita:          7/16  (+1,  0)
        - Abaixo-esquerda:  3/16  (-1, +1)
        - Abaixo:           5/16  ( 0, +1)
        - Abaixo-direita:   1/16  (+1, +1)

    Suporta tanto matrizes 2D em escala de cinza (H, W) quanto tensores coloridos RGB (H, W, 3),
    difundindo o vetor de erro residual [eR, eG, eB] de forma contínua.

    Possui tratamento estrito de bordas para não estourar os limites da matriz e
    retorna a imagem final com valores recortados no intervalo [0, 255] em dtype uint8.

    Args:
        imagem_uint8: Array NumPy (H, W) ou (H, W, 3) dtype uint8.
        n_bits: Número de bits de quantização (1 a 8), definindo 2^n_bits níveis por canal.

    Returns:
        Array NumPy (H, W) ou (H, W, 3) dtype uint8 com a imagem quantizada e aprimorada por dithering.

    Raises:
        ValueError: Se `n_bits` estiver fora do intervalo [1, 8] ou se a imagem for inválida.
    """
    _validate_bits(n_bits)
    _validate_image(imagem_uint8)

    if n_bits == 8:
        return imagem_uint8.copy()

    n_levels = 2 ** n_bits
    scale = 255.0 / (n_levels - 1)
    inv_scale = (n_levels - 1) / 255.0
    max_k = n_levels - 1
    palette = np.array([round(k * scale) for k in range(n_levels)], dtype=np.uint8)

    # -----------------------------------------------------------------------
    # Caso 1: Imagem Colorida RGB (H, W, 3)
    # -----------------------------------------------------------------------
    if imagem_uint8.ndim == 3:
        h, w, c = imagem_uint8.shape
        arr = imagem_uint8.astype(np.float32)
        out = np.empty((h, w, 3), dtype=np.uint8)

        for y in range(h):
            has_next_row = (y + 1 < h)
            for x in range(w):
                # Processa os 3 canais [R, G, B] para o pixel atual
                err_r: float = 0.0
                err_g: float = 0.0
                err_b: float = 0.0

                for ch in range(3):
                    old_val = arr[y, x, ch]
                    v = old_val * inv_scale
                    k = int(v + 0.5) if v >= 0.0 else int(v - 0.5)
                    k = max(0, min(max_k, k))
                    new_val_f = k * scale
                    out[y, x, ch] = palette[k]
                    err = old_val - new_val_f
                    if ch == 0:
                        err_r = err
                    elif ch == 1:
                        err_g = err
                    else:
                        err_b = err

                # Difusão vetorial do resíduo [err_r, err_g, err_b] aos vizinhos
                err_vec = np.array([err_r, err_g, err_b], dtype=np.float32)

                # Direita (+1, 0) -> 7/16
                if x + 1 < w:
                    arr[y, x + 1] += err_vec * _FS_W7

                if has_next_row:
                    # Abaixo-esquerda (-1, +1) -> 3/16
                    if x > 0:
                        arr[y + 1, x - 1] += err_vec * _FS_W3
                    # Abaixo (0, +1) -> 5/16
                    arr[y + 1, x] += err_vec * _FS_W5
                    # Abaixo-direita (+1, +1) -> 1/16
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err_vec * _FS_W1

        del arr, palette
        return out

    # -----------------------------------------------------------------------
    # Caso 2: Imagem Monocromática 2D (H, W)
    # -----------------------------------------------------------------------
    h, w = imagem_uint8.shape
    arr = imagem_uint8.astype(np.float32)
    flat = arr.ravel()
    out = np.empty((h, w), dtype=np.uint8)
    out_flat = out.ravel()

    for y in range(h):
        y_offset = y * w
        next_y_offset = y_offset + w
        has_next_row = (y + 1 < h)
        for x in range(w):
            idx = y_offset + x
            old = flat[idx]

            # Quantização para o nível mais próximo em float
            v = old * inv_scale
            k = int(v + 0.5) if v >= 0.0 else int(v - 0.5)
            k = max(0, min(max_k, k))

            new_val_f = k * scale
            out_flat[idx] = palette[k]
            err = old - new_val_f

            # Difusão de erro para vizinhos
            _diffuse_floyd_steinberg_2d(flat, idx, next_y_offset, x, w, has_next_row, err)

    del arr, flat, out_flat, palette
    return out


# Alias compatível com as demais funções de quantização do módulo
quantize_floyd_steinberg = quantizacao_dithering_floyd_steinberg


def technique_label(technique: QuantizationTechnique) -> str:
    """
    Retorna uma string legível descrevendo a técnica de quantização.

    Args:
        technique: Instância de QuantizationTechnique.

    Returns:
        Nome formatado da técnica para exibição na interface ou logs.
    """
    labels = {
        QuantizationTechnique.UNIFORM: "Quantização Uniforme (Centróides)",
        QuantizationTechnique.KMEANS: "Quantização Não-Uniforme (K-Means)",
        QuantizationTechnique.HISTOGRAM: "Quantização por Histograma (Frequência)",
        QuantizationTechnique.FLOYD_STEINBERG: "Dithering por Difusão de Erro (Floyd-Steinberg)",
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


def _validate_image(image: np.ndarray) -> None:
    """Valida que o array de entrada é uma imagem NumPy válida (2D cinza ou 3D RGB com 3 canais) em uint8."""
    if not isinstance(image, np.ndarray):
        raise ValueError("A imagem deve ser um array NumPy.")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Esperado array 2D (H, W) ou 3D (H, W, 3). "
            f"Recebido array com {image.ndim} dimensões."
        )
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(
            f"Esperado array 3D com 3 canais RGB (H, W, 3). Recebido shape: {image.shape}."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"Esperado dtype uint8. Recebido: {image.dtype}."
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
            f"Esperado dtype uint8. Recebido: {image.dtype}."
        )

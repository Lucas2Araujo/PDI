"""
binary_ops.py — Módulo de Operações Binárias (Aritméticas e Lógicas) em Imagens.

Implementa operações ponto a ponto puras baseadas em NumPy e PIL,
suportando imagens monocromáticas (Grayscale) e coloridas (RGB),
com múltiplos modos de compatibilização dimensional e rigor acadêmico.

Referências:
    - Gonzalez & Woods, "Digital Image Processing", Cap. 2 (Image Operations).
"""

from enum import Enum, auto
from typing import Any
import numpy as np
from PIL import Image


class ResizeMode(Enum):
    """
    Modos de compatibilização dimensional entre dois operandos de imagem.

    Attributes:
        STRICT: Exige que ambas as imagens tenham exatamente o mesmo shape.
                Lança ValueError amigável em caso de divergência (Rigor acadêmico).
        RESIZE_B_TO_A: Redimensiona a imagem B para coincidir com as dimensões de A
                       utilizando interpolação bilinear via PIL.
        CROP_COMMON: Recorta ambas as imagens para o menor retângulo comum
                     (min(H_a, H_b), min(W_a, W_b)) a partir da origem (0, 0).
    """
    STRICT = auto()
    RESIZE_B_TO_A = auto()
    CROP_COMMON = auto()


def _is_scalar(val: Any) -> bool:
    """Verifica se um operando é um valor escalar numérico."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    if isinstance(val, np.ndarray) and val.ndim == 0:
        return True
    return False


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Valida e normaliza o array de entrada para garantir formato e dtype uint8.

    Args:
        image: Array NumPy representando a imagem.

    Returns:
        Array NumPy no formato uint8 [0, 255].

    Raises:
        ValueError: Se o array não possuir 2 ou 3 dimensões ou canais inválidos.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Esperado np.ndarray, recebido {type(image).__name__}")

    if image.ndim not in (2, 3):
        raise ValueError(f"Imagem deve possuir 2 ou 3 dimensões. Shape recebido: {image.shape}")

    arr = image

    # Descarta canal alfa se for RGBA (H, W, 4)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] != 3:
            raise ValueError(f"Imagem 3D deve possuir 1, 3 ou 4 canais. Shape recebido: {image.shape}")

    # Normalização de ponto flutuante para uint8 [0, 255]
    if np.issubdtype(arr.dtype, np.floating):
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        if max_v <= 1.0 and min_v >= 0.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def align_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mode: ResizeMode = ResizeMode.STRICT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compatibiliza as dimensões espaciais e número de canais entre duas imagens.

    Args:
        img_a: Imagem de referência (ou primeiro operando).
        img_b: Segunda imagem a ser compatibilizada.
        mode: Estratégia de compatibilização (STRICT, RESIZE_B_TO_A, CROP_COMMON).

    Returns:
        Tupla (img_a_alinhada, img_b_alinhada) ambas com mesmo shape e dtype uint8.

    Raises:
        ValueError: Se o modo for STRICT e os shapes forem divergentes.
    """
    a = _ensure_uint8(img_a)
    b = _ensure_uint8(img_b)

    if mode == ResizeMode.STRICT:
        if a.shape != b.shape:
            raise ValueError(
                f"Dimensões incompatíveis em modo STRICT: Imagem A possui shape {a.shape} "
                f"e Imagem B possui shape {b.shape}. Para operar sobre imagens de tamanhos "
                "distintos, utilize ResizeMode.RESIZE_B_TO_A ou ResizeMode.CROP_COMMON."
            )
        return a, b

    h_a, w_a = a.shape[:2]
    h_b, w_b = b.shape[:2]
    c_a = 3 if a.ndim == 3 else 1
    c_b = 3 if b.ndim == 3 else 1

    if mode == ResizeMode.RESIZE_B_TO_A:
        # Redimensiona B espacialmente para (w_a, h_a)
        if (w_b, h_b) != (w_a, h_a):
            pil_b = Image.fromarray(b)
            pil_b = pil_b.resize((w_a, h_a), resample=Image.Resampling.BILINEAR)
            b = np.array(pil_b, dtype=np.uint8)

        # Compatibilização de canais
        if c_a == 3 and c_b == 1:
            # Expande B (Grayscale -> RGB)
            b = np.repeat(b[:, :, np.newaxis], 3, axis=2)
        elif c_a == 1 and c_b == 3:
            # Converte B (RGB -> Grayscale ITU-R BT.601)
            b = np.round(b[:, :, 0] * 0.2989 + b[:, :, 1] * 0.5870 + b[:, :, 2] * 0.1140).astype(np.uint8)

        return a, b

    if mode == ResizeMode.CROP_COMMON:
        common_h = min(h_a, h_b)
        common_w = min(w_a, w_b)

        a_cropped = a[:common_h, :common_w]
        b_cropped = b[:common_h, :common_w]

        # Compatibilização de canais se divergirem
        if c_a == 3 and c_b == 1:
            b_cropped = np.repeat(b_cropped[:, :, np.newaxis], 3, axis=2)
        elif c_a == 1 and c_b == 3:
            b_cropped = np.round(
                b_cropped[:, :, 0] * 0.2989 + b_cropped[:, :, 1] * 0.5870 + b_cropped[:, :, 2] * 0.1140
            ).astype(np.uint8)

        return a_cropped, b_cropped

    raise ValueError(f"Modo de compatibilização não reconhecido: {mode}")


def add(
    img_a: np.ndarray,
    img_b_or_val: np.ndarray | int | float,
    clip: bool = True,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Soma ponto a ponto entre imagem A e imagem B (ou escalar).

    Operação: A + B
    Com clip=True, valores são saturados no intervalo [0, 255].
    Com clip=False, aplica aritmética modular uint8 (wrap-around).

    Args:
        img_a: Imagem base uint8.
        img_b_or_val: Segunda imagem ou valor escalar (int/float).
        clip: Se True, satura valores em [0, 255]. Se False, wrap-around.
        resize_mode: Modo de compatibilização caso img_b seja imagem.

    Returns:
        Array NumPy uint8 resultante.
    """
    a = _ensure_uint8(img_a)

    if _is_scalar(img_b_or_val):
        val = float(img_b_or_val)
        res = a.astype(np.int32) + int(round(val))
    else:
        a_aligned, b_aligned = align_images(a, img_b_or_val, mode=resize_mode)
        res = a_aligned.astype(np.int32) + b_aligned.astype(np.int32)

    if clip:
        return np.clip(res, 0, 255).astype(np.uint8)
    return res.astype(np.uint8)


def subtract(
    img_a: np.ndarray,
    img_b_or_val: np.ndarray | int | float,
    absolute: bool = False,
    clip: bool = True,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Subtração ponto a ponto entre imagem A e imagem B (ou escalar).

    Operação: A - B (ou |A - B| se absolute=True).
    Com clip=True, valores inferiores a zero são saturados em 0.
    Com absolute=True, retorna a diferença absoluta (ideal para detecção de variações).

    Args:
        img_a: Imagem base uint8.
        img_b_or_val: Segunda imagem ou valor escalar.
        absolute: Se True, calcula a magnitude absoluta da diferença |A - B|.
        clip: Se True, satura valores em [0, 255].
        resize_mode: Modo de compatibilização caso img_b seja imagem.

    Returns:
        Array NumPy uint8 resultante.
    """
    a = _ensure_uint8(img_a)

    if _is_scalar(img_b_or_val):
        val = float(img_b_or_val)
        res = a.astype(np.int32) - int(round(val))
    else:
        a_aligned, b_aligned = align_images(a, img_b_or_val, mode=resize_mode)
        res = a_aligned.astype(np.int32) - b_aligned.astype(np.int32)

    if absolute:
        res = np.abs(res)

    if clip:
        return np.clip(res, 0, 255).astype(np.uint8)
    return res.astype(np.uint8)


def blend(
    img_a: np.ndarray,
    img_b: np.ndarray,
    alpha: float = 0.5,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Mistura linear ponderada (Alpha Blending) entre duas imagens.

    Operação: alpha * A + (1 - alpha) * B.

    Args:
        img_a: Primeira imagem (peso alpha).
        img_b: Segunda imagem (peso 1 - alpha).
        alpha: Fator de ponderação no intervalo [0.0, 1.0].
        resize_mode: Modo de compatibilização dimensional.

    Returns:
        Array NumPy uint8 resultante.

    Raises:
        ValueError: Se alpha estiver fora do intervalo [0.0, 1.0].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"O parâmetro alpha deve estar contido em [0.0, 1.0]. Recebido: {alpha}")

    a_aligned, b_aligned = align_images(img_a, img_b, mode=resize_mode)

    res = alpha * a_aligned.astype(np.float32) + (1.0 - alpha) * b_aligned.astype(np.float32)
    return np.clip(np.round(res), 0, 255).astype(np.uint8)


def multiply(
    img_a: np.ndarray,
    scalar_or_img: np.ndarray | int | float,
    clip: bool = True,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Multiplicação ponto a ponto entre imagem e escalar (ou máscara/imagem).

    Operação: A * B.
    Útil para mascaramento binário (ROI) e ajuste de brilho/ganho de contraste.

    Args:
        img_a: Imagem base uint8.
        scalar_or_img: Fator escalar multiplicativo ou imagem/máscara.
        clip: Se True, satura valores em 255. Se False, aritmética modular uint8.
        resize_mode: Modo de compatibilização caso seja imagem.

    Returns:
        Array NumPy uint8 resultante.
    """
    a = _ensure_uint8(img_a)

    if _is_scalar(scalar_or_img):
        res = a.astype(np.float32) * float(scalar_or_img)
    else:
        a_aligned, b_aligned = align_images(a, scalar_or_img, mode=resize_mode)
        res = a_aligned.astype(np.float32) * b_aligned.astype(np.float32)

    if clip:
        return np.clip(np.round(res), 0, 255).astype(np.uint8)
    return np.round(res).astype(np.int64).astype(np.uint8)


def divide(
    img_a: np.ndarray,
    scalar_or_img: np.ndarray | int | float,
    eps: float = 1e-5,
    clip: bool = True,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Divisão ponto a ponto entre imagem e escalar (ou imagem/fundo).

    Operação: A / (B + eps).
    Evita divisão por zero adicionando eps ao denominador ou substituindo zeros.
    Útil para correção de sombreamento não uniforme (flattening de fundo).

    Args:
        img_a: Imagem dividendo uint8.
        scalar_or_img: Escalar divisor ou imagem/estimativa de fundo.
        eps: Fator infinitesimal para prevenir divisão por zero (padrão 1e-5).
        clip: Se True, satura em [0, 255]. Se False, aritmética modular uint8.
        resize_mode: Modo de compatibilização caso seja imagem.

    Returns:
        Array NumPy uint8 resultante.
    """
    a = _ensure_uint8(img_a)

    if _is_scalar(scalar_or_img):
        val = float(scalar_or_img)
        denom = val if abs(val) >= eps else (eps if val >= 0 else -eps)
        res = a.astype(np.float32) / denom
    else:
        a_aligned, b_aligned = align_images(a, scalar_or_img, mode=resize_mode)
        b_f = b_aligned.astype(np.float32)
        denom = np.where(np.abs(b_f) < eps, eps, b_f)
        res = a_aligned.astype(np.float32) / denom

    if clip:
        return np.clip(np.round(res), 0, 255).astype(np.uint8)
    return np.round(res).astype(np.int64).astype(np.uint8)


def bitwise_and(
    img_a: np.ndarray,
    img_b_or_val: np.ndarray | int,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Operação lógica E (Bitwise AND) bit a bit.

    Operação: A & B.
    Comum em fatiamento de planos de bits e mascaramento com regiões de interesse.

    Args:
        img_a: Imagem uint8.
        img_b_or_val: Segunda imagem uint8 ou máscara escalar inteira.
        resize_mode: Modo de compatibilização caso seja imagem.

    Returns:
        Array NumPy uint8 resultante.
    """
    a = _ensure_uint8(img_a)
    if _is_scalar(img_b_or_val):
        val = np.uint8(int(img_b_or_val) & 0xFF)
        return np.bitwise_and(a, val)

    a_aligned, b_aligned = align_images(a, img_b_or_val, mode=resize_mode)
    return np.bitwise_and(a_aligned, b_aligned)


def bitwise_or(
    img_a: np.ndarray,
    img_b_or_val: np.ndarray | int,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Operação lógica OU (Bitwise OR) bit a bit.

    Operação: A | B.
    Comum para combinação de mapas binários e reconstrução de planos.

    Args:
        img_a: Imagem uint8.
        img_b_or_val: Segunda imagem uint8 ou máscara escalar inteira.
        resize_mode: Modo de compatibilização caso seja imagem.

    Returns:
        Array NumPy uint8 resultante.
    """
    a = _ensure_uint8(img_a)
    if _is_scalar(img_b_or_val):
        val = np.uint8(int(img_b_or_val) & 0xFF)
        return np.bitwise_or(a, val)

    a_aligned, b_aligned = align_images(a, img_b_or_val, mode=resize_mode)
    return np.bitwise_or(a_aligned, b_aligned)


def bitwise_xor(
    img_a: np.ndarray,
    img_b_or_val: np.ndarray | int,
    resize_mode: ResizeMode = ResizeMode.STRICT,
) -> np.ndarray:
    """
    Operação lógica OU Exclusivo (Bitwise XOR) bit a bit.

    Operação: A ^ B.
    Comum para criptografia visual, detecção de diferenças e marcas d'água.

    Args:
        img_a: Imagem uint8.
        img_b_or_val: Segunda imagem uint8 ou máscara escalar inteira.
        resize_mode: Modo de compatibilização caso seja imagem.

    Returns:
        Array NumPy uint8 resultante.
    """
    a = _ensure_uint8(img_a)
    if _is_scalar(img_b_or_val):
        val = np.uint8(int(img_b_or_val) & 0xFF)
        return np.bitwise_xor(a, val)

    a_aligned, b_aligned = align_images(a, img_b_or_val, mode=resize_mode)
    return np.bitwise_xor(a_aligned, b_aligned)


def bitwise_not(img_a: np.ndarray) -> np.ndarray:
    """
    Operação lógica NÃO (Bitwise NOT / Inversão de bits / Negativo digital).

    Operação: ~A (equivalente a 255 - A para uint8).

    Args:
        img_a: Imagem uint8.

    Returns:
        Array NumPy uint8 com todos os bits invertidos.
    """
    a = _ensure_uint8(img_a)
    return np.bitwise_not(a)


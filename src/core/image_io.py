"""
image_io.py — Utilitários de I/O, Redimensionamento Preventivo e Gerenciamento de Imagens.

Centraliza as rotinas de carregamento, downscaling preventivo (máximo 800×800 pixels)
e conversão entre bytes PNG, objetos PIL e arrays NumPy.

Mitiga riscos de Out-Of-Memory (OOM) no WebAssembly/Pyodide e em dispositivos móveis.
"""

import gc
import io
from pathlib import Path
import numpy as np
from PIL import Image

# Limite máximo de resolução padrão para imagens de entrada
MAX_IMAGE_DIMENSION: int = 800
MAX_THUMBNAIL_DIMENSION: int = 160


def preventive_resize(
    pil_img: Image.Image,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> Image.Image:
    """
    Redimensiona preventivamente uma imagem PIL mantendo a proporção de aspecto (aspect ratio)
    caso sua largura ou altura ultrapasse `max_dim`.

    Args:
        pil_img: Imagem PIL de entrada.
        max_dim: Dimensão máxima permitida em pixels (largura ou altura).

    Returns:
        Imagem PIL redimensionada (ou original caso já esteja dentro do limite).
    """
    w, h = pil_img.size
    if w > max_dim or h > max_dim:
        resized = pil_img.copy()
        resized.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return resized
    return pil_img


def open_and_downscale_image(
    source: Path | str | bytes | io.BytesIO | Image.Image,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    """
    Abre e decodifica uma imagem a partir de diversas fontes (Path, bytes ou PIL),
    aplica o redimensionamento preventivo (máx `max_dim` px) e converte para array NumPy uint8.

    Normaliza modos de cor:
      - 'RGBA', 'LA', 'P' -> 'RGB'
      - 'L' -> 2D uint8
      - 'RGB' -> 3D (H, W, 3) uint8

    Args:
        source: Caminho do arquivo, bytes brutos, stream ou instância de Image.Image.
        max_dim: Limite máximo para largura e altura. Padrão: 800 px.

    Returns:
        Array NumPy (H, W, 3) ou (H, W) dtype uint8 com dimensões <= max_dim.
    """
    pil_img: Image.Image | None = None
    should_close = False

    if isinstance(source, Image.Image):
        pil_img = source
    elif isinstance(source, bytes):
        pil_img = Image.open(io.BytesIO(source))
        should_close = True
    elif isinstance(source, io.BytesIO):
        pil_img = Image.open(source)
        should_close = True
    else:
        path_obj = Path(source)
        if not path_obj.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path_obj}")
        pil_img = Image.open(path_obj)
        should_close = True

    try:
        # Converte modos indexados ou com canal alfa para RGB mantendo fidelidade cromática
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")

        # Aplica o downscaling preventivo
        pil_img = preventive_resize(pil_img, max_dim=max_dim)

        arr = np.array(pil_img)

        # Garante uint8
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Se for 3D com mais de 3 canais, descarta extras
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]

        return arr
    finally:
        if should_close and pil_img is not None:
            pil_img.close()
        gc.collect()


def _array_to_pil(array: np.ndarray) -> Image.Image:
    """Converte um array NumPy uint8 em objeto Image do Pillow preservando canais."""
    if array.ndim == 3 and array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA")
    if array.ndim == 3 and array.shape[2] >= 3:
        return Image.fromarray(array[:, :, :3], mode="RGB")
    return Image.fromarray(array.astype(np.uint8), mode="L")


def array_to_png_bytes(
    array: np.ndarray,
    max_dim: int | None = None,
    optimize: bool = True,
) -> bytes:
    """
    Converte um array NumPy uint8 em bytes codificados no formato PNG em memória.

    Args:
        array: Matriz NumPy (H, W) ou (H, W, 3) uint8.
        max_dim: Opcional, dimensão máxima para redimensionamento antes da codificação.
        optimize: Se True, ativa a otimização de compressão PNG do Pillow.

    Returns:
        Bytes da imagem PNG.
    """
    pil_img = _array_to_pil(array)

    if max_dim is not None:
        pil_img = preventive_resize(pil_img, max_dim=max_dim)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=optimize)
    result = buf.getvalue()
    buf.close()
    pil_img.close()
    return result


def make_thumbnail_png(
    array: np.ndarray,
    max_size: int = MAX_THUMBNAIL_DIMENSION,
) -> bytes:
    """
    Gera bytes PNG compactos de miniatura para exibição ultra-rápida e leve na interface Flet.

    Args:
        array: Matriz NumPy da imagem.
        max_size: Dimensão máxima da miniatura (padrão 160 px).

    Returns:
        Bytes PNG da miniatura.
    """
    pil_img = _array_to_pil(array)
    pil_img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    result = buf.getvalue()
    buf.close()
    pil_img.close()
    return result



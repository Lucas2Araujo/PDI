"""
samples.py — Gerenciamento e carregamento de imagens de exemplo embutidas no app.
"""

import os
from pathlib import Path
import numpy as np

from src.core.image_io import (
    MAX_IMAGE_DIMENSION,
    array_to_png_bytes,
    open_and_downscale_image,
)


def _get_asset_search_dirs() -> list[Path]:
    """Retorna os diretórios candidatos para busca de assets em diferentes ambientes."""
    candidates: list[Path] = []
    if "FLET_ASSETS_DIR" in os.environ:
        candidates.append(Path(os.environ["FLET_ASSETS_DIR"]))
    if "FLET_APP_DIR" in os.environ:
        candidates.append(Path(os.environ["FLET_APP_DIR"]) / "assets")

    candidates.extend([
        Path(__file__).resolve().parent.parent / "assets",         # src/assets
        Path(__file__).resolve().parent.parent.parent / "assets",  # assets raiz do projeto
        Path(__file__).resolve().parent / "assets",                # src/core/assets
        Path.cwd() / "assets",                                     # cwd/assets
    ])
    return candidates


def _find_assets_dir() -> Path:
    """
    Localiza o diretório de assets com suporte a múltiplos ambientes de execução:
    Desktop local, Pyodide (WebAssembly), FLET_ASSETS_DIR e caminhos relativos.
    """
    candidates = _get_asset_search_dirs()

    for cand in candidates:
        if cand.exists() and (cand / "sample_portrait.png").exists():
            return cand

    for cand in candidates:
        if cand.exists():
            return cand

    return candidates[0]


# Localização padrão dos assets
ASSETS_DIR = _find_assets_dir()

SAMPLE_PORTRAIT_NAME = "sample_portrait.png"
SAMPLE_BENCHMARK_NAME = "sample_benchmark.png"
SAMPLE_LENA_NAME = "lena_color.png"
SAMPLE_AYLA_NAME = "ayla.jpg"
SAMPLE_PENTAGONO_NAME = "pentagono.tiff"

SAMPLE_OPTIONS = [
    {
        "id": "portrait",
        "name": SAMPLE_PORTRAIT_NAME,
        "title": "Exemplo 1: Retrato RGB (512×512)",
        "button_label": "Retrato RGB",
        "icon": "PERSON",
        "description": "Imagem colorida com tons de pele, sombras e gradientes naturais.",
    },
    {
        "id": "benchmark",
        "name": SAMPLE_BENCHMARK_NAME,
        "title": "Exemplo 2: Benchmark Sintético (512×512)",
        "button_label": "Benchmark Sintético",
        "icon": "AUTO_GRAPH",
        "description": "Degraus de luminância, formas geométricas e gradientes contínuos.",
    },
    {
        "id": "lena",
        "name": SAMPLE_LENA_NAME,
        "title": "Exemplo 3: Lenna Clássica (512×512)",
        "button_label": "Lenna Clássica",
        "icon": "FACE",
        "description": "Imagem clássica padrão de testes em Processamento Digital de Imagens.",
    },
    {
        "id": "ayla",
        "name": SAMPLE_AYLA_NAME,
        "title": "Exemplo 4: Ayla HD (Otimizada máx 800×800)",
        "button_label": "Ayla (Foto HD)",
        "icon": "PETS",
        "description": "Fotografia real em alta resolução com texturas ricas de pelos e iluminação.",
    },
    {
        "id": "pentagono",
        "name": SAMPLE_PENTAGONO_NAME,
        "title": "Exemplo 5: Pentágono PDI (Otimizado máx 800×800)",
        "button_label": "Pentágono (TIFF 8-bit)",
        "icon": "ACCOUNT_BALANCE",
        "description": "Benchmark monocromático de PDI com estruturas geométricas e pistas aéreas.",
    },
]


def get_sample_path(sample_name: str) -> Path:
    """
    Retorna o Path para a imagem de exemplo solicitada, buscando em todos os
    diretórios candidatos disponíveis no ambiente de execução.
    """
    candidates = [d / sample_name for d in _get_asset_search_dirs()]
    if (ASSETS_DIR / sample_name) not in candidates:
        candidates.insert(0, ASSETS_DIR / sample_name)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Imagem de exemplo não encontrada: {sample_name} (buscado em {len(candidates)} locais)"
    )


def load_sample_array(
    sample_name: str,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    """
    Carrega e retorna a imagem de exemplo como array NumPy uint8 (RGB ou Cinza),
    aplicando downscaling preventivo (máx 800×800 px) para otimização de memória.
    """
    path = get_sample_path(sample_name)
    return open_and_downscale_image(path, max_dim=max_dim)


def load_sample_bytes(
    sample_name: str,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> bytes:
    """Retorna os bytes da imagem de exemplo codificados em PNG com downscaling preventivo."""
    arr = load_sample_array(sample_name, max_dim=max_dim)
    return array_to_png_bytes(arr)

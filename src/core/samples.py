"""
samples.py — Gerenciamento e carregamento de imagens de exemplo embutidas no app.
"""

from pathlib import Path
import numpy as np
from PIL import Image

# Localização padrão dos assets
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

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
        "title": "Exemplo 4: Ayla HD (1494×1600)",
        "button_label": "Ayla (Foto HD)",
        "icon": "PETS",
        "description": "Fotografia real em alta resolução com texturas ricas de pelos e iluminação.",
    },
    {
        "id": "pentagono",
        "name": SAMPLE_PENTAGONO_NAME,
        "title": "Exemplo 5: Pentágono PDI (1024×1024)",
        "button_label": "Pentágono (TIFF 8-bit)",
        "icon": "ACCOUNT_BALANCE",
        "description": "Benchmark monocromático de PDI com estruturas geométricas e pistas aéreas.",
    },
]


def get_sample_path(sample_name: str) -> Path:
    """Retorna o Path para a imagem de exemplo solicitada."""
    path = ASSETS_DIR / sample_name
    if not path.exists():
        raise FileNotFoundError(f"Imagem de exemplo não encontrada: {path}")
    return path


def load_sample_array(sample_name: str) -> np.ndarray:
    """Carrega e retorna a imagem de exemplo como array NumPy uint8 (RGB ou Cinza)."""
    path = get_sample_path(sample_name)
    with Image.open(path) as pil_img:
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
        img_array = np.array(pil_img)
    if img_array.ndim == 2:
        return img_array.astype(np.uint8)
    # Garante 3 canais RGB (descarta alpha se houver)
    return img_array[:, :, :3].astype(np.uint8)


def load_sample_bytes(sample_name: str) -> bytes:
    """Retorna os bytes da imagem de exemplo codificados em PNG."""
    path = get_sample_path(sample_name)
    if path.suffix.lower() == ".png":
        return path.read_bytes()
    # Converte outros formatos (JPG, TIFF) para PNG em memória para compatibilidade com a UI
    arr = load_sample_array(sample_name)
    pil_img = Image.fromarray(arr)
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

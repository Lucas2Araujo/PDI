"""
common.py — Utilitários e Helpers Compartilhados da Camada de Interface (UI).

Centraliza rotinas de conversão de imagens, Data URIs, registro de serviços Flet
e constantes de métodos de cinza e técnicas de quantização, evitando duplicação
de código entre as diferentes visualizações da aplicação.
"""

import base64
import io
from pathlib import Path
from typing import Any

import flet as ft
import numpy as np
from PIL import Image

from src.core.grayscale import GrayscaleMethod
from src.core.quantization import QuantizationTechnique


# ---------------------------------------------------------------------------
# Dicionários de Informações Didáticas e Opções
# ---------------------------------------------------------------------------

_GRAYSCALE_DETAILS: dict[GrayscaleMethod, dict[str, Any]] = {
    GrayscaleMethod.LUMINANCE: {
        "title": "Luminância Ponderada Fisiológica (ITU-R BT.601)",
        "formula": "Y = 0.299·R + 0.587·G + 0.114·B",
        "desc": (
            "Compensa a sensibilidade espectral do olho humano (máxima no verde, "
            "média no vermelho, mínima no azul). Padrão da indústria e do projeto."
        ),
        "icon": ft.Icons.VISIBILITY,
    },
    GrayscaleMethod.AVERAGE: {
        "title": "Média Aritmética Simples",
        "formula": "Y = (R + G + B) / 3",
        "desc": (
            "Média uniforme dos três canais RGB sem compensação fisiológica. "
            "Gera imagem monocromática direta."
        ),
        "icon": ft.Icons.CALCULATE,
    },
    GrayscaleMethod.CHANNEL_R: {
        "title": "Isolamento do Canal Vermelho (R)",
        "formula": "Matriz RGB pura: [R, 0, 0] (Tons de Vermelho)",
        "desc": (
            "Isola e exibe exclusivamente o canal vermelho em cores reais; "
            "quantização em níveis da cor vermelha."
        ),
        "icon": ft.Icons.LOOKS_ONE,
    },
    GrayscaleMethod.CHANNEL_G: {
        "title": "Isolamento do Canal Verde (G)",
        "formula": "Matriz RGB pura: [0, G, 0] (Tons de Verde)",
        "desc": (
            "Isola e exibe exclusivamente o canal verde em cores reais; "
            "quantização em níveis da cor verde."
        ),
        "icon": ft.Icons.LOOKS_TWO,
    },
    GrayscaleMethod.CHANNEL_B: {
        "title": "Isolamento do Canal Azul (B)",
        "formula": "Matriz RGB pura: [0, 0, B] (Tons de Azul)",
        "desc": (
            "Isola e exibe exclusivamente o canal azul em cores reais; "
            "quantização em níveis da cor azul."
        ),
        "icon": ft.Icons.LOOKS_3,
    },
}

_TECHNIQUE_OPTIONS: list[tuple[QuantizationTechnique | str, str]] = [
    (QuantizationTechnique.UNIFORM, "Modo 1: Quantização Uniforme (Intervalos Iguais)"),
    (QuantizationTechnique.KMEANS, "Modo 2: Quantização Não-Uniforme (K-Means Adaptativo)"),
    (QuantizationTechnique.HISTOGRAM, "Modo 3: Quantização por Histograma (Frequência/Quantis)"),
    ("BOTH", "Modo 4: Comparação Completa (Script Histograma Comparativo 2×3)"),
]


# ---------------------------------------------------------------------------
# Helpers de Compatibilidade e Conversão de Imagens
# ---------------------------------------------------------------------------


def _register_file_pickers(page: ft.Page, *pickers: ft.FilePicker) -> None:
    """Registra os FilePickers como serviços da página (Flet 0.86+)."""
    for picker in pickers:
        if hasattr(page, "_services") and hasattr(page._services, "register_service"):
            already = any(s is picker for s in page._services._services)
            if not already:
                page._services.register_service(picker)
        elif hasattr(page, "overlay") and picker not in page.overlay:
            page.overlay.append(picker)


def _ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    """Converte um array NumPy uint8 em bytes PNG em memória."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        pil_img = Image.fromarray(arr, mode="RGBA")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil_img = Image.fromarray(arr, mode="RGB")
    else:
        pil_img = Image.fromarray(arr, mode="L")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def _bytes_to_data_uri(image_bytes: bytes) -> str:
    """Converte bytes de imagem PNG em Data URI Base64 para exibição no Flet."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _read_image_file(path: Path | str) -> np.ndarray:
    """Carrega uma imagem do disco via PIL e converte para array NumPy RGB/L."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path_obj}")

    with Image.open(path_obj) as pil_img:
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
        return np.array(pil_img)


"""
histograma_comparativo.py — Script CLI para Comparação de Técnicas de Quantização.

Usa os módulos de histograma e quantização para gerar um gráfico comparativo
completo (2x3) com as imagens e histogramas das 3 versões: Original,
Quantização Uniforme e Quantização K-Means.

Uso:
    python scripts/histograma_comparativo.py <caminho_da_imagem> <bits>

Exemplo:
    python scripts/histograma_comparativo.py imagem.png 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np

from src.core.grayscale import GrayscaleMethod, to_grayscale
from src.core.histogram import calculate_metrics, generate_full_comparison_figure
from src.core.quantization import QuantizationTechnique, quantize


def main(image_path: str, bits: int) -> None:
    """
    Gera e salva o gráfico comparativo completo das técnicas de quantização.

    Args:
        image_path: Caminho para o arquivo de imagem de entrada.
        bits: Número de bits de quantização (1 a 8).
    """
    source = Path(image_path)
    if not source.is_file():
        print(f"Erro: arquivo '{image_path}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    n_tons = 2 ** bits
    print(f"Carregando imagem: {source.name}")
    with Image.open(source) as img:
        image_array = np.array(img)

    print("Convertendo para escala de cinza (ITU-R BT.601)...")
    gray = to_grayscale(image_array, method=GrayscaleMethod.LUMINANCE)
    print(f"  Formato cinza: {gray.shape} | Mín: {gray.min()} | Máx: {gray.max()}")

    print(f"\nAplicando quantizações com {bits} bits / {n_tons} tons...")
    print("  1/2 — Quantização Uniforme...")
    uniform = quantize(gray, bits=bits, technique=QuantizationTechnique.UNIFORM)

    print("  2/2 — Quantização K-Means (pode demorar alguns segundos)...")
    kmeans = quantize(gray, bits=bits, technique=QuantizationTechnique.KMEANS)

    print("\nCalculando métricas de qualidade...")
    metrics_unif = calculate_metrics(gray, uniform, bits)
    metrics_km = calculate_metrics(gray, kmeans, bits)

    print("\n--- Métricas de Qualidade ---")
    print(f"  Quantização Uniforme:")
    print(f"    MSE   : {metrics_unif.mse:.4f}")
    print(f"    PSNR  : {metrics_unif.psnr:.2f} dB")
    print(f"    Níveis: {metrics_unif.unique_levels}")
    print(f"  Quantização K-Means:")
    print(f"    MSE   : {metrics_km.mse:.4f}")
    print(f"    PSNR  : {metrics_km.psnr:.2f} dB")
    print(f"    Níveis: {metrics_km.unique_levels}")

    print("\nGerando gráfico comparativo...")
    figure_bytes = generate_full_comparison_figure(
        original=gray,
        uniform=uniform,
        kmeans=kmeans,
        bits=bits,
    )

    output_path = source.parent / f"{source.stem}_comparativo_{bits}bits.png"
    output_path.write_bytes(figure_bytes)
    print(f"\nGráfico salvo em: {output_path}")


def _parse_args() -> tuple[str, int]:
    """Valida e retorna os argumentos da linha de comando."""
    if len(sys.argv) != 3:
        print("Uso: python scripts/histograma_comparativo.py <caminho_da_imagem> <bits (1-8)>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        bits = int(sys.argv[2])
        if not (1 <= bits <= 8):
            raise ValueError
    except ValueError:
        print("Erro: <bits> deve ser um inteiro entre 1 e 8.", file=sys.stderr)
        sys.exit(1)

    return image_path, bits


if __name__ == "__main__":
    _image_path, _bits = _parse_args()
    main(_image_path, _bits)


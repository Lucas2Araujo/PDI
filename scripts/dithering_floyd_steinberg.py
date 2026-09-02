"""
dithering_floyd_steinberg.py — Script CLI para Quantização com Dithering (Floyd-Steinberg).

Uso:
    python scripts/dithering_floyd_steinberg.py <caminho_da_imagem> <bits>

Exemplo:
    python scripts/dithering_floyd_steinberg.py assets/lena_color.png 2

Técnica:
    Difusão de Erro Residual de Floyd-Steinberg (1976) para quantização digital de imagens.
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path para carregar os módulos de src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skimage import io

from src.core.grayscale import GrayscaleMethod, to_grayscale
from src.core.histogram import calculate_metrics
from src.core.quantization import (
    QuantizationTechnique,
    quantizacao_dithering_floyd_steinberg,
    quantize,
    technique_label,
)


def main(image_path: str, bits: int) -> None:
    """
    Executa a quantização por difusão de erro Floyd-Steinberg em uma imagem,
    calcula as métricas de qualidade e salva o resultado no disco.

    Args:
        image_path: Caminho para o arquivo de imagem de entrada.
        bits: Número de bits de quantização (1 a 8).
    """
    source = Path(image_path)
    if not source.is_file():
        print(f"Erro: arquivo '{image_path}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando imagem: {source.name}")
    image_array = io.imread(str(source))
    print(f"  Formato original : {image_array.shape} | dtype: {image_array.dtype}")

    print("Convertendo para escala de cinza (ITU-R BT.601)...")
    gray = to_grayscale(image_array, method=GrayscaleMethod.LUMINANCE)
    print(f"  Formato cinza    : {gray.shape} | Mín: {gray.min()} | Máx: {gray.max()}")

    n_tons = 2 ** bits
    print(f"\nAplicando {technique_label(QuantizationTechnique.FLOYD_STEINBERG)} ({bits} bits / {n_tons} tons)...")
    dithered = quantizacao_dithering_floyd_steinberg(gray, n_bits=bits)
    direct_unif = quantize(gray, bits=bits, technique=QuantizationTechnique.UNIFORM)

    # Cálculo comparativo de métricas
    m_dither = calculate_metrics(gray, dithered, bits)
    m_direct = calculate_metrics(gray, direct_unif, bits)

    print("\n--- Métricas de Qualidade e Fidelidade ---")
    print(f"  Quantização Direta (Uniforme):")
    print(f"    MSE           : {m_direct.mse:.4f}")
    print(f"    PSNR          : {m_direct.psnr:.2f} dB")
    print(f"    Níveis Únicos : {m_direct.unique_levels}")
    print(f"  Com Floyd-Steinberg (Dithering):")
    print(f"    MSE           : {m_dither.mse:.4f}")
    print(f"    PSNR          : {m_dither.psnr:.2f} dB")
    print(f"    Níveis Únicos : {m_dither.unique_levels}")

    output_path = source.parent / f"{source.stem}_floyd_steinberg_{bits}bits.png"
    io.imsave(str(output_path), dithered)

    print(f"\nImagem resultante salva em : {output_path}")
    print(f"  Mín / Máx               : {dithered.min()} / {dithered.max()}")


def _parse_args() -> tuple[str, int]:
    """Valida e retorna os argumentos da linha de comando."""
    if len(sys.argv) != 3:
        print("Uso: python scripts/dithering_floyd_steinberg.py <caminho_da_imagem> <bits (1-8)>")
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

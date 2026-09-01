"""
uniforme.py — Script CLI para Quantização Uniforme de Imagem.

Uso:
    python scripts/uniforme.py <caminho_da_imagem> <bits>

Exemplo:
    python scripts/uniforme.py imagem.png 4

Técnica:
    Quantização Uniforme — divide o intervalo [0, 255] em 2^bits intervalos iguais.
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path para que os módulos src/ sejam encontrados
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skimage import io

from src.core.grayscale import GrayscaleMethod, to_grayscale
from src.core.quantization import QuantizationTechnique, quantize, technique_label


def main(image_path: str, bits: int) -> None:
    """
    Executa a quantização uniforme em uma imagem e salva o resultado.

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
    print(f"\nAplicando {technique_label(QuantizationTechnique.UNIFORM)} ({bits} bits / {n_tons} tons)...")
    quantized = quantize(gray, bits=bits, technique=QuantizationTechnique.UNIFORM)

    output_path = source.parent / f"{source.stem}_uniforme_{bits}bits.png"
    io.imsave(str(output_path), quantized)

    print(f"\nImagem salva em : {output_path}")
    print(f"  Níveis únicos  : {len(set(quantized.ravel()))}")
    print(f"  Mín / Máx      : {quantized.min()} / {quantized.max()}")


def _parse_args() -> tuple[str, int]:
    """Valida e retorna os argumentos da linha de comando."""
    if len(sys.argv) != 3:
        print("Uso: python scripts/uniforme.py <caminho_da_imagem> <bits (1-8)>")
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


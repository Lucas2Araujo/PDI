"""
uniforme.py — Script de Quantização Uniforme.
"""

import sys
from pathlib import Path

from skimage import color, io, util


def main(image_path: str = "teste.png", n_niveis: int | None = None) -> None:
    source = Path(image_path)
    if not source.exists():
        print(f"Erro: Arquivo '{image_path}' não encontrado.", file=sys.stderr)
        return

    imagem_rgb = io.imread(str(source))
    imagem_Tcinza = color.rgb2gray(imagem_rgb)

    print("Formato da imagem RGB é : ", imagem_rgb.shape)
    print("Tipo de dado da imagem RGB : ", imagem_rgb.dtype)
    print("Formato da imagem em tons de cinza é : ", imagem_Tcinza.shape)
    print("Tipo de dado da imagem em tons de cinza : ", imagem_Tcinza.dtype)
    print("Piso e teto dos valores na imagem em tons de cinza:", imagem_Tcinza.min(), imagem_Tcinza.max())

    imagem_uint8 = util.img_as_ubyte(imagem_Tcinza)

    if n_niveis is None:
        while True:
            try:
                n_niveis = int(input("Qual nível de bits você deseja? "))
                if 1 <= n_niveis <= 8:
                    break
                print("Entrada inválida, por favor escreva um valor entre 1 e 8 (diferente de zero)")
            except ValueError:
                print("Entrada inválida, por favor digite apenas números inteiros de 1 à 8.")

    print(f"Nível escolhido de {n_niveis} bits.")
    n_tons = 2 ** n_niveis
    passo = 256 // n_tons
    indices = imagem_uint8 // passo
    fator_escala = 255 // (n_tons - 1)
    imagem_QuantizadaUniforme = indices * fator_escala

    nome_saida = f"teste_quantizanum{n_niveis}v2.png"
    io.imsave(nome_saida, imagem_QuantizadaUniforme)
    print("Valor máximo e mínimo de bits : ", imagem_QuantizadaUniforme.max(), imagem_QuantizadaUniforme.min())


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else "teste.png"
    _bits = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(_path, _bits)
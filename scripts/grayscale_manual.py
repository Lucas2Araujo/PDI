"""
grayscale_manual.py — Script CLI para Conversão Manual Pixel a Pixel para Tons de Cinza.

Demonstração didática da fórmula de luminância perceptual ITU-R BT.601
implementada manualmente com Pillow, sem bibliotecas de processamento de imagem.

Uso:
    python scripts/grayscale_manual.py <caminho_da_imagem>

Exemplo:
    python scripts/grayscale_manual.py imagem.png
"""

import sys
from pathlib import Path

from PIL import Image


def to_grayscale_manual(image_rgb: Image.Image) -> Image.Image:
    """
    Converte uma imagem RGB para escala de cinza pixel a pixel.

    Aplica a fórmula de luminância perceptual ITU-R BT.601 (NTSC):
        Y = 0.2989·R + 0.5870·G + 0.1140·B

    Os coeficientes refletem a sensibilidade diferenciada do olho humano
    para cada comprimento de onda:
      - Verde (~58.7%): maior sensibilidade
      - Vermelho (~29.9%): sensibilidade média
      - Azul (~11.4%): menor sensibilidade

    Args:
        image_rgb: Imagem Pillow no modo RGB.

    Returns:
        Nova imagem Pillow no modo 'L' (8-bit grayscale).
    """
    largura, altura = image_rgb.size
    imagem_cinza = Image.new("L", (largura, altura))

    pixels_rgb = image_rgb.load()
    pixels_cinza = imagem_cinza.load()

    for y in range(altura):
        for x in range(largura):
            r, g, b = pixels_rgb[x, y]
            intensidade = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            # Garante que o valor permanece estritamente em [0, 255]
            pixels_cinza[x, y] = max(0, min(255, intensidade))

    return imagem_cinza


def main(image_path: str) -> None:
    """
    Executa a conversão manual para tons de cinza e salva o resultado.

    Args:
        image_path: Caminho para o arquivo de imagem de entrada.
    """
    source = Path(image_path)
    if not source.is_file():
        print(f"Erro: arquivo '{image_path}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando imagem: {source.name}")
    imagem_original = Image.open(str(source))

    # Garante que a imagem está no formato RGB (descarta canal Alpha se houver)
    imagem_rgb = imagem_original.convert("RGB")
    largura, altura = imagem_rgb.size

    print(f"  Modo de cor   : {imagem_rgb.mode}")
    print(f"  Dimensões     : {largura} x {altura} pixels")
    print(f"  Total pixels  : {largura * altura:,}")

    print("\nConvertendo para escala de cinza (varredura manual pixel a pixel)...")
    imagem_cinza = to_grayscale_manual(imagem_rgb)

    output_path = source.parent / f"{source.stem}_cinza_manual.jpg"
    imagem_cinza.save(str(output_path))

    # Avalia piso e teto dos valores calculados usando getextrema() — eficiente e sem depreciação
    valor_min, valor_max = imagem_cinza.getextrema()

    print(f"\nImagem salva em : {output_path}")
    print(f"  Modo de cor   : {imagem_cinza.mode} (8-bit grayscale)")
    print(f"  Mín / Máx     : {valor_min} / {valor_max}")


def _parse_args() -> str:
    """Valida e retorna o argumento da linha de comando."""
    if len(sys.argv) != 2:
        print("Uso: python scripts/grayscale_manual.py <caminho_da_imagem>")
        sys.exit(1)
    return sys.argv[1]


if __name__ == "__main__":
    _image_path = _parse_args()
    main(_image_path)

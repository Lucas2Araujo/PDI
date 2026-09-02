"""
namão.py — Conversão manual pixel a pixel para tons de cinza via ITU-R BT.601.
"""

import sys
from pathlib import Path
from PIL import Image


def main(image_path: str = "teste.png") -> None:
    source = Path(image_path)
    if not source.exists():
        print(f"Erro: Arquivo '{image_path}' não encontrado.", file=sys.stderr)
        return

    # 1. Carregar a imagem RGB
    imagem_original = Image.open(str(source))
    imagem_rgb = imagem_original.convert("RGB")

    largura, altura = imagem_rgb.size
    print("--- IMAGEM ORIGINAL (RGB) ---")
    print(f"Dimensões (Largura x Altura): {largura} x {altura}")
    print(f"Modo de cor: {imagem_rgb.mode}")
    print(f"Quantidade total de pixels: {largura * altura}")

    # 2. Criar uma nova imagem para tons de cinza
    imagem_cinza = Image.new("L", (largura, altura))
    pixels_rgb = imagem_rgb.load()
    pixels_cinza = imagem_cinza.load()

    valor_min = 255
    valor_max = 0

    # 3. Varredura manual (Pixel por Pixel)
    for y in range(altura):
        for x in range(largura):
            r, g, b = pixels_rgb[x, y]
            intensidade = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            intensidade = max(0, min(255, intensidade))

            if intensidade < valor_min:
                valor_min = intensidade
            if intensidade > valor_max:
                valor_max = intensidade

            pixels_cinza[x, y] = intensidade

    print("\n--- IMAGEM RESULTANTE (TONS DE CINZA - NA MÃO) ---")
    print(f"Dimensões (Largura x Altura): {imagem_cinza.size[0]} x {imagem_cinza.size[1]}")
    print(f"Modo de cor: {imagem_cinza.mode} (8-bit grayscale)")
    print(f"Piso e teto dos valores na imagem em tons de cinza: min={valor_min}, max={valor_max}")

    # 4. Salvar a imagem resultante em disco
    nome_arquivo_saida = "teste_cinza_namao.jpg"
    imagem_cinza.save(nome_arquivo_saida)
    print(f"\nImagem salva com sucesso em: {nome_arquivo_saida}")

    imagem_original.close()
    imagem_rgb.close()
    imagem_cinza.close()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else "teste.png"
    main(_path)

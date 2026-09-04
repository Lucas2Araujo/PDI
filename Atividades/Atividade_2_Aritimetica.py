"""
UNIVERSIDADE FEDERAL DO MARANHÃO (UFMA)
Processamento Digital de Imagens (PDI)
Trabalho: Operações Aritméticas em Imagens (Escalar e Imagem a Imagem)
Aluno : Lucas Araújo Dominici
- 1. Adição (+)
- 2. Subtração (-)
- 3. Multiplicação (x)
- 4. Divisão (/)
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data

EPSILON = 1e-5

def adicao_esc(img, c):
    """Soma um escalar à imagem com saturação em 255."""
    resultado = img.astype(np.uint16) + c
    return np.clip(resultado, 0, 255).astype(np.uint8)


def adicao_img(img1, img2):
    """Soma duas imagens pixel a pixel com saturação."""
    resultado = img1.astype(np.uint16) + img2.astype(np.uint16)
    return np.clip(resultado, 0, 255).astype(np.uint8)


def subtracao_esc(img, c):
    """Subtrai um escalar da imagem com corte inferior em 0."""
    resultado = img.astype(np.int16) - c
    return np.clip(resultado, 0, 255).astype(np.uint8)


def subtracao_abs(img1, img2):
    """Diferença absoluta pixel a pixel entre duas imagens."""
    resultado = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    return np.clip(resultado, 0, 255).astype(np.uint8)


def multiplicacao_esc(img, c):
    """Ajuste de contraste/ganho com arredondamento."""
    resultado = img.astype(np.float32) * c
    return np.clip(resultado.round(), 0, 255).astype(np.uint8)


def multiplicacao_img(img1, img2):
    """Multiplicação ponderada elemento a elemento na escala [0, 1]."""
    norm_1 = img1.astype(np.float32) / 255.0
    norm_2 = img2.astype(np.float32) / 255.0
    resultado = (norm_1 * norm_2) * 255.0
    return np.clip(resultado.round(), 0, 255).astype(np.uint8)


def divisao_esc(img, c):
    """Divisão por escalar não nulo com arredondamento."""
    if c == 0:
        raise ValueError("O divisor escalar não pode ser zero.")
    resultado = img.astype(np.float32) / c
    return np.clip(resultado.round(), 0, 255).astype(np.uint8)


def divisao_img(img1, img2, eps=EPSILON):
    """Razão entre imagens com corte no percentil 99.5 para preservar contraste dinâmico."""
    resultado = img1.astype(np.float32) / (img2.astype(np.float32) + eps)
    
    val_min = resultado.min()
    val_max = np.percentile(resultado, 99.5)

    denominador = val_max - val_min
    if denominador == 0:
        return np.zeros_like(resultado, dtype=np.uint8)

    resultado_norm = ((resultado - val_min) / denominador) * 255.0
    return np.clip(resultado_norm.round(), 0, 255).astype(np.uint8)


def obter_escalar(operacao):
    """Obtém e valida o escalar de entrada com avisos informativos."""
    while True:
        try:
            if operacao in ["1", "2"]:
                return float(input("\nInforme o valor do escalar (ex: 50): ").strip())

            elif operacao == "3":
                c = float(input("\nInforme o fator multiplicador (ex: 1.5, 0.5): ").strip())
                if c <= 0:
                    print("[!] Aviso: Multiplicar por um valor menor ou igual a zero fará a imagem sumir (preto puro).")
                    confirma = input("Deseja continuar mesmo assim? (s/n): ").strip().lower()
                    if confirma != "s":
                        continue
                return c

            elif operacao == "4":
                c = float(input("\nInforme o divisor escalar (diferente de zero, ex: 2.0): ").strip())
                if c == 0:
                    print("[!] Erro: O divisor não pode ser zero.")
                    continue
                return c

        except ValueError:
            print("[!] Entrada inválida. Digite um número real válido.")


def plotar_escalar(img_orig, img_res, titulo_res):
    """Exibe painel 2x2 para operações com escalar (Original e Resultado)."""
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img_orig, cmap="gray", vmin=0, vmax=255)
    plt.title("Original: Cameraman")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img_res, cmap="gray", vmin=0, vmax=255)
    plt.title(titulo_res)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.hist(img_orig.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title("Histograma: Original")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Qtd. Pixels")
    plt.xlim(0, 255)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 2, 4)
    plt.hist(img_res.ravel(), bins=256, range=(0, 256), color="steelblue")
    plt.title("Histograma: Resultado")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Qtd. Pixels")
    plt.xlim(0, 255)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def plotar_duas_imagens(img1, img2, img_res, titulo_res):
    """Exibe painel 2x3 para operações entre duas imagens."""
    plt.figure(figsize=(15, 8))

    # Imagens
    plt.subplot(2, 3, 1)
    plt.imshow(img1, cmap="gray", vmin=0, vmax=255)
    plt.title("Imagem 1: Cameraman")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(img2, cmap="gray", vmin=0, vmax=255)
    plt.title("Imagem 2: Moon")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(img_res, cmap="gray", vmin=0, vmax=255)
    plt.title(titulo_res)
    plt.axis("off")

    # Histogramas
    plt.subplot(2, 3, 4)
    plt.hist(img1.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title("Histograma: Cameraman")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Qtd. Pixels")
    plt.xlim(0, 255)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 3, 5)
    plt.hist(img2.ravel(), bins=256, range=(0, 256), color="darkorange")
    plt.title("Histograma: Moon")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Qtd. Pixels")
    plt.xlim(0, 255)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 3, 6)
    plt.hist(img_res.ravel(), bins=256, range=(0, 256), color="steelblue")
    plt.title("Histograma: Resultado")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Qtd. Pixels")
    plt.xlim(0, 255)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("      OPERAÇÕES ARITMÉTICAS EM IMAGENS      ")
    print("=" * 60)

    img1 = data.camera()
    img2 = data.moon()
    print(f"[OK] Imagem base 1: 'Cameraman' ({img1.shape[1]}x{img1.shape[0]})")
    print(f"[OK] Imagem base 2: 'Moon'      ({img2.shape[1]}x{img2.shape[0]})")

    print("\n" + "-" * 60)
    print("              SELEÇÃO DA OPERAÇÃO")
    print("-" * 60)
    print("1 - Adição (+)")
    print("2 - Subtração (-)")
    print("3 - Multiplicação (x)")
    print("4 - Divisão (/)")

    while True:
        opcao_op = input("\nEscolha a operação (1, 2, 3 ou 4): ").strip()
        if opcao_op in ["1", "2", "3", "4"]:
            break
        print("[!] Opção inválida. Digite 1, 2, 3 ou 4.")

    print("\n" + "-" * 60)
    print("                 MODO DE OPERAÇÃO")
    print("-" * 60)
    print("1 - Imagem com Escalar (Constante)")
    print("2 - Imagem com Imagem (Cameraman com Moon)")

    while True:
        modo = input("\nEscolha o modo (1 ou 2): ").strip()
        if modo in ["1", "2"]:
            break
        print("[!] Opção inválida. Digite 1 ou 2.")

    nomes_op = {
        "1": "Adição",
        "2": "Subtração",
        "3": "Multiplicação",
        "4": "Divisão",
    }
    nome_base = nomes_op[opcao_op]

    print("\nProcessando operação...")

    if modo == "1":
        c = obter_escalar(opcao_op)
        if opcao_op == "1":
            img_resultado = adicao_esc(img1, c)
        elif opcao_op == "2":
            img_resultado = subtracao_esc(img1, c)
        elif opcao_op == "3":
            img_resultado = multiplicacao_esc(img1, c)
        else:
            img_resultado = divisao_esc(img1, c)

        nome_saida = f"saida_{nome_base}_escalar_{c}.png"
        titulo_resultado = f"Resultado: {nome_base} (c = {c})"
        Image.fromarray(img_resultado).save(nome_saida)
        print(f"\n[+] Imagem resultante salva com sucesso como: '{nome_saida}'")

        plotar_escalar(img1, img_resultado, titulo_resultado)

    else:
        if opcao_op == "1":
            img_resultado = adicao_img(img1, img2)
        elif opcao_op == "2":
            img_resultado = subtracao_abs(img1, img2)
        elif opcao_op == "3":
            img_resultado = multiplicacao_img(img1, img2)
        else:
            img_resultado = divisao_img(img1, img2)

        nome_saida = f"saida_{nome_base}_duas_imagens.png"
        titulo_resultado = f"Resultado: {nome_base} (Cameraman & Moon)"
        Image.fromarray(img_resultado).save(nome_saida)
        print(f"\n[+] Imagem resultante salva com sucesso como: '{nome_saida}'")

        plotar_duas_imagens(img1, img2, img_resultado, titulo_resultado)


if __name__ == "__main__":
    main()
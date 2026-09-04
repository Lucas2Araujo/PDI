"""
UNIVERSIDADE FEDERAL DO MARANHÃO (UFMA)
Processamento Digital de Imagens (PDI)
Trabalho: Operações Lógicas
Aluno : Lucas Araújo Dominici
- 1. And
- 2. Or
- 3. Not
- 4. Xor
- 5. Sub
"""

"""
UNIVERSIDADE FEDERAL DO MARANHÃO (UFMA)
Processamento Digital de Imagens (PDI)
Trabalho: Operações Lógicas em Imagens Binárias Sintéticas
Aluno : Lucas Araújo Dominici
- 1. AND (Interseção)
- 2. OR (União)
- 3. NOT (Inversão / Complemento)
- 4. XOR (Diferença Simétrica)
- 5. SUB (Diferença de Conjuntos: A AND NOT B)
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def criar_formas_geometricas(tamanho=512):
    """
    Gera duas imagens binárias (booleanas) de 512x512:
    - Imagem A: Círculo centralizado à esquerda.
    - Imagem B: Quadrado / retângulo centralizado à direita (com sobreposição).
    """
    y, x = np.ogrid[:tamanho, :tamanho]

    # Imagem A: Círculo (centro em x=220, y=256, raio=140)
    centro_x, centro_y, raio = 220, 256, 140
    img_a = (x - centro_x) ** 2 + (y - centro_y) ** 2 <= raio**2

    # Imagem B: Retângulo (x de 200 a 440, y de 150 a 390)
    img_b = (x >= 200) & (x <= 440) & (y >= 150) & (y <= 390)

    return img_a, img_b


def bin_para_uint8(img_bin):
    """Converte matriz booleana para uint8 (0 e 255) para exportação."""
    return img_bin.astype(np.uint8) * 255


# --- Operações Lógicas Elemento a Elemento ---


def op_and(b1, b2):
    """Interseção lógica (A AND B)."""
    return b1 & b2


def op_or(b1, b2):
    """União lógica (A OR B)."""
    return b1 | b2


def op_not(b):
    """Complemento / inversão lógica (~A)."""
    return ~b


def op_xor(b1, b2):
    """Diferença simétrica (A XOR B)."""
    return b1 ^ b2


def op_sub(b1, b2):
    """Diferença de conjuntos: pixels exclusivos de A (A AND NOT B)."""
    return b1 & (~b2)


# --- Funções de Plotagem ---


def plotar_unaria(b_orig, b_res, titulo_res):
    """Exibe painel comparativo para a operação NOT."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(b_orig, cmap="gray")
    axes[0].set_title("Entrada: Forma A (Círculo)")
    axes[0].axis("off")

    axes[1].imshow(b_res, cmap="gray")
    axes[1].set_title(titulo_res)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plotar_binarias(b1, b2, b_res, titulo_res):
    """Exibe painel 1x3 clássico para operações de duas entradas."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(b1, cmap="gray")
    axes[0].set_title("Forma A (Círculo)")
    axes[0].axis("off")

    axes[1].imshow(b2, cmap="gray")
    axes[1].set_title("Forma B (Quadrado)")
    axes[1].axis("off")

    axes[2].imshow(b_res, cmap="gray")
    axes[2].set_title(titulo_res)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("      OPERAÇÕES LÓGICAS EM IMAGENS BINÁRIAS      ")
    print("=" * 60)

    # Geração direta das matrizes booleanas sem conversão por threshold
    img_a, img_b = criar_formas_geometricas(tamanho=512)

    print(f"[OK] Imagem Binária A gerada: Círculo ({img_a.shape[1]}x{img_a.shape[0]})")
    print(f"[OK] Imagem Binária B gerada: Retângulo ({img_b.shape[1]}x{img_b.shape[0]})")
    print(f"[OK] Tipagem nativa: {img_a.dtype} (Valores: True / False)")

    print("\n" + "-" * 60)
    print("              SELEÇÃO DA OPERAÇÃO LÓGICA")
    print("-" * 60)
    print("1 - AND (A e B - Interseção)")
    print("2 - OR  (A ou B - União)")
    print("3 - NOT (Inversão de A)")
    print("4 - XOR (A ou exclusivo B - Diferença Simétrica)")
    print("5 - SUB (Diferença: A - B => A e NOT B)")

    while True:
        opcao = input("\nEscolha a operação (1 a 5): ").strip()
        if opcao in ["1", "2", "3", "4", "5"]:
            break
        print("[!] Opção inválida. Digite um número de 1 a 5.")

    nomes = {
        "1": ("AND (Interseção)", "A_and_B"),
        "2": ("OR (União)", "A_or_B"),
        "3": ("NOT (Inversão de A)", "NOT_A"),
        "4": ("XOR (Diferença Simétrica)", "A_xor_B"),
        "5": ("SUB (Diferença A - B)", "A_sub_B"),
    }
    nome_exibicao, nome_slug = nomes[opcao]

    print(f"\nProcessando operação {nome_exibicao}...")

    if opcao == "3":
        b_resultado = op_not(img_a)
        res_uint8 = bin_para_uint8(b_resultado)

        nome_saida = f"saida_logica_{nome_slug}.png"
        Image.fromarray(res_uint8).save(nome_saida)
        print(f"\n[+] Imagem resultante salva com sucesso como: '{nome_saida}'")

        plotar_unaria(img_a, b_resultado, f"Resultado: {nome_exibicao}")
    else:
        if opcao == "1":
            b_resultado = op_and(img_a, img_b)
        elif opcao == "2":
            b_resultado = op_or(img_a, img_b)
        elif opcao == "4":
            b_resultado = op_xor(img_a, img_b)
        elif opcao == "5":
            b_resultado = op_sub(img_a, img_b)

        res_uint8 = bin_para_uint8(b_resultado)

        nome_saida = f"saida_logica_{nome_slug}.png"
        Image.fromarray(res_uint8).save(nome_saida)
        print(f"\n[+] Imagem resultante salva com sucesso como: '{nome_saida}'")

        plotar_binarias(img_a, img_b, b_resultado, f"Resultado: {nome_exibicao}")


if __name__ == "__main__":
    main()
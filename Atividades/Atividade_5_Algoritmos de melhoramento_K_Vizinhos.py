"""
UNIVERSIDADE FEDERAL DO MARANHÃO (UFMA)
Processamento Digital de Imagens (PDI)
Trabalho: Algoritmos de Melhoramento Espacial e Difusão de Erro
Aluno : Lucas Araújo Dominici
- 1. Filtro da Média (Suavização Linear)
- 2. Filtro da Mediana (Suavização Não-Linear / Sal e Pimenta)
- 3. Unsharp Masking (Realce de Bordas / Aguçamento)
- 4. Halftoning com Dithering de Floyd-Steinberg (Difusão de Erro)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, data

# Imagens de teste padrão do scikit-image
img1 = data.camera()
img2 = data.moon()
img3 = (color.rgb2gray(data.cat()) * 255).astype(np.uint8)
img4 = (color.rgb2gray(data.astronaut()) * 255).astype(np.uint8)
img5 = data.cell()
img6 = data.coins()
img7 = data.page()


def carregar_imagens(nome="camera"):
    if nome == "teste":
        caminho_teste = "teste.png"
        if os.path.exists(caminho_teste):
            img_pil = Image.open(caminho_teste).convert("L")
            return np.array(img_pil, dtype=np.uint8)
        print("\n[!] Arquivo 'teste.png' não encontrado no diretório atual.")
        print("[!] Carregando 'camera' como fallback padrão.")
        return img1.astype(np.uint8)

    banco = {
        "camera": img1,
        "moon": img2,
        "cat": img3,
        "astronaut": img4,
        "cell": img5,
        "coins": img6,
        "page": img7,
    }
    return banco.get(nome, img1).astype(np.uint8)


# --- 1. Filtro da Média (Filtro Box / Convolução 2D) ---
def filtro_media(img, tamanho_janela=3):
    pad = tamanho_janela // 2
    img_pad = np.pad(img.astype(np.float64), pad, mode="reflect")
    h, w = img.shape
    saida = np.zeros((h, w), dtype=np.float64)

    peso = 1.0 / (tamanho_janela * tamanho_janela)

    for i in range(h):
        for j in range(w):
            vizinhanca = img_pad[i : i + tamanho_janela, j : j + tamanho_janela]
            saida[i, j] = np.sum(vizinhanca) * peso

    return np.clip(saida.round(), 0, 255).astype(np.uint8)


# --- 2. Filtro da Mediana (Estatística de Ordem) ---
def filtro_mediana(img, tamanho_janela=3):
    pad = tamanho_janela // 2
    img_pad = np.pad(img, pad, mode="reflect")
    h, w = img.shape
    saida = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            janela = img_pad[i : i + tamanho_janela, j : j + tamanho_janela]
            saida[i, j] = np.median(janela)

    return saida


# --- 3. Unsharp Masking (Realce com Máscara de Desfoque) ---
def unsharp_masking(img, k=1.0, tamanho_janela=5):
    img_f = img.astype(np.float64)
    img_suave = filtro_media(img, tamanho_janela=tamanho_janela).astype(np.float64)

    # Máscara de altas frequências
    mascara = img_f - img_suave

    # Soma ponderada com a imagem original
    img_realcada = img_f + (k * mascara)

    return np.clip(img_realcada.round(), 0, 255).astype(np.uint8)


# --- 4. Algoritmo de Floyd-Steinberg (Difusão de Erro) ---
def floyd_steinberg_dithering(img, n_niveis=2):
    h, w = img.shape
    img_diff = img.astype(np.float64).copy()
    
    # Degraus de quantização (ex: para 4 níveis -> [0, 85, 170, 255])
    degraus = np.linspace(0, 255, n_niveis)

    for y in range(h):
        for x in range(w):
            antigo_pixel = img_diff[y, x]
            
            # Encontra o nível mais próximo
            idx = np.argmin(np.abs(degraus - antigo_pixel))
            novo_pixel = degraus[idx]
            img_diff[y, x] = novo_pixel

            # Calcula o erro residual
            erro = antigo_pixel - novo_pixel

            # Espalha o erro para os vizinhos ainda não processados
            if x + 1 < w:
                img_diff[y, x + 1] += erro * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    img_diff[y + 1, x - 1] += erro * (3.0 / 16.0)
                img_diff[y + 1, x] += erro * (5.0 / 16.0)
                if x + 1 < w:
                    img_diff[y + 1, x + 1] += erro * (1.0 / 16.0)

    return np.clip(img_diff, 0, 255).astype(np.uint8)


# --- Funções Auxiliares e Plotagem ---
def adicionar_ruido_sal_pimenta(img, prob=0.05):
    ruidosa = img.copy()
    rnd = np.random.rand(*img.shape)
    ruidosa[rnd < (prob / 2.0)] = 0
    ruidosa[(rnd >= (prob / 2.0)) & (rnd < prob)] = 255
    return ruidosa


def plotar_comparacao_2x2(img_esq, img_dir, titulo_esq, titulo_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Imagem da Esquerda
    axes[0, 0].imshow(img_esq, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"{titulo_esq} [min={np.min(img_esq)}, max={np.max(img_esq)}]")
    axes[0, 0].axis("off")

    # Histograma da Esquerda
    axes[1, 0].hist(img_esq.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.8)
    axes[1, 0].set_title(f"Histograma: {titulo_esq}")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Imagem da Direita
    axes[0, 1].imshow(img_dir, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"{titulo_dir} [min={np.min(img_dir)}, max={np.max(img_dir)}]")
    axes[0, 1].axis("off")

    # Histograma da Direita
    axes[1, 1].hist(img_dir.ravel(), bins=256, range=(0, 256), color="steelblue", alpha=0.8)
    axes[1, 1].set_title(f"Histograma: {titulo_dir}")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def selecionar_imagem():
    imagens_disponiveis = {
        "1": ("camera", "data.camera() (Padrão)"),
        "2": ("moon", "data.moon() (Cráteras e texturas)"),
        "3": ("page", "data.page() (Texto / Documento)"),
        "4": ("coins", "data.coins() (Bordas contrastadas)"),
        "5": ("cat", "data.cat() (Pelo e altas frequências)"),
        "6": ("astronaut", "data.astronaut() (Transições suaves)"),
        "7": ("cell", "data.cell() (Imagem biomédica)"),
        "8": ("teste", "Carregar 'teste.png' do diretório atual"),
    }

    print("\n" + "-" * 60)
    print("              SELEÇÃO DA IMAGEM DE TESTE              ")
    print("-" * 60)
    for k, v in imagens_disponiveis.items():
        print(f"{k} - {v[0]} ({v[1]})")

    while True:
        op = input("\nEscolha a imagem (1 a 8) [Padrão: 1]: ").strip() or "1"
        if op in imagens_disponiveis:
            nome_chave = imagens_disponiveis[op][0]
            img = carregar_imagens(nome_chave)
            print(f"[+] Imagem '{nome_chave}' carregada com resolução {img.shape}.")
            return nome_chave, img
        print("[!] Opção inválida. Digite um número de 1 a 8.")


def main():
    print("=" * 60)
    print("   ALGORITMOS DE FILTRAGEM ESPACIAL E DIFUSÃO DE ERRO   ")
    print("=" * 60)

    nome_chave, img_original = selecionar_imagem()

    # Opção para injetar ruído impulsivo (ideal para testar Mediana vs Média)
    print("\n" + "-" * 60)
    print("                CONFIGURAÇÃO DE RUÍDO                 ")
    print("-" * 60)
    print("0 - Manter imagem original íntegra")
    print("1 - Injetar ruído Sal e Pimenta (Recomendado p/ testar Mediana/Média)")
    op_ruido = input("\nEscolha (0 ou 1) [Padrão: 0]: ").strip() or "0"

    if op_ruido == "1":
        p_str = input("Informe a probabilidade de ruído (ex: 0.05 para 5%) [0.05]: ").strip()
        p = float(p_str) if p_str else 0.05
        img_trabalho = adicionar_ruido_sal_pimenta(img_original, prob=p)
        print(f"[!] {p * 100:.1f}% de ruído sal e pimenta injetado.")
    else:
        img_trabalho = img_original.copy()

    print("\n" + "-" * 60)
    print("              SELEÇÃO DO ALGORITMO                   ")
    print("-" * 60)
    print("1 - Filtro da Média (Box Filter)")
    print("2 - Filtro da Mediana")
    print("3 - Unsharp Masking (Realce de Bordas / Aguçamento)")
    print("4 - Dithering de Floyd-Steinberg (Halftoning / Difusão de Erro)")

    while True:
        opcao = input("\nEscolha a operação (1 a 4): ").strip()
        if opcao in ["1", "2", "3", "4"]:
            break
        print("[!] Opção inválida. Digite um número de 1 a 4.")

    if opcao == "1":
        slug = "media"
        nome_op = "Filtro da Média"
        tam_str = input("Tamanho da máscara ímpar k (ex: 3 para 3x3, 5 para 5x5) [3]: ").strip()
        k_size = int(tam_str) if tam_str else 3
        if k_size % 2 == 0:
            k_size += 1
            print(f"[!] Ajustado para {k_size} (a janela precisa ser ímpar).")
        res_img = filtro_media(img_trabalho, tamanho_janela=k_size)
        nome_op = f"Filtro Média ({k_size}x{k_size})"

    elif opcao == "2":
        slug = "mediana"
        tam_str = input("Tamanho da máscara ímpar k (ex: 3 para 3x3, 5 para 5x5) [3]: ").strip()
        k_size = int(tam_str) if tam_str else 3
        if k_size % 2 == 0:
            k_size += 1
            print(f"[!] Ajustado para {k_size} (a janela precisa ser ímpar).")
        res_img = filtro_mediana(img_trabalho, tamanho_janela=k_size)
        nome_op = f"Filtro Mediana ({k_size}x{k_size})"

    elif opcao == "3":
        slug = "unsharp"
        k_str = input("Fator de ganho k (> 0; k=1 padrão, k>1 highboost) [1.2]: ").strip()
        k_val = float(k_str) if k_str else 1.2
        tam_str = input("Tamanho da máscara de suavização [3]: ").strip()
        k_size = int(tam_str) if tam_str else 3
        res_img = unsharp_masking(img_trabalho, k=k_val, tamanho_janela=k_size)
        nome_op = f"Unsharp Masking (k={k_val}, janela={k_size}x{k_size})"

    else:
        slug = "floyd_steinberg"
        nome_op = "Dithering de Floyd-Steinberg"
        n_str = input("Quantidade de níveis de quantização (ex: 2, 4, 8) [2]: ").strip()
        n_niveis = int(n_str) if n_str else 2
        res_img = floyd_steinberg_dithering(img_trabalho, n_niveis=n_niveis)
        nome_op = f"Floyd-Steinberg ({n_niveis} níveis)"
        
    # Exportação e plotagem final
    nome_saida = f"saida_{slug}_{nome_chave}.png"
    Image.fromarray(res_img).save(nome_saida)
    print(f"\n[+] Imagem resultante salva com sucesso como: '{nome_saida}'")

    print("[i] Exibindo painel comparativo...")
    plotar_comparacao_2x2(img_trabalho, res_img, "Entrada", f"Resultado: {nome_op}")


if __name__ == "__main__":
    main()
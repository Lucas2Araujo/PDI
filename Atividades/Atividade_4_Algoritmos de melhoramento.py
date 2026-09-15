"""
UNIVERSIDADE FEDERAL DO MARANHÃO (UFMA)
Processamento Digital de Imagens (PDI)
Trabalho: Algoritmos de Melhoramento
Aluno : Lucas Araújo Dominici
- 1. Algoritmo de Stretching
- 2. Linear Mapping
- 3. Transformação Logarítmica
- 4. Ajuste de Contraste e Brilho
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, data

# --- Carregamento das Imagens do skimage.data ---
img1 = data.camera()
img2 = data.moon()
img3 = color.rgb2gray(data.cat()) * 255
img4 = color.rgb2gray(data.astronaut()) * 255
img5 = data.cell()
img6 = data.coins()
img7 = data.page()


def carregar_imagens(nome="camera"):
    """Carrega imagem do banco embutido ou imagem externa 'teste.png'."""
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


# --- 1. Algoritmo de Stretching ---
def stretching(img_f, r_min, r_max):
    """
    Alongamento de Contraste (Contrast Stretching / Normalização Min-Max):
    Expande a faixa dinâmica atual [r_min, r_max] para a faixa total [0, 255].
    Fórmula: s = ((r - r_min) / (r_max - r_min)) * 255
    """
    if r_max == r_min:
        return img_f.astype(np.uint8)

    s = ((img_f - r_min) / (r_max - r_min)) * 255.0
    return np.clip(s, 0, 255).astype(np.uint8)


# --- 2. Linear Mapping ---
def linear_mapping(img_f, a, b, c=0.0, d=255.0):
    """
    Mapeamento Linear por Partes:
    Mapeia linearmente o intervalo de entrada [a, b] para o intervalo de saída [c, d].
    Fórmula: s = c + ((d - c) / (b - a)) * (r - a)
    """
    if b == a:
        return img_f.astype(np.uint8)

    s = c + ((d - c) / (b - a)) * (img_f - a)
    return np.clip(s, c, d).astype(np.uint8)


# --- 3. Transformação Logarítmica ---
def transformacao_logaritmica(img_f, r_max):
    """
    Transformação Logarítmica:
    Expande regiões escuras comprimindo regiões claras.
    Fórmula: s = c * log(1 + r), onde c = 255 / log(1 + r_max)
    """
    if r_max == 0:
        return img_f.astype(np.uint8)

    c = 255.0 / np.log(1.0 + r_max)
    s = c * np.log(1.0 + img_f)
    return np.clip(s, 0, 255).astype(np.uint8)


# --- 4. Ajuste de Contraste ---
def ajustar_contraste(img_f, alpha=1.5):
    """
    Ajuste de Contraste (Ganho):
    Multiplica os valores de intensidade pelo fator alpha.
    alpha > 1 aumenta contraste, 0 < alpha < 1 diminui contraste.
    """
    s = img_f * alpha
    return np.clip(s, 0, 255).astype(np.uint8)


# --- 5. Ajuste de Brilho ---
def ajustar_brilho(img_f, beta=40.0):
    """
    Ajuste de Brilho (Bias):
    Desloca os valores de intensidade somando o valor beta.
    beta > 0 clareia a imagem, beta < 0 escurece.
    """
    s = img_f + beta
    return np.clip(s, 0, 255).astype(np.uint8)


def plotar_comparacao_2x2(img_esq, img_dir, titulo_esq, titulo_dir):
    """
    Exibe duas imagens lado a lado juntamente com seus respectivos histogramas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Imagem da Esquerda
    axes[0, 0].imshow(img_esq, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"{titulo_esq} [min={np.min(img_esq):.1f}, max={np.max(img_esq):.1f}]")
    axes[0, 0].axis("off")

    # Histograma da Esquerda
    axes[1, 0].hist(img_esq.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.8)
    axes[1, 0].set_title(f"Histograma: {titulo_esq}")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Imagem da Direita
    axes[0, 1].imshow(img_dir, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"{titulo_dir} [min={np.min(img_dir):.1f}, max={np.max(img_dir):.1f}]")
    axes[0, 1].axis("off")

    # Histograma da Direita
    axes[1, 1].hist(img_dir.ravel(), bins=256, range=(0, 256), color="steelblue", alpha=0.8)
    axes[1, 1].set_title(f"Histograma: {titulo_dir}")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def selecionar_imagem():
    """Exibe o menu de imagens e permite escolher do scikit-image ou 'teste.png'."""
    imagens_disponiveis = {
        "1": ("camera", "data.camera() (Padrão)"),
        "2": ("moon", "data.moon() (Baixo contraste / Sombras)"),
        "3": ("page", "data.page() (Iluminação não uniforme)"),
        "4": ("coins", "data.coins() (Bordas e moedas)"),
        "5": ("cat", "data.cat() (Escala de cinza)"),
        "6": ("astronaut", "data.astronaut() (Escala de cinza)"),
        "7": ("cell", "data.cell() (Imagem biológica)"),
        "8": ("teste", "Carregar 'teste.png' do diretório atual"),
    }

    print("\n" + "-" * 60)
    print("              SELEÇÃO DA IMAGEM DE TESTE              ")
    print("-" * 60)
    for k, v in imagens_disponiveis.items():
        print(f"{k} - {v[0]} ({v[1]})")

    while True:
        op_img = input("\nEscolha a imagem (1 a 8) [Padrão: 1]: ").strip() or "1"
        if op_img in imagens_disponiveis:
            nome_chave = imagens_disponiveis[op_img][0]
            img = carregar_imagens(nome_chave)
            print(f"[+] Imagem '{nome_chave}' carregada com formato {img.shape}.")
            return nome_chave, img
        print("[!] Opção inválida. Digite um número de 1 a 8.")


def menu_pre_processamento(img_original):
    """
    Menu opcional de pré-processamento para degradar/preparar a imagem didaticamente.
    """
    print("\n" + "-" * 60)
    print("            PRÉ-PROCESSAMENTO DIDÁTICO DA IMAGEM              ")
    print("-" * 60)
    print("0 - Manter imagem original (Sem pré-processamento)")
    print("1 - Comprimir faixa de contraste (Para testar STRETCHING)")
    print("2 - Espectro de Fourier - FFT (Exemplo clássico para TRANSFORMAÇÃO LOG)")

    op = input("\nEscolha o pré-processamento (0 a 2) [Padrão: 0]: ").strip() or "0"

    if op == "1":
        print("\nConfiguração da compressão de faixa dinâmica [min_alvo, max_alvo]:")
        entrada_min = input("Informe o novo mínimo [Padrão: 80]: ").strip()
        entrada_max = input("Informe o novo máximo [Padrão: 150]: ").strip()

        min_alvo = float(entrada_min) if entrada_min else 80.0
        max_alvo = float(entrada_max) if entrada_max else 150.0

        # Mapeia linearmente [0, 255] da original para a faixa comprimida [min_alvo, max_alvo]
        img_comprimida = min_alvo + (img_original.astype(float) / 255.0) * (max_alvo - min_alvo)
        img_prep = np.clip(img_comprimida, 0, 255).astype(np.uint8)

        print(f"\n[!] Imagem comprimida para [{np.min(img_prep)}, {np.max(img_prep)}].")
        print("[i] Exibindo comparação da imagem original vs comprimida (Etapa 1)...")
        plotar_comparacao_2x2(img_original, img_prep, "Original", f"Comprimida [{min_alvo:.0f}-{max_alvo:.0f}]")
        return img_prep

    if op == "2":
        # Espectro de Fourier (Magnitude 2D FFT com shift para centralizar frequências zero)
        f_transform = np.fft.fft2(img_original.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Normaliza a magnitude crua para exibição comparativa
        mag_max = np.max(magnitude_spectrum)
        mag_normalizada = (magnitude_spectrum / mag_max * 255.0) if mag_max > 0 else magnitude_spectrum
        img_prep = np.clip(mag_normalizada, 0, 255).astype(np.uint8)

        print("\n[!] Espectro de Fourier (FFT 2D) calculado com sucesso!")
        print("[i] Exibindo comparação: Imagem no Espaço vs Espectro FFT Cru (Etapa 1)...")
        plotar_comparacao_2x2(img_original, img_prep, "Imagem Espacial", "Espectro FFT (Cru sem Log)")
        return img_prep

    return img_original


def selecionar_operacao():
    """Exibe o menu de operações e retorna (opcao, nome_exibicao, nome_slug)."""
    print("\n" + "-" * 60)
    print("              SELEÇÃO DA OPERAÇÃO DE MELHORAMENTO             ")
    print("-" * 60)
    print("1 - STRETCHING (Alongamento de Contraste)")
    print("2 - LINEAR MAPPING (Mapeamento Linear Arbitrário)")
    print("3 - TRANSFORMAÇÃO LOGARÍTMICA")
    print("4 - AJUSTE DE CONTRASTE")
    print("5 - AJUSTE DE BRILHO")

    nomes = {
        "1": ("STRETCHING", "ST"),
        "2": ("LINEAR MAPPING", "LM"),
        "3": ("TRANSFORMAÇÃO LOGARÍTMICA", "TL"),
        "4": ("AJUSTE DE CONTRASTE", "CONTRASTE"),
        "5": ("AJUSTE DE BRILHO", "BRILHO"),
    }

    while True:
        opcao = input("\nEscolha a operação (1 a 5): ").strip()
        if opcao in nomes:
            nome_exib, slug = nomes[opcao]
            return opcao, nome_exib, slug
        print("[!] Opção inválida. Digite um número de 1 a 5.")


def obter_parametros_linear_mapping(r_min, r_max):
    """Solicita os parâmetros de entrada [a, b] e saída [c, d] para Linear Mapping."""
    print("\nConfiguração dos limites [a, b] -> [c, d]:")
    print(f"Valores atuais da imagem: a={r_min:.1f}, b={r_max:.1f}")

    entrada_a = input(f"Informe 'a' (início da faixa de entrada) [{r_min:.1f}]: ").strip()
    entrada_b = input(f"Informe 'b' (fim da faixa de entrada) [{r_max:.1f}]: ").strip()
    entrada_c = input("Informe 'c' (mínimo desejado de saída) [0]: ").strip()
    entrada_d = input("Informe 'd' (máximo desejado de saída) [255]: ").strip()

    a = float(entrada_a) if entrada_a else float(r_min)
    b = float(entrada_b) if entrada_b else float(r_max)
    c = float(entrada_c) if entrada_c else 0.0
    d = float(entrada_d) if entrada_d else 255.0
    return a, b, c, d


def executar_operacao(opcao, img_f, r_min, r_max):
    """Encaminha a execução do algoritmo correspondente à opção escolhida."""
    if opcao == "1":
        return stretching(img_f, r_min, r_max)
    if opcao == "2":
        a, b, c, d = obter_parametros_linear_mapping(r_min, r_max)
        return linear_mapping(img_f, a=a, b=b, c=c, d=d)
    if opcao == "3":
        return transformacao_logaritmica(img_f, r_max)
    if opcao == "4":
        entrada_alpha = input("Informe o fator de contraste alpha (> 1 aumenta, < 1 reduz) [1.5]: ").strip()
        alpha = float(entrada_alpha) if entrada_alpha else 1.5
        return ajustar_contraste(img_f, alpha=alpha)
    if opcao == "5":
        entrada_beta = input("Informe o incremento de brilho beta (positivo clareia, negativo escurece) [40]: ").strip()
        beta = float(entrada_beta) if entrada_beta else 40.0
        return ajustar_brilho(img_f, beta=beta)
    return img_f.astype(np.uint8)


def main():
    print("=" * 60)
    print("      ALGORITMOS DE MELHORAMENTO      ")
    print("=" * 60)

    # 1. Seleção da Imagem (scikit-image ou teste.png)
    nome_img_chave, img_original = selecionar_imagem()

    # 2. Etapa de Pré-processamento opcional (Compressão para Stretching ou Espectro FFT)
    img_trabalho = menu_pre_processamento(img_original)

    # 3. Conversão para Float e métricas globais da imagem que entrará no algoritmo
    img_float = img_trabalho.astype(float)
    r_min = np.min(img_float)
    r_max = np.max(img_float)
    print(f"\n[i] Informações de Entrada: r_min={r_min:.1f}, r_max={r_max:.1f}")

    # 4. Seleção e Execução do Algoritmo de Melhoramento
    opcao, nome_exibicao, nome_slug = selecionar_operacao()
    print(f"\nProcessando operação {nome_exibicao}...")
    res_img = executar_operacao(opcao, img_float, r_min, r_max)

    # 5. Salvamento e Exibição da Etapa Final (Imagem de entrada vs Imagem Melhorada)
    nome_saida = f"saida_{nome_slug}_{nome_img_chave}.png"
    Image.fromarray(res_img).save(nome_saida)
    print(f"\n[+] Imagem resultante salva como: '{nome_saida}'")

    print("[i] Exibindo comparação: Imagem de Entrada vs Imagem Processada (Etapa 2)...")
    plotar_comparacao_2x2(img_trabalho, res_img, "Entrada do Algoritmo", f"Resultado: {nome_exibicao}")


if __name__ == "__main__":
    main()

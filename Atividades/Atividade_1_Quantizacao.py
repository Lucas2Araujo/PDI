"""
UNIVERSIDADE FEDERAL DO MARANHÃO (UFMA)
Processamento Digital de Imagens (PDI)
Trabalho: Algoritmos de Quantização de Imagens
Aluno : Lucas Araújo Dominici
- 1. Quantização Uniforme (Centroide)
- 2. Quantização Não-Uniforme por Quantis 
- 3. Quantização Não-Uniforme Ótima via K-Means
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data
from sklearn.cluster import KMeans

def quantizacao_uniforme(img, n_bits):
    """Quantização Uniforme com reconstrução pelo ponto médio."""
    n_tons = 2**n_bits
    passo = 256.0 / n_tons

    indices = np.clip(np.floor(img / passo).astype(int), 0, n_tons - 1)
    reconstrucao = np.clip(
        np.array([(i + 0.5) * passo for i in range(n_tons)]), 0, 255
    ).astype(np.uint8)

    return reconstrucao[indices]


def quantizacao_quantis(img, n_bits):
    """Quantização Não-Uniforme baseada em quantis do histograma."""
    n_tons = 2**n_bits
    dados = img.flatten()
    bordas = np.percentile(dados, np.linspace(0, 100, n_tons + 1))

    reconstrucao = np.zeros(n_tons, dtype=np.uint8)
    indices = np.zeros_like(dados, dtype=int)

    for i in range(n_tons):
        if i == n_tons - 1:
            mascara = (dados >= bordas[i]) & (dados <= bordas[i + 1])
        else:
            mascara = (dados >= bordas[i]) & (dados < bordas[i + 1])

        indices[mascara] = i
        if np.any(mascara):
            reconstrucao[i] = int(np.mean(dados[mascara]))
        else:
            reconstrucao[i] = int((bordas[i] + bordas[i + 1]) / 2)

    return reconstrucao[indices].reshape(img.shape)


def quantizacao_kmeans(img, n_bits):
    """Quantização Não-Uniforme ótima via K-Means."""
    n_tons = 2**n_bits
    pixels = img.reshape(-1, 1).astype(np.float32)

    kmeans = KMeans(n_clusters=n_tons, random_state=42, n_init=10)
    kmeans.fit(pixels)

    centroides = kmeans.cluster_centers_.round().astype(np.uint8)
    return centroides[kmeans.labels_].reshape(img.shape)

def main():
    print("=" * 60)
    print("      QUANTIZAÇÃO DE IMAGENS      ")
    print("=" * 60)

    img_original = data.camera()
    print(
        f"[OK] Imagem base embutida: 'Cameraman' ({img_original.shape[1]}x{img_original.shape[0]})"
    )

    print("\n" + "-" * 60)
    print("              SELEÇÃO DO ALGORITMO")
    print("-" * 60)
    print("1 - Quantização Uniforme (Ponto Médio)")
    print("2 - Quantização Não-Uniforme por Quantis (Histograma)")
    print("3 - Quantização Não-Uniforme via K-Means")

    while True:
        opcao_alg = input("\nEscolha o algoritmo (1, 2 ou 3): ").strip()
        if opcao_alg in ["1", "2", "3"]:
            break
        print("[!] Opção inválida. Digite 1, 2 ou 3.")

    print("\n" + "-" * 60)
    print("            QUANTIDADE DE BITS DA IMAGEM")
    print("-" * 60)

    while True:
        try:
            n_bits = int(
                input("Selecione a quantidade de bits (1 a 8): ").strip()
            )
            if 1 <= n_bits <= 8:
                break
            print("[!] Valor fora do intervalo. Digite de 1 a 8.")
        except ValueError:
            print("[!] Entrada inválida. Digite apenas números inteiros.")

    print("\nProcessando quantização...")
    n_tons = 2**n_bits

    if opcao_alg == "1":
        nome_alg = "Uniforme"
        img_quantizada = quantizacao_uniforme(img_original, n_bits)
    elif opcao_alg == "2":
        nome_alg = "Quantis_Histograma"
        img_quantizada = quantizacao_quantis(img_original, n_bits)
    else:
        nome_alg = "KMeans"
        img_quantizada = quantizacao_kmeans(img_original, n_bits)

    nome_saida = f"saida_{nome_alg}_{n_bits}bits.png"
    Image.fromarray(img_quantizada).save(nome_saida)
    print(f"\n[+] Imagem resultante salva com sucesso como: '{nome_saida}'")

    plt.figure(figsize=(10, 8))

    # 1. Imagem Original
    plt.subplot(2, 2, 1)
    plt.imshow(img_original, cmap="gray", vmin=0, vmax=255)
    plt.title("Original (8 bits / 256 tons)")
    plt.axis("off")

    # 2. Imagem Quantizada
    plt.subplot(2, 2, 2)
    plt.imshow(img_quantizada, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Quantizada: {nome_alg} ({n_bits} bits / {n_tons} tons)")
    plt.axis("off")

    # 3. Histograma Original
    plt.subplot(2, 2, 3)
    plt.hist(img_original.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title("Histograma Original")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Qtd. Pixels")
    plt.xlim(0, 255)
    plt.grid(True, linestyle="--", alpha=0.5)

    # 4. Histograma Quantizado
    plt.subplot(2, 2, 4)
    plt.hist(img_quantizada.ravel(), bins=256, range=(0, 256), color="steelblue")
    plt.title(f"Histograma Quantizado ({n_tons} níveis)")
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Qtd. Pixels")
    plt.xlim(0, 255)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
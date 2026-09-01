import matplotlib.pyplot as plt
import numpy as np
import skimage as ski 
from skimage import io, color, util
from sklearn.cluster import KMeans

# 1. Carregamento da imagem e conversão para tons de cinza
imagem_rgb = io.imread("teste.png")
imagem_Tcinza = color.rgb2gray(imagem_rgb)

print("Formato da imagem RGB é : ", imagem_rgb.shape)
print("Tipo de dado da imagem RGB : ", imagem_rgb.dtype)
print("Formato da imagem em tons de cinza é : ", imagem_Tcinza.shape)
print("Tipo de dado da imagem em tons de cinza : ", imagem_Tcinza.dtype)
print("Piso e teto dos valores na imagem em tons de cinza: ", imagem_Tcinza.min(), imagem_Tcinza.max())

imagem_uint8 = util.img_as_ubyte(imagem_Tcinza)

# 2. Entrada do usuário com validação de bits (1 a 8)
while True:
    try:
        n_niveis = int(input("Qual nível de bits você deseja? "))
        if 1 <= n_niveis <= 8:
            break
        else:
            print("Entrada inválida, por favor escreva um valor entre 1 e 8 (diferente de zero)")
    except ValueError:
        print("Entrada inválida, por favor digite apenas números inteiros de 1 à 8.")

print(f"Nível escolhido de {n_niveis} bits.")     
n_tons = 2 ** n_niveis

# 3. Quantização Uniforme (testes.py)
passo = 256 // n_tons
indices = imagem_uint8 // passo
fator_escala = 255 // (n_tons - 1)
imagem_QuantizadaUniforme = indices * fator_escala

# 4. Quantização Não-Uniforme com K-Means (QuantizaNunifor.py)
pixels = imagem_uint8.reshape(-1, 1).astype(np.float32)
kmeans = KMeans(n_clusters=n_tons, random_state=42, n_init=10)
kmeans.fit(pixels)
centroides = np.uint8(np.round(kmeans.cluster_centers_))
imagem_QuantizadaKMeans = centroides[kmeans.labels_].reshape(imagem_uint8.shape)

# 5. Geração e Comparação dos Histogramas
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Linha 1: Exibição das Imagens
axes[0, 0].imshow(imagem_uint8, cmap='gray')
axes[0, 0].set_title("Original (8 bits / 256 tons)")
axes[0, 0].axis('off')

axes[0, 1].imshow(imagem_QuantizadaUniforme, cmap='gray')
axes[0, 1].set_title(f"Quantização Uniforme ({n_niveis} bits / {n_tons} tons)")
axes[0, 1].axis('off')

axes[0, 2].imshow(imagem_QuantizadaKMeans, cmap='gray')
axes[0, 2].set_title(f"Quantização K-Means ({n_niveis} bits / {n_tons} tons)")
axes[0, 2].axis('off')

# Linha 2: Histogramas correspondentes
axes[1, 0].hist(imagem_uint8.ravel(), bins=256, range=[0, 256], color='gray')
axes[1, 0].set_title("Histograma - Original")
axes[1, 0].set_xlim([0, 256])
axes[1, 0].set_xlabel("Intensidade")
axes[1, 0].set_ylabel("Frequência (Pixels)")

axes[1, 1].hist(imagem_QuantizadaUniforme.ravel(), bins=256, range=[0, 256], color='blue')
axes[1, 1].set_title("Histograma - Quantização Uniforme")
axes[1, 1].set_xlim([0, 256])
axes[1, 1].set_xlabel("Intensidade")

axes[1, 2].hist(imagem_QuantizadaKMeans.ravel(), bins=256, range=[0, 256], color='green')
axes[1, 2].set_title("Histograma - Quantização K-Means")
axes[1, 2].set_xlim([0, 256])
axes[1, 2].set_xlabel("Intensidade")

plt.tight_layout()

# Salvar e exibir o gráfico comparativo
nome_grafico = f"comparacao_histogramas_{n_niveis}bits.png"
plt.savefig(nome_grafico, dpi=300)
print(f"\nGráfico comparativo salvo como: {nome_grafico}")

plt.show()

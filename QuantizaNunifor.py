import numpy as np
import skimage as ski 
from skimage import io, color, util
from sklearn.cluster import KMeans

imagem_rgb = io.imread("teste.png")
imagem_Tcinza = color.rgb2gray(imagem_rgb)

print("Formato da imagem RGB é : ", imagem_rgb.shape)
print("Tipo de dado da imagem RGB : ", imagem_rgb.dtype)
print("Formato da imagem em tons de cinza é : ", imagem_Tcinza.shape)
print("Tipo de dado da imagem em tons de cinza : ", imagem_Tcinza.dtype)
print("Piso e teto dos valores na imagem em tons de cinza: ", imagem_Tcinza.min(), imagem_Tcinza.max())

imagem_uint8 = util.img_as_ubyte(imagem_Tcinza)

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

# Redimensionar a matriz da imagem para (N_pixels, 1) conforme esperado pelo scikit-learn
pixels = imagem_uint8.reshape(-1, 1).astype(np.float32) 

# Ajustar o modelo K-Means para encontrar os k centróides ideais
kmeans = KMeans(n_clusters=n_tons, random_state=42, n_init=10)
kmeans.fit(pixels)

# Pegar os valores dos centróides e converter para uint8
centroides = np.uint8(np.round(kmeans.cluster_centers_))

# Substituir cada pixel pelo centróide atribuído ao seu rótulo (label)
imagem_QuantizadaKMeans = centroides[kmeans.labels_].reshape(imagem_uint8.shape)

# 4. Salvar a imagem resultante e exibir informações
nome_saida = f"teste_quantiza_kmeans_{n_niveis}bits.png"
io.imsave(nome_saida, imagem_QuantizadaKMeans)
print(f"Imagem salva com sucesso como: {nome_saida}")
print("Valor máximo e mínimo da imagem quantizada: ", imagem_QuantizadaKMeans.max(), imagem_QuantizadaKMeans.min())

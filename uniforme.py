from skimage.io import imsave
import skimage as ski 
from skimage import io, color, util

imagem_rgb = io.imread("teste.png")

imagem_Tcinza = color.rgb2gray(imagem_rgb)

print("Formato da imagem RGB é : ", imagem_rgb.shape)
print("Tipo de dado da imagem RGB : ", imagem_rgb.dtype)
print("Formato da imagem em tons de cinza é : ", imagem_Tcinza.shape)
print("Tipo de dado da imagem em tons de cinza : ", imagem_Tcinza.dtype)
print("Piso e teto dos valores na imagem em tons de cinza", imagem_Tcinza.min(), imagem_Tcinza.max())

imagem_uint8 = util.img_as_ubyte(imagem_Tcinza)
#io.imsave("teste_cinza.jpg", imagem_uint8)
while True:
    try:
        n_niveis = int(input("Qual nível de bits você deseja? "))
        if 1 <= n_niveis <=8:
            break
        else:
            print("Entrada inválida, por favor escreva um valor entre 1 e 8 (diferente de zero)")

    except ValueError:
        print("Entrada inválida, por favor digite apenas númeeros inteiros de 1 à 8.")

print(f"Nível escolhido de {n_niveis} bits.")     
n_tons = 2 ** n_niveis
passo= 256 // n_tons
indices = imagem_uint8// passo
fator_escala = 255 // (n_tons - 1)
imagem_QuantizadaUniforme = indices * fator_escala

io.imsave(f"teste_quantizanum{n_niveis}v2.png", imagem_QuantizadaUniforme)
print("Valor máximo e minimo de bits : ", imagem_QuantizadaUniforme.max(), imagem_QuantizadaUniforme.min())
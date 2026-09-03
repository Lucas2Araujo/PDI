import matplotlib.pyplot as plt
import numpy as np
from skimage import data, filters

# 1. Carregar duas imagens de teste do mesmo tamanho (512x512)
img_camera = data.camera()       # Homem com câmera
img_moon = data.moon()           # Superfície da lua

# -------------------------------------------------------
# Operação 1: Mistura com Transparência (Alpha Blending)
# -------------------------------------------------------
alpha = 0.6
beta = 0.4
# Convertemos para float para calcular sem estouro e voltamos para uint8
fusao = (alpha * img_camera + beta * img_moon)
fusao = np.clip(fusao, 0, 255).astype(np.uint8)

# -------------------------------------------------------
# Operação 2: Subtração Absoluta (Diferença entre imagens)
# -------------------------------------------------------
# A diferença absoluta evita valores negativos antes de converter
diferenca = np.abs(img_camera.astype(np.int16) - img_moon.astype(np.int16))
diferenca = np.clip(diferenca, 0, 255).astype(np.uint8)

# -------------------------------------------------------
# Visualização dos Resultados
# -------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titulos = ["Imagem A (Câmera)", "Imagem B (Lua)", "Fusão (Blending)", "Diferença Absoluta"]
imagens = [img_camera, img_moon, fusao, diferenca]

for ax, img, tit in zip(axes, imagens, titulos):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(tit)
    ax.axis("off")

plt.tight_layout()
plt.show()

# Carregando duas imagens de teste clássicas
img_moedas = data.coins()   # (303, 384) - Tons de cinza
img_texto = data.page()     # (191, 261) - Texto com iluminação irregular

# -----------------------------------------------------------------
# 1. MULTIPLICAÇÃO POR MÁSCARA BINÁRIA (Isolando uma moeda)
# -----------------------------------------------------------------
# Criando uma máscara de zeros do mesmo tamanho da imagem
linhas, colunas = img_moedas.shape
mascara = np.zeros((linhas, colunas), dtype=np.uint8)

# Desenhando um círculo branco (valor 1) sobre a primeira moeda
raio = 25
centro_y, centro_x = 40, 45
y, x = np.ogrid[:linhas, :colunas]
distancia = (x - centro_x)**2 + (y - centro_y)**2 <= raio**2
mascara[distancia] = 1

# Multiplicação ponto a ponto
moeda_isolada = img_moedas * mascara

# -----------------------------------------------------------------
# 2. DIVISÃO: CORREÇÃO DE ILUMINAÇÃO (Flattening de fundo)
# -----------------------------------------------------------------
# Estimamos o fundo criando uma versão extremamente borrada da página
# (isso remove as letras e mantém apenas o gradiente de luz/sombra)
fundo_iluminacao = filters.gaussian(img_texto, sigma=20)

# Convertendo o texto para float [0.0, 1.0] para evitar estouro
texto_float = img_texto.astype(np.float32) / 255.0

# Divisão: dividimos a imagem original pela iluminação de fundo estimada
# Adicionamos um valor ínfimo (1e-5) para evitar divisão por zero
texto_corrigido = texto_float / (fundo_iluminacao + 1e-5)

# Normalizamos para a escala [0, 255]
texto_corrigido = (texto_corrigido / texto_corrigido.max()) * 255.0
texto_corrigido = np.clip(texto_corrigido, 0, 255).astype(np.uint8)

# -----------------------------------------------------------------
# 3. VISUALIZAÇÃO DOS RESULTADOS
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Linha 1: Multiplicação
axes[0, 0].imshow(img_moedas, cmap="gray")
axes[0, 0].set_title("Moedas Original")
axes[0, 0].axis("off")

axes[0, 1].imshow(mascara, cmap="gray")
axes[0, 1].set_title("Máscara Binária (0 e 1)")
axes[0, 1].axis("off")

axes[0, 2].imshow(moeda_isolada, cmap="gray")
axes[0, 2].set_title("Multiplicação (Moeda Isolada)")
axes[0, 2].axis("off")

# Linha 2: Divisão
axes[1, 0].imshow(img_texto, cmap="gray")
axes[1, 0].set_title("Texto Original (Luz Irregular)")
axes[1, 0].axis("off")

axes[1, 1].imshow(fundo_iluminacao, cmap="gray")
axes[1, 1].set_title("Estimativa de Iluminação (Fundo)")
axes[1, 1].axis("off")

axes[1, 2].imshow(texto_corrigido, cmap="gray")
axes[1, 2].set_title("Divisão (Iluminação Uniformizada)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()
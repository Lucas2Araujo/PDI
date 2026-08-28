from PIL import Image

# 1. Carregar a imagem RGB
# Usamos a Pillow apenas para abrir o arquivo de imagem do disco
imagem_original = Image.open("teste.png")

# Garantir que a imagem esteja no formato RGB (caso possua canal Alpha / RGBA)
imagem_rgb = imagem_original.convert("RGB")

largura, altura = imagem_rgb.size
print("--- IMAGEM ORIGINAL (RGB) ---")
print(f"Dimensões (Largura x Altura): {largura} x {altura}")
print(f"Modo de cor: {imagem_rgb.mode}")
print(f"Quantidade total de pixels: {largura * altura}")

# 2. Criar uma nova imagem para tons de cinza
# Modo "L" significa Luminância / 8-bit pixels em tons de cinza (0 a 255)
imagem_cinza = Image.new("L", (largura, altura))

# Obter acesso direto aos pixels para leitura e escrita
pixels_rgb = imagem_rgb.load()
pixels_cinza = imagem_cinza.load()

# Variáveis para rastrear o piso (mínimo) e teto (máximo) dos valores calculados
valor_min = 255
valor_max = 0

# 3. Varredura manual (Pixel por Pixel)
# Percorremos todas as linhas (y) e colunas (x)
for y in range(altura):
    for x in range(largura):
        # Obter os canais R, G e B do pixel atual
        r, g, b = pixels_rgb[x, y]
        
        # Fórmula da Luminância Perceptual (Padrão ITU-R BT.601)
        # O olho humano tem sensibilidade diferente para cada comprimento de onda:
        # Verde (~58.7%), Vermelho (~29.9%) e Azul (~11.4%)
        intensidade = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
        
        # Garantir que o valor permaneça estritamente no intervalo [0, 255]
        intensidade = max(0, min(255, intensidade))
        
        # Atualizar limites mínimo e máximo
        if intensidade < valor_min:
            valor_min = intensidade
        if intensidade > valor_max:
            valor_max = intensidade
            
        # Atribuir o valor de cinza calculado à nova imagem
        pixels_cinza[x, y] = intensidade

print("\n--- IMAGEM RESULTANTE (TONS DE CINZA - NA MÃO) ---")
print(f"Dimensões (Largura x Altura): {imagem_cinza.size[0]} x {imagem_cinza.size[1]}")
print(f"Modo de cor: {imagem_cinza.mode} (8-bit grayscale)")
print(f"Piso e teto dos valores na imagem em tons de cinza: min={valor_min}, max={valor_max}")

# 4. Salvar a imagem resultante em disco
nome_arquivo_saida = "teste_cinza_namao.jpg"
imagem_cinza.save(nome_arquivo_saida)
print(f"\nImagem salva com sucesso em: {nome_arquivo_saida}")


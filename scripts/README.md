# Scripts CLI — PDI

Esta pasta contém os scripts de linha de comando refatorados do projeto.
Cada script é independente e pode ser executado diretamente via terminal.

> **Nota**: Execute sempre a partir da **raiz do projeto** (`PDI/`) para que os módulos `src/` sejam encontrados corretamente.

---

## Scripts Disponíveis

### `dithering_floyd_steinberg.py` — Quantização com Dithering (Floyd-Steinberg)

```bash
python scripts/dithering_floyd_steinberg.py <caminho_da_imagem> <bits>
```

**Exemplo:**
```bash
python scripts/dithering_floyd_steinberg.py assets/lena_color.png 2
```

Gera: `lena_color_floyd_steinberg_2bits.png` e exibe relatório comparativo de **MSE** e **PSNR** com a quantização direta.

---

### `uniforme.py` — Quantização Uniforme (Centróides)

```bash
python scripts/uniforme.py <caminho_da_imagem> <bits>
```

**Exemplo:**
```bash
python scripts/uniforme.py assets/sample_portrait.png 4
```

Gera: `sample_portrait_uniforme_4bits.png` — Reconstrução calculada no centróide ótimo de cada intervalo.

---

### `quantiza_nao_uniforme.py` — Quantização K-Means

```bash
python scripts/quantiza_nao_uniforme.py <caminho_da_imagem> <bits>
```

**Exemplo:**
```bash
python scripts/quantiza_nao_uniforme.py assets/sample_portrait.png 4
```

Gera: `sample_portrait_kmeans_4bits.png`

> **Atenção**: O K-Means pode demorar alguns segundos em imagens grandes.

---

### `histograma_comparativo.py` — Comparação Completa com Histogramas

```bash
python scripts/histograma_comparativo.py <caminho_da_imagem> <bits>
```

**Exemplo:**
```bash
python scripts/histograma_comparativo.py assets/sample_portrait.png 4
```

Gera: `sample_portrait_comparativo_4bits.png` — Grade 2×3 com as 3 imagens e seus histogramas.
Também exibe as métricas **MSE** e **PSNR** no terminal.

---

### `grayscale_manual.py` — Conversão Manual Pixel a Pixel

```bash
python scripts/grayscale_manual.py <caminho_da_imagem>
```

**Exemplo:**
```bash
python scripts/grayscale_manual.py assets/sample_portrait.png
```

Gera: `sample_portrait_cinza_manual.jpg`

Demonstração didática da fórmula ITU-R BT.601 implementada pixel a pixel com Pillow.

---

## Parâmetros Comuns

| Parâmetro | Descrição | Valores |
|---|---|---|
| `<caminho_da_imagem>` | Arquivo de entrada | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` |
| `<bits>` | Nível de quantização | Inteiro de 1 a 8 |

| Bits | Tons de cinza | Resultado esperado |
|---|---|---|
| 1 | 2 tons | Binário (preto e branco puro) |
| 2 | 4 tons | Muito degradado |
| 4 | 16 tons | Posterização visível |
| 6 | 64 tons | Qualidade aceitável |
| 8 | 256 tons | Qualidade original |


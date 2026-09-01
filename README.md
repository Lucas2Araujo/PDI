# PDI — Processamento Digital de Imagens `v0.1`

Repositório de trabalhos e aplicações da disciplina de **Processamento Digital de Imagens (PDI)** — UFMA.

Inclui uma **interface gráfica moderna** (Desktop + Web) para conversão para tons de cinza, quantização digital (Uniforme e K-Means), comparação visual direta com imagem colorida original, análise de histogramas e processamento em lote.

---

## 🚀 Início Rápido

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o aplicativo

```bash
# Janela Desktop nativa (padrão)
python main.py

# Modo Web — abre no navegador padrão
python main.py --web

# Web com porta customizada
python main.py --web --port 9000
```

### 3. Executar a Suíte de Testes

```bash
# Execução via unittest padrão
python -m unittest discover -s tests -v

# Ou via pytest
pytest -v
```

---

## 🖼️ Funcionalidades da Interface (v0.1)

### 🌓 Tema Automático Claro / Escuro
- Suporte nativo ao tema do sistema operacional (`Auto / System`).
- Seletor interativo na barra superior: **Auto (🌓)**, **Claro (☀️)** ou **Escuro (🌙)**.

### 🎨 Aba: Imagem Individual
- Seleção de imagem via diálogo nativo do sistema (PNG, JPG, BMP, TIFF, WebP).
- **5 Imagens de Teste Embutidas (Disponíveis em Desktop e Web)**:
  - 👤 **Exemplo 1 (Retrato RGB)**: Imagem $512 \times 512$ com tons de pele e gradientes naturais.
  - 📊 **Exemplo 2 (Benchmark Sintético)**: Imagem $512 \times 512$ com degraus de luminância e formas geométricas.
  - 👒 **Exemplo 3 (Lenna Clássica)**: A clássica imagem de teste de PDI ($512 \times 512$ RGB).
  - 🐕 **Exemplo 4 (Ayla HD)**: Fotografia de alta resolução ($1600 \times 1494$ RGB).
  - 🏛️ **Exemplo 5 (Pentágono TIFF)**: Imagem aérea clássica monocromática ($1024 \times 1024$ TIFF).
- **Menu Didático & Organizado de Conversão para Tons de Cinza**:
  - **Ponderação & Média**:
    - 🌟 **Luminância ITU-R BT.601** (Padrão perceptual fisiológico): $Y = 0.2989R + 0.5870G + 0.1140B$
    - ⚖️ **Média Aritmética**: $Y = (R + G + B) / 3$
  - **Isolamento de Canais de Cor (RGB)**:
    - 🔴 **Canal Vermelho (R)** — Realce de tons quentes e pele.
    - 🟢 **Canal Verde (G)** — Canal com maior nitidez e menor ruído.
    - 🔵 **Canal Azul (B)** — Realce de céu e sombras.
  - Caixa explicativa com fórmulas matemáticas e guia didático em tempo real.
- **📥 Botão Rápido de Conversão para Tons de Cinza**:
  - Converte diretamente qualquer imagem RGB para escala de cinza de 8 bits e abre o diálogo de download imediato (`_cinza_8bits.png`).
- **🔍 Zoom Interativo & Pan (`InteractiveViewer`)**:
  - Zoom de até 10× usando o scroll do mouse ou gesto de pinça no trackpad/tela touchscreen.
  - Arraste (Pan) para inspecionar artefatos de quantização pixel a pixel.
- **4 Modos de Algoritmo & Quantização**:
  - **Modo 1: Quantização Uniforme** — Intervalos de tamanho igual lineares $2^b$, complexidade linear $O(H \cdot W)$.
  - **Modo 2: Quantização Não-Uniforme (K-Means)** — Centróides adaptativos por agrupamento estatístico ótimo.
  - **Modo 3: Quantização Baseada em Histograma** — Particionamento adaptativo por quantis/frequência de ocorrência.
  - **Modo 4: Comparação Completa (2×3)** — Pipeline completo reproduzindo o script comparativo com métricas e histogramas lado a lado.
- **Slider interativo** de 1 a 8 bits ($2^b$ tons de cinza).
- **Múltiplos Modos de Visualização & Comparação**:
  - 📊 **Gráficos & Histogramas**: Visão analítica em tons de cinza.
  - 🎨 **Gráfico com Cores**: Visão analítica completa com histogramas RGB sobrepostos.
  - 🖼️ **Apenas Quantizada**: Imagem processada pura em alta definição com zoom.
  - 🌓 **Lado a Lado (Cinza × Quantizada)**: Comparação de detalhes.
  - 🌈 **Lado a Lado (Colorida × Quantizada)**: Comparação direta da imagem original colorida com a quantizada.
  - 📑 **Grade Tripla**: Visualização simultânea [Original Colorida | Cinza 8-bit | Quantizada].
- **Métricas de Qualidade**: MSE (Mean Squared Error), PSNR (dB) e tempo de execução.
- **Exportação**: Salva imagem pura ou gráfico analítico correspondente ao modo ativo.

### 📁 Aba: Processamento em Lote
- Seleção de diretório de entrada e saída.
- Conversão e quantização em lote em segundo plano (não trava a interface).
- Barra de progresso em tempo real e relatório final de sucessos e falhas.

---

## 🤖 Integração Contínua & Deploy Web (CI/CD)

O projeto conta com automação via **GitHub Actions** (`.github/workflows/ci-cd.yml`):
1. **Suíte de Testes**: Executa todos os testes unitários e de integração a cada `push` e `pull_request`.
2. **Deploy Web**: Compila automaticamente o app em Web Assembly/HTML estático via `flet build web` e publica no **GitHub Pages** a cada commit nas branches `main`/`master`.

---

## 🗂️ Estrutura do Projeto

```
PDI/
├── main.py                       # Ponto de entrada (--web para modo Web)
├── requirements.txt              # Dependências do projeto
├── .github/workflows/ci-cd.yml   # Workflow de Testes e Deploy Web no GitHub Pages
│
├── tests/                        # Suíte de Testes Automatizados
│   ├── test_grayscale.py         # Testes de conversão para escala de cinza
│   ├── test_quantization.py      # Testes de quantização uniforme e K-Means
│   ├── test_histogram.py         # Testes de histogramas, MSE, PSNR e figuras
│   ├── test_batch.py             # Testes de processamento em lote
│   └── test_ui.py                # Testes de tema, configuração e UI
│
├── scripts/                      # Scripts CLI didáticos
│   ├── uniforme.py
│   ├── quantiza_nao_uniforme.py
│   ├── histograma_comparativo.py
│   └── grayscale_manual.py
│
└── src/
    ├── core/                     # Algoritmos de PDI (puros, sem dependência de UI)
    │   ├── grayscale.py          # Conversão RGB→Cinza
    │   ├── quantization.py       # Quantização Uniforme e K-Means
    │   ├── histogram.py          # Histogramas e métricas (MSE, PSNR)
    │   └── batch.py              # Motor de processamento em lote
    └── ui/                       # Interface Gráfica Flet
        ├── theme.py              # Paleta Claro/Escuro e temas dinâmicos
        ├── app.py                # Aplicação principal, abas e seletor de tema
        └── views/
            ├── single_view.py    # Aba de imagem individual com comparações
            └── batch_view.py     # Aba de processamento em lote
```

---

## 📚 Teoria dos Algoritmos

### Conversão para Tons de Cinza

**Luminância Perceptual (ITU-R BT.601)**:
$$Y = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$$

Os pesos refletem a sensibilidade fisiológica do olho humano (maior para a faixa do verde, moderada para o vermelho e menor para o azul).

### Quantização Uniforme

Divide o espaço de intensidades $[0, 255]$ em $N = 2^b$ intervalos de tamanho igual:
$$\text{passo} = \lfloor 256 / N \rfloor, \quad \text{saída} = \lfloor \text{pixel} / \text{passo} \rfloor \times \lfloor 255 / (N-1) \rfloor$$

### Quantização Não-Uniforme (K-Means)

Encontra os $k = 2^b$ centróides que minimizam a inércia intra-cluster no espaço de intensidades, resultando em uma quantização adaptada à distribuição real dos pixels.

### Métricas de Qualidade

**MSE** (Mean Squared Error):
$$\text{MSE} = \frac{1}{H \cdot W} \sum_{i,j} (I_{orig}[i,j] - I_{quant}[i,j])^2$$

**PSNR** (Peak Signal-to-Noise Ratio):
$$\text{PSNR} = 20 \cdot \log_{10}(255) - 10 \cdot \log_{10}(\text{MSE}) \quad \text{[dB]}$$

---

## 🛠️ Dependências

| Biblioteca | Uso |
|---|---|
| `flet` | Interface gráfica Desktop e Web moderna |
| `numpy` | Operações vetorizadas sobre arrays multidimensionais |
| `scikit-image` | Carregamento, conversão e manipulação de imagens |
| `scikit-learn` | Algoritmo K-Means para quantização não-uniforme |
| `matplotlib` | Geração de histogramas e figuras comparativas |
| `pillow` | Codificação e salvamento de buffers de imagem |


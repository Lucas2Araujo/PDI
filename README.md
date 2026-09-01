# PDI — Processamento Digital de Imagens `v0.1`

Repositório de trabalhos e aplicações práticas da disciplina de **Processamento Digital de Imagens (PDI)** — Universidade Federal do Maranhão (**UFMA**).

O projeto conta com uma **interface gráfica moderna e responsiva** (com suporte nativo a **Desktop** e **Deploy Web**) desenvolvida em Python e Flet, integrando algoritmos didáticos de conversão para escala de cinza, isolamento de canais RGB, quantização digital (Uniforme, K-Means e Histograma), inspeção de telemetria do pipeline, análise de histogramas e processamento em lote com exportação em disco ou download em ZIP.

---

## 🚀 Início Rápido

### 1. Instalar dependências

Recomenda-se o uso de um ambiente virtual Python (3.10+):

```bash
# Criação e ativação do ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate no Windows

# Instalação dos pacotes
pip install -r requirements.txt
```

### 2. Executar o aplicativo

```bash
# Janela Desktop nativa (padrão)
python main.py

# Modo Web — abre no navegador padrão (porta 8550)
python main.py --web

# Modo Web com porta customizada
python main.py --web --port 9000
```

### 3. Executar a Suíte de Testes Automatizados

```bash
# Execução via unittest padrão (60 testes integrados)
python -m unittest discover -s tests -v

# Ou via pytest
pytest -v
```

---

## 🖼️ Funcionalidades da Interface

### 🌓 Tema Dinâmico Claro / Escuro
- Suporte nativo ao tema do sistema operacional (`ThemeMode.SYSTEM`).
- Seletor interativo na barra de cabeçalho: **Auto (🌓)**, **Claro (☀️)** ou **Escuro (🌙)**.
- Interface adaptativa para telas desktop, tablets e dispositivos móveis.

---

### 🎨 Aba 1: Imagem Individual

1. **Seleção e Amostras Embutidas**:
   - Seleção de arquivos locais (`PNG`, `JPG`, `JPEG`, `BMP`, `TIFF`, `WebP`).
   - **5 Amostras Integradas** prontas para teste com 1 clique (Desktop & Web):
     - 🛰️ **Exemplo 1 (Imagem Aérea)**: $512 \times 512$ RGB, rica em texturas e relevo.
     - 📊 **Exemplo 2 (Benchmark Sintético)**: $512 \times 512$ RGB, degraus de luminância e formas geométricas.
     - 👒 **Exemplo 3 (Lenna Clássica)**: $512 \times 512$ RGB, o clássico padrão de PDI.
     - 🐕 **Exemplo 4 (Ayla HD)**: $1600 \times 1494$ RGB, fotografia em alta definição com iluminação natural.
     - 🏛️ **Exemplo 5 (Pentágono TIFF)**: $1024 \times 1024$ TIFF monocromático, fotografia aérea de alta frequência espacial.

2. **Conversão Didática para Escala de Cinza & Canais**:
   - **Ponderação & Média**:
     - 🌟 **Luminância ITU-R BT.601** (Padrão fisiológico humano): $Y = 0.2989R + 0.5870G + 0.1140B$
     - ⚖️ **Média Aritmética**: $Y = (R + G + B) / 3$
   - **Isolamento de Canais de Cor (RGB)**:
     - 🔴 **Canal Vermelho (R)** — Matriz $[R, 0, 0]$ (tons de vermelho).
     - 🟢 **Canal Verde (G)** — Matriz $[0, G, 0]$ (tons de verde).
     - 🔵 **Canal Azul (B)** — Matriz $[0, 0, B]$ (tons de azul).
   - Caixa explicativa com fórmulas matemáticas e descrições teóricas em tempo real.
   - Botão de atalho **📥 Converter & Salvar em Tons de Cinza (8 bits)**.

3. **Técnicas de Quantização**:
   - **Modo 1: Quantização Uniforme** — Intervalos de tamanho igual lineares $2^b$, complexidade $O(H \cdot W)$.
   - **Modo 2: Quantização Não-Uniforme (K-Means)** — Centróides adaptativos por agrupamento estatístico ótimo.
   - **Modo 3: Quantização Baseada em Histograma** — Particionamento adaptativo por quantis/frequência de ocorrência.
   - **Modo 4: Comparação Completa (2×3)** — Figura analítica comparando imagem original, quantizações e histogramas.
   - **Slider de Profundidade de Bits**: Ajuste contínuo de 1 a 8 bits ($2^1 = 2$ até $2^8 = 256$ tons).

4. **Modos de Visualização & Inspeção**:
   - 📊 **Gráfico Cinza**: Visão analítica monocromática com histogramas de frequência.
   - 🎨 **Gráfico Colorido**: Visão analítica com histogramas RGB sobrepostos.
   - 🖼️ **Apenas Quantizada**: Imagem pura de alta definição.
   - 🌓 **Lado a Lado (Cinza × Quantizada)**: Inspeção direta de perdas por quantização.
   - 🌈 **Lado a Lado (Colorida × Quantizada)**: Comparação do original RGB com o resultado quantizado.
   - 📑 **Grade Tripla**: Visualização em 3 colunas [Original RGB | Cinza 8-bit | Quantizada].
   - 🔍 **Visualizador com Zoom & Pan**: Diálogo modal interativo com zoom de até 10× e pan para análise pixel a pixel de artefatos.
   - 🔬 **Entranhas do Processo (Inspetor de Telemetria)**: Diagnóstico detalhado de dimensões, canais, centroides K-Means, limites de quantização e métricas.

5. **Métricas de Fidelidade**:
   - Cálculo automático de **MSE** (*Mean Squared Error*) e **PSNR** (*Peak Signal-to-Noise Ratio* em dB), além de tempo de execução.

---

### 📁 Aba 2: Processamento em Lote (Batch Processing)

Projetada com arquitetura híbrida para operar tanto no ambiente Desktop nativo quanto em Deploy Web:

| Recurso | Ambiente Desktop | Ambiente Web (Browser) |
|---|---|---|
| **Seleção por Pasta** | ✅ `FilePicker.get_directory_path` | ❌ *(Incompatível com APIs de navegadores)* |
| **Multi-seleção de Arquivos** | ✅ `pick_files(allow_multiple=True)` | ✅ `pick_files(allow_multiple=True, with_data=True)` |
| **5 Amostras do App** | ✅ Carrega do disco local | ✅ Carrega do servidor Python em RAM |
| **Armazenamento de Saída** | ✅ Salva na pasta de saída escolhida | ✅ Processamento em memória (*Zero-disk client*) |
| **Exportação dos Resultados** | ✅ Arquivos gravados no diretório | ✅ **Download Tudo em ZIP** ou **Downloads individuais** |
| **Feedback de Progresso** | ✅ Barra de progresso + Log em tempo real | ✅ Barra de progresso + Log em tempo real |

---

## 🤖 Integração Contínua & Deploy Web (CI/CD)

O repositório possui fluxo automatizado via **GitHub Actions** (`.github/workflows/ci-cd.yml`):
1. **Testes Automatizados**: Executa a suíte de 60 testes unitários e de integração a cada `push` e `pull_request`.
2. **Build & Deploy Web**: Compila a interface com Flet Web / WASM e publica automaticamente no **GitHub Pages** a cada commit nas branches principais (`main`/`master`).

---

## 🗂️ Estrutura do Projeto

```
PDI/
├── main.py                       # Ponto de entrada CLI/GUI (--web para modo Web)
├── requirements.txt              # Dependências do projeto
├── .github/workflows/ci-cd.yml   # Workflow de Testes e Deploy Web no GitHub Pages
│
├── assets/                       # Imagens de exemplo, ícones e assets estáticos
│   ├── favicon.png / .ico
│   ├── sample_portrait.png
│   ├── sample_benchmark.png
│   ├── lena_color.png
│   ├── ayla.jpg
│   └── pentagono.tiff
│
├── tests/                        # Suíte de Testes Automatizados (unittest)
│   ├── test_grayscale.py         # Conversão para tons de cinza e isolamento RGB
│   ├── test_quantization.py      # Quantização Uniforme, K-Means e Histograma
│   ├── test_histogram.py         # Histogramas e métricas (MSE, PSNR)
│   ├── test_inspector.py         # Inspetor de telemetria e diagnósticos
│   ├── test_samples.py           # Carregamento e integridade das amostras
│   ├── test_batch.py             # Processamento em lote em disco e em memória
│   └── test_ui.py                # Componentes visuais, temas e vistas
│
├── scripts/                      # Scripts didáticos em linha de comando (CLI)
│   ├── uniforme.py               # Demonstração de quantização uniforme
│   ├── quantiza_nao_uniforme.py  # Demonstração de K-Means
│   ├── histograma_comparativo.py # Geração da figura analítica 2×3
│   └── grayscale_manual.py       # Conversão manual para cinza
│
└── src/
    ├── core/                     # Módulos puros de processamento (sem dependência de UI)
    │   ├── grayscale.py          # Fórmulas de luminância, média e isolamento de canal
    │   ├── quantization.py       # Algoritmos de quantização (Uniforme, K-Means, Histograma)
    │   ├── histogram.py          # Cálculo de histogramas, métricas MSE/PSNR e Matplotlib
    │   ├── inspector.py          # Extração de telemetria do pipeline
    │   ├── samples.py            # Gerenciador das 5 amostras embutidas
    │   └── batch.py              # Motor de processamento em lote (disco e memória)
    │
    └── ui/                       # Interface Gráfica Flet
        ├── theme.py              # Paleta Claro/Escuro, badges e tokens de design
        ├── app.py                # Barra de título, abas e gerenciador de temas
        └── views/
            ├── single_view.py    # Aba de processamento individual, zoom e comparações
            └── batch_view.py     # Aba de processamento em lote adaptativa (Desktop/Web)
```

---

## 📚 Fundamentação Teórica

### 1. Conversão para Tons de Cinza

- **Luminância Perceptual (ITU-R BT.601)**:
  $$Y = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$$
  Compensa a sensibilidade fotópica do olho humano, que possui maior densidade de cones sensíveis ao comprimento de onda verde.

- **Média Aritmética**:
  $$Y = \frac{R + G + B}{3}$$

---

### 2. Quantização Uniforme

Divide linearmente o intervalo dinâmico $[0, 255]$ em $N = 2^b$ intervalos de largura igual:
$$\Delta = \left\lfloor \frac{256}{N} \right\rfloor, \quad q(x) = \left\lfloor \frac{x}{\Delta} \right\rfloor \cdot \left\lfloor \frac{255}{N - 1} \right\rfloor$$

---

### 3. Quantização Não-Uniforme (K-Means)

Encontra $k = 2^b$ centróides $C = \{c_1, c_2, \dots, c_k\}$ que minimizam a inércia intra-cluster (soma dos quadrados dos desvios):
$$J = \sum_{j=1}^k \sum_{x \in S_j} \|x - c_j\|^2$$
Adaptando os níveis de cinza às regiões de maior densidade de pixels no histograma.

---

### 4. Quantização Baseada em Histograma (Quantis)

Particiona o histograma acumulado em $N = 2^b$ intervalos equiprováveis, garantindo que cada faixa de quantização represente aproximadamente a mesma quantidade de pixels da imagem original.

---

### 5. Métricas de Fidelidade

- **MSE** (*Mean Squared Error*):
  $$\text{MSE} = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W (I_{\text{orig}}[i,j] - I_{\text{quant}}[i,j])^2$$

- **PSNR** (*Peak Signal-to-Noise Ratio*):
  $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right) = 20 \cdot \log_{10}(255) - 10 \cdot \log_{10}(\text{MSE}) \quad \text{[dB]}$$

---

## 🛠️ Tecnologias & Dependências

| Biblioteca | Versão | Finalidade |
|---|---|---|
| `flet` | `>=0.86.0` | Interface gráfica declarativa moderna para Desktop e Web |
| `numpy` | `>=1.24.0` | Processamento matricial e operações vetorizadas |
| `scikit-image` | `>=0.21.0` | Algoritmos de processamento e manipulação de imagens |
| `scikit-learn` | `>=1.3.0` | Algoritmo K-Means para quantização não-uniforme |
| `matplotlib` | `>=3.7.0` | Renderização de histogramas analíticos e figuras comparativas |
| `pillow` | `>=10.0.0` | Decodificação, conversão de formatos e buffers em memória |

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos na Universidade Federal do Maranhão (UFMA).
Distribuído sob licença MIT.

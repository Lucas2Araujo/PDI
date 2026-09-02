# PDI — Processamento Digital de Imagens `v0.2`

Repositório de trabalhos e aplicações práticas da disciplina de **Processamento Digital de Imagens (PDI)** — Universidade Federal do Maranhão (**UFMA**).

O projeto conta com uma **interface gráfica moderna, interativa e responsiva** (com suporte nativo a **Desktop** e **Deploy Web**) desenvolvida em Python e Flet, integrando algoritmos didáticos de conversão para escala de cinza, isolamento de canais RGB, quantização digital (Uniforme, K-Means, Histograma e Dithering Floyd-Steinberg), histogramas acelerados por hardware via Flutter Canvas, inspeção profunda de telemetria ("Entranhas do Processo"), otimizações de memória contra Out-Of-Memory (OOM) e processamento em lote com exportação em disco ou download em ZIP.

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
# Execução via unittest padrão (99 testes integrados)
python -m unittest discover -s tests -v

# Ou via pytest
pytest -v
```

---

## 🖼️ Funcionalidades da Interface

### 🌓 Tema Dinâmico Claro / Escuro & Responsividade
- Suporte nativo ao tema do sistema operacional (`ThemeMode.SYSTEM`).
- Seletor interativo na barra de cabeçalho: **Auto (🌓)**, **Claro (☀️)** ou **Escuro (🌙)**.
- Layout adaptativo automático para resoluções Desktop, Tablets e Navegadores Mobile.

---

### 🎨 Aba 1: Imagem Individual

1. **Seleção e Amostras Embutidas**:
   - Seleção de arquivos locais (`PNG`, `JPG`, `JPEG`, `BMP`, `TIFF`, `WebP`).
   - **5 Amostras Integradas** prontas para teste com 1 clique (Desktop & Web):
     - 🛰️ **Exemplo 1 (Imagem Aérea)**: $512 \times 512$ RGB, rica em texturas e relevo geográfico.
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
   - **Modo Colorido Direto (RGB)**: Opção de quantizar diretamente nos 3 canais de cores sem conversão prévia.
   - Caixa explicativa com fórmulas matemáticas e descrições teóricas dinâmicas.
   - Botão de atalho **📥 Converter & Salvar em Tons de Cinza (8 bits)**.

3. **Técnicas de Quantização**:
   - **Modo 1: Quantização Uniforme** — Intervalos lineares de igual largura e reconstrução no centróide. Em RGB, aplica particionamento escalar por canal gerando $(2^b)^3$ cores.
   - **Modo 2: Quantização Não-Uniforme (K-Means)** — Centróides adaptativos por agrupamento estatístico ótimo no espaço 1D (cinza) ou espaço 3D de cores (RGB). Carregado via *Lazy Loading* para boot instantâneo.
   - **Modo 3: Quantização Baseada em Histograma** — Particionamento adaptativo por quantis/percentis com faixas equiprováveis de pixels.
   - **Modo 4: Dithering por Difusão de Erro (Floyd-Steinberg)** — Difunde o resíduo da quantização espacialmente para os 4 vizinhos imediatos ($7/16, 3/16, 5/16, 1/16$), eliminando o efeito de falso contorno (*color banding*) em profundidades de 1 a 4 bits.
   - **Modo 5: Comparação Geral (Uniforme × K-Means)** — Painel analítico lado a lado com métricas comparativas simultâneas.
   - **Slider Contínuo de Bits**: Ajuste de 1 a 8 bits ($2^1 = 2$ até $2^8 = 256$ níveis).

4. **Modos Dinâmicos de Visualização (SegmentedButton)**:
   - 📊 **Painel Analítico (Cinza ou RGB)**: Histograma nativo de entrada vs. histograma quantizado com métricas em tempo real.
   - 🎨 **Comparação Colorida**: Visão analítica com canais RGB sobrepostos.
   - 🖼️ **Apenas Quantizada**: Exibição da imagem processada pura em alta definição.
   - 🌓 **Lado a Lado (Original × Quantizada)**: Inspeção direta de perdas e artefatos de quantização.
   - 📑 **Grade Tripla**: [Original RGB | Cinza 8-bit | Quantizada].
   - 🪄 **Comparação Direta × Floyd-Steinberg**: Visualização comparativa direta ressaltando o ganho perceptual do Dithering.

5. **Histogramas Nativos em Tempo Real (`NativeHistogramChart`)**:
   - Renderização acelerada por hardware via `flet.canvas.Canvas` (Flutter Canvas), eliminando overhead do Matplotlib na UI.
   - Suporte a histograma monocromático com preenchimento suave, histograma RGB multi-canal sobreposto e histograma discreto de degraus de quantização.
   - **🔍 Zoom Dedicado do Histograma (`open_histogram_zoom_dialog`)**: Pop-up modal com gráfico ampliado e telemetria estatística completa:
     - Resolução total (px), Níveis Únicos, Mínimo e Máximo, Média ($\mu$) e Desvio Padrão ($\sigma$).

6. **Visualizador Modal com Zoom & Pan (`open_zoom_dialog`)**:
   - Inspeção interativa pixel a pixel com amplitude de **0.25× a 10×** e navegação por Pan (arraste).
   - Controles de zoom in, zoom out, reset (100%) e atalhos rápidos.
   - **📥 Botão de Download Integrado**: Baixe a imagem inspecionada diretamente da janela modal.

7. **🔬 Entranhas do Processo (Raio-X Didático em 4 Abas)**:
   - **Aba 1: Estrutura Matricial Original** — Amostra central $5 \times 5$ detalhada com coordenadas de linha/coluna e valores numéricos dos pixels (RGB e Cinza).
   - **Aba 2: Aritmética da Conversão** — Demonstração passo a passo da aritmética de conversão de cor para cada pixel inspecionado.
   - **Aba 3: Mecânica de Quantização** — Tabela completa de partições, limites de decisão $[\text{min}, \text{max}]$, centróides de reconstrução e tamanho do passo ($\Delta$).
   - **Aba 4: Auditoria de Erro Residual e Mapa de Calor** — Matriz de resíduos absolutos $|I_{\text{cinza}} - I_{\text{quant}}|$, mapa térmico (Heatmap) gerado em puro NumPy e botão com zoom modal dedicado.

8. **Exportação & Pacote Completo**:
   - Botões individuais de download para imagem quantizada, imagem em tons de cinza e figuras analíticas.
   - **📦 Botão "Download Tudo em ZIP"**: Empacota e baixa de uma vez todas as variações geradas da imagem (Original, Cinza, Quantizada, Dithering e Painéis).

---

### 📁 Aba 2: Processamento em Lote (Batch Processing)

Projetada com arquitetura híbrida para operar com máxima performance tanto no ambiente Desktop nativo quanto no Web:

| Recurso | Ambiente Desktop | Ambiente Web (Browser / WASM) |
|---|---|---|
| **Seleção por Pasta** | ✅ `FilePicker.get_directory_path` | ❌ *(Incompatível com sandbox dos navegadores)* |
| **Multi-seleção de Arquivos** | ✅ `pick_files(allow_multiple=True)` | ✅ `pick_files(allow_multiple=True, with_data=True)` |
| **5 Amostras do App** | ✅ Carrega do disco local | ✅ Carrega da memória RAM do servidor |
| **Pré-visualização da Fila** | ✅ Miniaturas leves de 160 px | ✅ Miniaturas leves de 160 px (*Proteção de RAM*) |
| **Armazenamento de Saída** | ✅ Gravação automática no diretório | ✅ Processamento em memória (*Zero-disk client*) |
| **Exportação dos Resultados** | ✅ Arquivos gravados na pasta | ✅ **Download Tudo em ZIP** ou **Downloads individuais** |
| **Telemetria Individual** | ✅ Zoom e "Entranhas do Processo" por item | ✅ Zoom e "Entranhas do Processo" por item |
| **Feedback de Progresso** | ✅ Barra de progresso + Log em tempo real | ✅ Barra de progresso + Log em tempo real |

---

## ⚡ Otimizações de Memória & Performance

1. **Downscaling Preventivo (`src/core/image_io.py`)**:
   - Imagens carregadas que ultrapassam $800 \times 800$ pixels são redimensionadas preventivamente mantendo o *aspect ratio* via Lanczos.
   - Evita travamentos por *Out-Of-Memory* (OOM) no WebAssembly/Pyodide e reduz drasticamente a latência de renderização.
2. **Carregamento Sob Demanda (Lazy Loading)**:
   - A biblioteca `scikit-learn` só é importada quando a Quantização K-Means é efetivamente solicitada, acelerando o tempo de inicialização da aplicação em mais de 70%.
3. **Cálculo Numérico e Heatmap em Puro NumPy**:
   - Histogramas numéricos, métricas de qualidade (MSE/PSNR) e mapas de calor de erro residual calculados diretamente em NumPy/PIL, dispensando chamadas bloqueantes ao Matplotlib.
4. **Coleta de Lixo Explícita**:
   - Invocação controlada de `gc.collect()` após processamento de matrizes grandes e operações de lote.

---

## 🤖 Integração Contínua & Deploy Web (CI/CD)

O repositório possui fluxo automatizado via **GitHub Actions** (`.github/workflows/ci-cd.yml`):
1. **Testes Automatizados**: Executa a suíte de **99 testes unitários e de integração** a cada `push` e `pull_request`.
2. **Build & Deploy Web**: Compila a aplicação com Flet Web / WASM e publica automaticamente no **GitHub Pages** a cada commit nas branches principais (`main`/`master`).

---

## 🗂️ Estrutura do Projeto

```
PDI/
├── main.py                       # Ponto de entrada CLI/GUI (--web para modo Web)
├── pyproject.toml                # Configurações do projeto e metadados
├── requirements.txt              # Dependências do projeto
├── .github/workflows/ci-cd.yml   # Workflow de Testes (99 testes) e Deploy Web no GitHub Pages
│
├── assets/ / src/assets/         # Imagens de exemplo, ícones e assets estáticos
│   ├── favicon.png / .ico
│   ├── sample_portrait.png
│   ├── sample_benchmark.png
│   ├── lena_color.png
│   ├── ayla.jpg
│   └── pentagono.tiff
│
├── tests/                        # Suíte de Testes Automatizados (99 testes no unittest)
│   ├── test_grayscale.py         # Conversão para tons de cinza e isolamento RGB
│   ├── test_quantization.py      # Quantização Uniforme, K-Means, Histograma e Floyd-Steinberg
│   ├── test_histogram.py         # Histogramas numéricos e métricas (MSE, PSNR)
│   ├── test_inspector.py         # Inspetor de telemetria e diagnósticos do pipeline
│   ├── test_samples.py           # Carregamento e integridade das amostras
│   ├── test_batch.py             # Processamento em lote em disco e em memória
│   ├── test_optimizations.py     # Otimizações de I/O, downscale, lazy-loading e Canvas
│   └── test_ui.py                # Componentes visuais, modais de zoom e vistas
│
├── scripts/                      # Scripts didáticos em linha de comando (CLI)
│   ├── README.md                 # Documentação dos scripts CLI
│   ├── dithering_floyd_steinberg.py # Quantização com Dithering (Floyd-Steinberg)
│   ├── uniforme.py               # Demonstração de quantização uniforme
│   ├── quantiza_nao_uniforme.py  # Demonstração de K-Means
│   ├── histograma_comparativo.py # Geração da figura analítica comparativa
│   └── grayscale_manual.py       # Conversão manual para cinza
│
└── src/
    ├── core/                     # Módulos puros de processamento (sem dependência de UI)
    │   ├── grayscale.py          # Fórmulas de luminância, média e isolamento de canais RGB
    │   ├── quantization.py       # Algoritmos de quantização (Uniforme, K-Means, Histograma, Dithering)
    │   ├── histogram.py          # Cálculo numérico de histogramas, métricas MSE/PSNR e figuras analíticas
    │   ├── inspector.py          # Telemetria do pipeline, tabelas de decisão e mapa de calor puro NumPy
    │   ├── image_io.py           # Downscaling preventivo (máx 800×800), thumbnails e I/O eficiente
    │   ├── samples.py            # Gerenciador das 5 amostras embutidas
    │   └── batch.py              # Motor de processamento em lote (disco e memória)
    │
    └── ui/                       # Interface Gráfica Flet (Desktop e Web)
        ├── theme.py              # Paleta Claro/Escuro, badges e tokens de design
        ├── app.py                # Barra de título, abas e gerenciador de temas
        ├── dialogs.py            # Modais compartilhados: Zoom/Pan, Histograma Estatístico e Entranhas do Processo
        ├── common.py             # Utilitários visuais, cards e conversores Data-URI
        ├── components/           # Componentes customizados reutilizáveis
        │   └── histogram_chart.py # Histograma nativo acelerado via Flutter Canvas
        └── views/
            ├── single_view.py    # Aba de processamento individual, zoom, dithering e comparações
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

Divide linearmente o intervalo dinâmico $[0, 255]$ em $N = 2^b$ intervalos de largura igual e reconstrói no ponto médio (centróide) do intervalo:
$$\Delta = \frac{256}{N}, \quad q(x) = \left\lfloor \frac{x}{\Delta} \right\rfloor \cdot \Delta + \frac{\Delta}{2}$$

---

### 3. Quantização Não-Uniforme (K-Means)

Encontra $k = 2^b$ centróides $C = \{c_1, c_2, \dots, c_k\}$ que minimizam a inércia intra-cluster (soma dos quadrados dos resíduos):
$$J = \sum_{j=1}^k \sum_{x \in S_j} \|x - c_j\|^2$$
Adaptando os níveis de cinza ou a paleta de cores 3D (RGB) às regiões de maior densidade de pixels.

---

### 4. Quantização Baseada em Histograma (Quantis)

Particiona o histograma acumulado em $N = 2^b$ intervalos equiprováveis via percentis, garantindo que cada faixa de quantização contenha aproximadamente a mesma quantidade de pixels da imagem original.

---

### 5. Dithering por Difusão de Erro (Floyd-Steinberg)

Algoritmo adaptativo espacial (Floyd & Steinberg, 1976) que propaga o resíduo de quantização $e = x - q(x)$ para os 4 vizinhos adjacentes ainda não quantizados usando a matriz de pesos:

$$\begin{bmatrix}
& * & \frac{7}{16} \\
\frac{3}{16} & \frac{5}{16} & \frac{1}{16}
\end{bmatrix}$$

Essa difusão quebra os contornos abruptos (*false contouring* ou *banding*), convertendo o erro de quantização em ruído de alta frequência (azul), muito menos perceptível ao sistema visual humano.

---

### 6. Métricas de Fidelidade

- **MSE** (*Mean Squared Error*):
  $$\text{MSE} = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W (I_{\text{orig}}[i,j] - I_{\text{quant}}[i,j])^2$$

- **PSNR** (*Peak Signal-to-Noise Ratio*):
  $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right) \quad \text{[dB]}$$

- **MAE** (*Mean Absolute Error*):
  $$\text{MAE} = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W |I_{\text{orig}}[i,j] - I_{\text{quant}}[i,j]|$$

---

## 🛠️ Tecnologias & Dependências

| Biblioteca | Versão | Finalidade |
|---|---|---|
| `flet` | `>=0.86.0` | Interface gráfica declarativa moderna para Desktop e Web com Canvas acelerado |
| `numpy` | `>=1.24.0` | Processamento matricial vetorizado de alto desempenho e mapa térmico |
| `scikit-image` | `>=0.21.0` | Algoritmos de processamento de imagens e I/O |
| `scikit-learn` | `>=1.3.0` | K-Means adaptativo carregado sob demanda (*Lazy Loading*) |
| `matplotlib` | `>=3.7.0` | Geração de figuras analíticas estáticas de exportação |
| `pillow` | `>=10.0.0` | Downscaling preventivo, thumbnails e conversão de buffers em memória |

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos na Universidade Federal do Maranhão (UFMA).
Distribuído sob licença MIT.

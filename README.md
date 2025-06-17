# Projeto final de PAD: Previsão de Demanda

Este repositório contém o projeto desenvolvido para a Previsão de Demanda, como parte do trabalho Final de PAD. O objetivo principal é analisar dados de demanda e realizar visualizações para obter insights de valor.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
.
├── .gitignore
├── Data Visualization/
│   ├── CARGA_ENERGIA.ods
│   ├── DADOS_BRUTOS_INPUTADO.csv
│   ├── DADOS_BRUTOS_MES_ANO.csv
│   ├── data_visualization.ipynb
│   └── requirements.txt
└── README.md
```

- **`Data Visualization/`**: Pasta contendo os notebooks Jupyter e arquivos de dados relacionados à visualização e análise de dados.
    - **`data_visualization.ipynb`**: Notebook Jupyter com visualizações de dados mais aprofundadas.
    - **`requirements.txt`**: Arquivo listando as dependências do Python para os notebooks.

## Como Utilizar a Visualização

Os notebooks Jupyter na pasta `Data Visualization/` são as principais ferramentas para explorar e visualizar os dados. Siga os passos abaixo para utilizá-los:

### Pré-requisitos

Certifique-se de ter o Python e o Jupyter Notebook instalados em seu ambiente. Se não tiver, você pode instalá-los através do Anaconda (recomendado) ou pip:

```bash
# Instalação do Anaconda (recomendado para gerenciar ambientes Python e Jupyter)
# Siga as instruções em https://www.anaconda.com/products/distribution

# Ou instalação via pip:
pip install jupyter
```

### Configuração do Ambiente

Clone o repositório:

```bash
git clone https://github.com/AndreKoraleski/PADfinalPrevDemanda.git
cd PADfinalPrevDemanda
```

Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r "Data Visualization/requirements.txt"
```

### Executando os Notebooks

Inicie o Jupyter Notebook:

```bash
jupyter notebook
```

Uma nova aba ou janela do navegador será aberta, mostrando o diretório do projeto. Navegue até a pasta `Data Visualization/`.

Abra e explore os notebooks:

- **data_visualization.ipynb**: Este notebook aprofunda-se nas visualizações, apresentando gráficos mais complexos e insights sobre a demanda, tendências, sazonalidade, etc. É aqui que você encontrará as visualizações finais e mais elaboradas do projeto.

Execute as células: Dentro de cada notebook, você pode executar as células individualmente (clicando na célula e pressionando Shift + Enter) ou todas as células em sequência (no menu "Cell" -> "Run All").

---

### Utilizando com Google Colab

Você pode executar os notebooks diretamente no Google Colab, uma plataforma baseada em nuvem que não requer instalação local.

#### Como fazer

1. **Abra o notebook diretamente no Colab:**

   * [`data_visualization.ipynb`](https://colab.research.google.com/github/AndreKoraleski/PADfinalPrevDemanda/blob/main/Data%20Visualization/data_visualization.ipynb)

2. **Clone apenas os arquivos necessários dentro do notebook:**

   * Descomente e execute a primeira célula para baixar os arquivos essenciais e instalar os pacotes em `requirements.txt`:
   
     ```python
     !wget -q https://raw.githubusercontent.com/AndreKoraleski/PADfinalPrevDemanda/raw/main/Data%20Visualization/requirements.txt
     !wget -q https://github.com/AndreKoraleski/PADfinalPrevDemanda/raw/main/Data%20Visualization/CARGA_ENERGIA.ods
     !pip install -r requirements.txt
     ```

3. **Execute as demais células normalmente:**

   * Use `Shift + Enter` para executar célula a célula, ou em **Runtime > Run all** para executar todo o notebook.

> **Importante:** O arquivo `.ods` e o `requirements.txt` ficarão disponíveis temporariamente enquanto o ambiente do Colab estiver ativo.

---

## Entendendo as Visualizações

Para cada visualização nos notebooks, preste atenção aos seguintes pontos:

- **Título do Gráfico**: Indica o que o gráfico está representando.
- **Rótulos dos Eixos**: Descrevem as variáveis que estão sendo plotadas.
- **Legendas**: Explicam diferentes categorias ou séries de dados no gráfico.
- **Análise Textual**: Muitas células de código nos notebooks são acompanhadas de células de Markdown com explicações e interpretações dos gráficos. Leia-as cuidadosamente para entender os insights extraídos.

## Dados

Os dados utilizados neste projeto são fornecidos nos arquivos CSV na pasta `Data Visualization/`:

- **CARGA_ENERGIA.ods**: Contém os dados brutos e da forma que resultou no melhor acesso até o momento.
- **DADOS_BRUTOS_INPUTADO.csv**: Contém os dados de demanda brutos, onde valores ausentes foram tratados (imputados). 
- **DADOS_BRUTOS_MES_ANO.csv**: Uma versão agregada dos dados brutos, mostrando a demanda por mês e ano. Pode ser útil para análises de séries temporais.

## Links Adicionais

- [Docs com análise do AGEMC no projeto FMF](https://docs.google.com/document/d/1PPNHQCxLLV-mb5vX07GgL9dPnaOQVnEGOMxIwyFBJX4/edit?usp=sharing)
- [Fonte de dados](https://dados.ons.org.br/dataset/carga-energia)
- [Reunião Semana 1 (02/06/2025)](https://docs.google.com/document/d/1S8HmkO4C1JfwL6B_pDNTo4M_6as4lxON-TS3fyrHrns/edit?usp=sharing)
- [Reunião Semana 2 (09/06/2025)](https://docs.google.com/document/d/1S8HmkO4C1JfwL6B_pDNTo4M_6as4lxON-TS3fyrHrns/edit?usp=sharing)
- [Reunião Semana 3 (16/06/2025)](https://docs.google.com/document/d/1S8HmkO4C1JfwL6B_pDNTo4M_6as4lxON-TS3fyrHrns/edit?usp=sharing)


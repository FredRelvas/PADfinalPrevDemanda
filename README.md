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
│   ├── data_seeing.ipynb
│   ├── data_visualization.ipynb
│   └── requirements.txt
└── README.md
```

- **`Data Visualization/`**: Pasta contendo os notebooks Jupyter e arquivos de dados relacionados à visualização e análise de dados.
    - **`data_seeing.ipynb`**: Notebook Jupyter para exploração e visualização inicial dos dados.
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

- **data_seeing.ipynb**: Este notebook é ideal para uma primeira olhada nos dados, entendendo suas características, distribuições e possíveis anomalias. Ele contém resultados exploratórios simples sobre como os dados estão estruturados.
- **data_visualization.ipynb**: Este notebook aprofunda-se nas visualizações, apresentando gráficos mais complexos e insights sobre a demanda, tendências, sazonalidade, etc. É aqui que você encontrará as visualizações finais e mais elaboradas do projeto.

Execute as células: Dentro de cada notebook, você pode executar as células individualmente (clicando na célula e pressionando Shift + Enter) ou todas as células em sequência (no menu "Cell" -> "Run All").

---

### Utilizando com Google Colab

Você também pode executar os notebooks diretamente no Google Colab, uma plataforma baseada em nuvem que não requer nenhuma instalação local.

Para utilizar os notebooks no Colab, siga estes passos:

1.  **Abra o Google Colab:**
    * Vá para [colab.research.google.com](https://colab.research.google.com/)

2.  **Clone o repositório:**
    * No Colab, crie um novo notebook (`File > New notebook`).
    * Na primeira célula de código, execute o seguinte comando para clonar o repositório:
        ```python
        !git clone https://github.com/AndreKoraleski/PADfinalPrevDemanda.git
        ```
    * Após a execução, você verá a pasta `PADfinalPrevDemanda` listada no painel de arquivos à esquerda.

3.  **Navegue para o diretório correto e instale as dependências:**
    * Crie uma nova célula de código e execute os comandos abaixo para mudar para o diretório do projeto e instalar as bibliotecas necessárias:
        ```python
        %cd PADfinalPrevDemanda/Data\ Visualization/
        !pip install -r requirements.txt
        ```

4.  **Abra e execute os notebooks:**
    * Agora você pode abrir os notebooks `data_seeing.ipynb` e `data_visualization.ipynb` diretamente no Colab (no painel de arquivos à esquerda, navegue até `PADfinalPrevDemanda/Data Visualization/` e clique nos notebooks).
    * Dentro de cada notebook, você pode executar as células individualmente (clicando na célula e pressionando `Shift + Enter`) ou todas as células em sequência (no menu "Runtime" -> "Run all").

    * **Observação:** Ao abrir os notebooks, o Colab pode solicitar para salvar uma cópia no seu Google Drive. Sinta-se à vontade para salvar uma cópia para fazer suas próprias modificações.

## Entendendo as Visualizações

Para cada visualização nos notebooks, preste atenção aos seguintes pontos:

- **Título do Gráfico**: Indica o que o gráfico está representando.
- **Rótulos dos Eixos**: Descrevem as variáveis que estão sendo plotadas.
- **Legendas**: Explicam diferentes categorias ou séries de dados no gráfico.
- **Análise Textual**: Muitas células de código nos notebooks são acompanhadas de células de Markdown com explicações e interpretações dos gráficos. Leia-as cuidadosamente para entender os insights extraídos.

## Dados

Os dados utilizados neste projeto são fornecidos nos arquivos CSV na pasta `Data Visualization/`:

- **CARGA_ENERGIA.ods**: Contém os dados brutos e da forma que resultou no melhor acesso até o momento.
- **DADOS_BRUTOS_INPUTADO.csv**: Contém os dados de demanda brutos, onde valores ausentes foram tratados (imputados). Este é o dataset principal para a análise.
- **DADOS_BRUTOS_MES_ANO.csv**: Uma versão agregada dos dados brutos, mostrando a demanda por mês e ano. Útil para análises de séries temporais.

## Links Adicionais

- [Docs com análise do AGEMC no projeto FMF](https://docs.google.com/document/d/1PPNHQCxLLV-mb5vX07GgL9dPnaOQVnEGOMxIwyFBJX4/edit?usp=sharing)
- [Fonte de dados](https://dados.ons.org.br/dataset/carga-energia)
- [Reunião Semana 1 (02/06/2025)](https://docs.google.com/document/d/1S8HmkO4C1JfwL6B_pDNTo4M_6as4lxON-TS3fyrHrns/edit?usp=sharing)
- Reunião Semana 2 (09/06/2025, em breve o relatório)


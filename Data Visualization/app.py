# ==============================================================================
# --- 1. Importação das Bibliotecas ---
# ==============================================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path
import warnings

# --- Configurações Iniciais da Página ---
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Dashboard de Análise de Carga")

# ==============================================================================
# --- 2. Funções de Carregamento e Plotagem ---
# ==============================================================================

@st.cache_data
def carregar_dados():
    """
    Carrega os dados do arquivo .ods, converte a coluna de data,
    renomeia colunas e define o índice como a data.
    A anotação @st.cache_data garante que os dados sejam carregados apenas uma vez.
    """

    NOME_ARQUIVO_DADOS = 'CARGA_ENERGIA.ods'
    CAMINHO_DADOS = Path(__file__).parent / NOME_ARQUIVO_DADOS
    try:
        df = pd.read_excel(CAMINHO_DADOS, thousands='.', decimal=',')
        df.columns = df.columns.str.strip()
        df['din_instante'] = pd.to_datetime(df['din_instante'], format='mixed', dayfirst=True)
        df.rename(columns={'val_cargaenergiamwmed': 'valor', 'nom_subsistema': 'regiao'}, inplace=True)
        df['valor'] = df['valor'].ffill()
        df.set_index('din_instante', inplace=True)
        return df
    except (FileNotFoundError, KeyError) as e:
        st.error(f"Erro ao carregar os dados: {e}. Verifique se o arquivo 'CARGA_ENERGIA.ods' está presente e contém as colunas corretas.")
        return None

def plot_carga_agrupada(df: pd.DataFrame, periodo: str, modo_regiao: str):
    """
    Gera um gráfico de linha interativo da carga total ao longo do tempo.
    """

    title = "Carga Total Agrupada - Brasil"
    color = None
    if modo_regiao == 'Individual':
        title = "Carga Total Agrupada por Região"
        color = 'regiao'
    
    df_plot = df.groupby(by=['regiao', pd.Grouper(freq=periodo)] if color else [pd.Grouper(freq=periodo)])['valor'].sum().reset_index()
    fig = px.line(df_plot, x='din_instante', y='valor', color=color, title=title, labels={'regiao': 'Região', 'din_instante': 'Data', 'valor': 'Carga de Energia (MW)'})
    st.plotly_chart(fig, use_container_width=True)

def plot_decomposicao(df: pd.DataFrame, periodo: str, modo_regiao: str):
    """
    Gera gráficos de decomposição (Tendência, Sazonalidade, Resíduos) para cada região.
    """

    st.header("Decomposição da Série Temporal")
    regioes = df['regiao'].unique() if modo_regiao == 'Individual' else ['Brasil']
    for regiao in regioes:
        with st.container(border=True):
            st.subheader(f"Componentes para: {regiao}")
            series_df = df[df['regiao'] == regiao] if modo_regiao == 'Individual' else df
            series = series_df['valor'].resample(periodo).mean().dropna()
            if len(series) > 24: # A decomposição precisa de pelo menos 2 ciclos completos
                decomposition = seasonal_decompose(series, model='additive', period=12)
                fig = decomposition.plot()
                fig.set_size_inches(12, 8)
                st.pyplot(fig)
            else:
                st.warning(f"Dados insuficientes para gerar decomposição. Selecione um período maior.")

def plot_histograma(df: pd.DataFrame, modo_regiao: str):
    """
    Gera um histograma interativo da distribuição dos valores de carga.
    """

    color = 'regiao' if modo_regiao == 'Individual' else None
    fig = px.histogram(df, x='valor', color=color, nbins=50, title="Distribuição dos Valores de Carga", labels={'valor': 'Carga (MW)', 'regiao': 'Região'})
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_anual(df: pd.DataFrame, modo_regiao: str):
    """
    Gera um mapa de calor para visualizar a sazonalidade anual/mensal.
    """

    regioes = df['regiao'].unique() if modo_regiao == 'Individual' else ['Brasil']
    for regiao in regioes:
        with st.container(border=True):
            st.subheader(f"Mapa de Calor para: {regiao}")
            df_regiao = df[df['regiao'] == regiao] if modo_regiao == 'Individual' else df
            heatmap_data = df_regiao.pivot_table(values='valor', index=df_regiao.index.year, columns=df_regiao.index.month, aggfunc='mean')
            heatmap_data.columns = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            fig = px.imshow(heatmap_data, labels=dict(x="Mês", y="Ano", color="Carga Média"), title=f"Carga Média Mensal - {regiao}")
            st.plotly_chart(fig, use_container_width=True)

def plot_acf_chart(df: pd.DataFrame, periodo: str, modo_regiao: str):
    """
    Gera o gráfico de Autocorrelação (ACF).
    """

    st.header("Autocorrelação (ACF)")
    regioes = df['regiao'].unique() if modo_regiao == 'Individual' else ['Brasil']
    for regiao in regioes:
        with st.container(border=True):
            st.subheader(f"ACF para: {regiao}")
            series = df[df['regiao'] == regiao]['valor'].resample(periodo).mean().dropna() if modo_regiao == 'Individual' else df['valor'].resample(periodo).mean().dropna()
            if not series.empty:
                fig, ax = plt.subplots()
                plot_acf(series, ax=ax, lags=min(50, len(series)//2 - 1))
                ax.set_title(f"Função de Autocorrelação - {regiao}")
                st.pyplot(fig)

def plot_boxplot_mes(df: pd.DataFrame, modo_regiao: str):
    """
    Gera um boxplot interativo da distribuição da carga por mês.
    """

    color = 'regiao' if modo_regiao == 'Individual' else None
    fig = px.box(df, x='mes', y='valor', color=color, title="Distribuição da Carga por Mês", labels={'mes': 'Mês', 'valor': 'Carga (MW)', 'regiao': 'Região'})
    st.plotly_chart(fig, use_container_width=True)

# --- Dicionários de Gráficos e Descrições ---
CHART_FUNCTIONS = {
    'Carga Total Agrupada': plot_carga_agrupada,
    'Gráfico de Decomposição': plot_decomposicao,
    'Histograma de Distribuição': plot_histograma,
    'Mapa de Calor Anual': plot_heatmap_anual,
    'Autocorrelação (ACF)': plot_acf_chart,
    'Distribuição por Mês': plot_boxplot_mes,
}
CHART_DESCRIPTIONS = {
    'Carga Total Agrupada': "Exibe a carga de energia ao longo do tempo. Permite ver múltiplas regiões em linhas separadas.",
    'Gráfico de Decomposição': "Quebra a série em: Tendência (direção geral), Sazonalidade (padrões repetitivos) e Resíduos (ruído). Gera um gráfico para cada região selecionada.",
    'Histograma de Distribuição': "Mostra a frequência dos valores de carga. Ajuda a entender se os valores se concentram em uma faixa ou são dispersos.",
    'Mapa de Calor Anual': "Visualiza a intensidade da carga para cada mês de cada ano. Excelente para comparar a sazonalidade entre anos. Gera um gráfico para cada região.",
    'Autocorrelação (ACF)': "Mostra a correlação da série com suas próprias versões defasadas (lags). Útil para identificar sazonalidade. Gera um gráfico para cada região.",
    'Distribuição por Mês': "Compara a distribuição (mediana, quartis, etc.) da carga de energia para cada mês do ano. Permite comparar regiões.",
}

# ==============================================================================
# --- 3. Corpo Principal da Aplicação ---
# ==============================================================================

# Carrega os dados uma única vez
df_original = carregar_dados()

# A aplicação só continua se os dados forem carregados com sucesso
if df_original is not None:
    st.title("Dashboard de Análise Exploratória de Carga de Energia")

    # --- BARRA LATERAL ---
    st.sidebar.title("Controles")

    # --- Filtro de Região ---
    st.sidebar.header("Filtro de Região")
    lista_regioes = ['Todos Aglutinados'] + sorted(df_original['regiao'].unique().tolist())
    regioes_selecionadas = st.sidebar.multiselect("Selecione:", options=lista_regioes, default=['Todos Aglutinados'])

    if not regioes_selecionadas:
        st.sidebar.error("Selecione ao menos uma opção.")
        st.stop()
    
    # --- Lógica de Filtragem de Dados ---
    
    # 1. Filtra por região
    modo_regiao = 'Aglutinados' if 'Todos Aglutinados' in regioes_selecionadas else 'Individual'
    df_regiao_filtrada = df_original
    if modo_regiao == 'Individual':
        df_regiao_filtrada = df_original[df_original['regiao'].isin(regioes_selecionadas)]

    # 2. Validação para evitar erros se a seleção de região resultar em dados vazios
    if df_regiao_filtrada.empty:
        st.warning("Nenhum dado encontrado para a região selecionada.")
        st.stop()
        
    if not isinstance(df_regiao_filtrada.index, pd.DatetimeIndex):
        st.error("O índice do DataFrame não é do tipo data/hora, algo deu errado no carregamento.")
        st.stop()

    # 3. Controles de Período de Análise
    st.sidebar.header("Período de Análise")
    ano_min, ano_max = df_regiao_filtrada.index.year.min(), df_regiao_filtrada.index.year.max()
    start_year, end_year = st.sidebar.slider("Intervalo de Anos:", ano_min, ano_max, (ano_min, ano_max))
    start_month, end_month = st.sidebar.slider("Intervalo de Meses:", 1, 12, (1, 12))
    
    # 4. Filtro final por data
    start_date, end_date = None, None
    try:
        start_date = pd.Timestamp(f'{start_year}-{start_month}-01')
        end_date = pd.Timestamp(f'{end_year}-{end_month}-01') + pd.offsets.MonthEnd(1)
        df_filtrado = df_regiao_filtrada.loc[start_date:end_date].copy()
    except Exception as e:
        st.error(f"Intervalo de datas inválido. Verifique os sliders. Erro: {e}")
        st.stop()

    # 5. Validação final após todos os filtros
    if df_filtrado.empty:
        st.warning("Não há dados disponíveis para os filtros selecionados. Por favor, ajuste o período ou a região.")
        st.stop()

    # 6. Criação segura de colunas para plotagem
    df_filtrado['mes'] = df_filtrado.index.month

    # --- Controles de Agrupamento e Seleção de Gráfico ---
    st.sidebar.header("Configuração de Agrupamento")
    opcoes_periodo = {'Mensal': 'ME', 'Semanal': 'W', 'Diário': 'D'}
    periodo_label = st.sidebar.selectbox("Agrupar dados por:", options=list(opcoes_periodo.keys()))
    periodo = opcoes_periodo[periodo_label]
    
    st.sidebar.header("Selecione a Visualização")
    lista_de_graficos = list(CHART_FUNCTIONS.keys())
    descricoes_graficos = [CHART_DESCRIPTIONS.get(nome, '') for nome in lista_de_graficos]
    
    grafico_selecionado = st.sidebar.radio(
        "Escolha um tipo de gráfico:",
        options=lista_de_graficos,
        captions=descricoes_graficos,
        index=0
    )

    # --- RENDERIZAÇÃO DA PÁGINA PRINCIPAL ---
    st.header(f"Visualização: {grafico_selecionado}")
    
    if start_date and end_date:
        st.markdown(f"**Período:** `{start_date.strftime('%d/%m/%Y')}` a `{end_date.strftime('%d/%m/%Y')}` | **Regiões:** `{', '.join(regioes_selecionadas)}`")
    
    plot_function = CHART_FUNCTIONS.get(grafico_selecionado)
    if plot_function:
        if grafico_selecionado in ['Carga Total Agrupada', 'Gráfico de Decomposição', 'Autocorrelação (ACF)']:
            plot_function(df_filtrado, periodo=periodo, modo_regiao=modo_regiao)
        else:
            plot_function(df_filtrado, modo_regiao=modo_regiao)

    # --- TABELA DE DADOS ---
    with st.expander("Visualizar Dados Filtrados (DD/MM/AAAA)"):
        df_display = df_filtrado.copy()
        df_display.index = df_display.index.strftime('%d/%m/%Y')
        df_display = df_display.rename(columns={'valor': 'Carga de Energia (MW)', 'regiao': 'Região'})
        st.dataframe(df_display[['Região', 'Carga de Energia (MW)']], use_container_width=True)
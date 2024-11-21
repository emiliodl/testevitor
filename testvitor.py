import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt


def decomposicao_stl(dados, nome_unico, mes_inicio, mes_fim):
    """
    Realiza a decomposição STL, calcula as forças de tendência e sazonalidade, e plota os resultados.
    """
    # Renomear a coluna 'Município' para 'nome_unico'
    dados.rename(columns={'Município': 'nome_unico'}, inplace=True)
    dados['nome_unico'] = dados['nome_unico'].str.strip().str.upper()

    # Normalizar o nome fornecido
    nome_unico = nome_unico.strip().upper()

    # Filtrar os dados para o município selecionado
    serie_dados = dados[dados['nome_unico'] == nome_unico]

    if serie_dados.empty:
        raise ValueError(
            f"Nenhum dado encontrado para o município: {nome_unico}")

    # Filtrar colunas com datas no intervalo especificado
    colunas_data = [col for col in dados.columns if '/' in col]

    # Mapeamento de meses em português para inglês
    meses_pt_en = {
        'Jan': 'Jan', 'Fev': 'Feb', 'Mar': 'Mar', 'Abr': 'Apr', 'Mai': 'May', 'Jun': 'Jun',
        'Jul': 'Jul', 'Ago': 'Aug', 'Set': 'Sep', 'Out': 'Oct', 'Nov': 'Nov', 'Dez': 'Dec'
    }

    # Substituir meses em português por inglês
    colunas_convertidas = [
        col.replace('/', '-').replace(col.split('/')
                                      [1], meses_pt_en[col.split('/')[1]])
        for col in colunas_data
    ]

    # Converter para datas
    datas = pd.to_datetime(colunas_convertidas, format='%Y-%b')

    # Filtrar pelo intervalo de datas
    data_inicio = pd.to_datetime(mes_inicio)
    data_fim = pd.to_datetime(mes_fim)
    colunas_filtradas = [col for col, data in zip(
        colunas_data, datas) if data_inicio <= data <= data_fim]

    if not colunas_filtradas:
        raise ValueError(
            "Nenhuma coluna corresponde ao intervalo de datas selecionado.")

    # Criar a série temporal
    serie = serie_dados[colunas_filtradas].iloc[0].replace(
        '-', 0).astype(float)
    indice_tempo = pd.date_range(
        start=data_inicio, periods=len(colunas_filtradas), freq='M')
    serie.index = indice_tempo[:len(serie)]

    # Realizar a decomposição STL
    stl = STL(serie, period=12)  # Dados mensais assumem período anual
    resultado = stl.fit()

    # Extrair componentes
    tendencia = resultado.trend
    sazonalidade = resultado.seasonal
    residuos = resultado.resid

    # Calcular variâncias
    var_residuo = np.var(residuos)
    var_tend_residuo = np.var(tendencia + residuos)
    var_sazon_residuo = np.var(sazonalidade + residuos)

    # Calcular forças
    forca_tendencia = max(0, 1 - var_residuo / var_tend_residuo)
    forca_sazonalidade = max(0, 1 - var_residuo / var_sazon_residuo)

    st.write(f"**Força da Tendência (F_T):** {forca_tendencia:.3f}")
    st.write(f"**Força da Sazonalidade (F_S):** {forca_sazonalidade:.3f}")

    # Plotar a decomposição
    st.write("### Decomposição STL")
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    axes[0].plot(resultado.observed)
    axes[0].set_title('Observado')
    axes[1].plot(tendencia)
    axes[1].set_title('Tendência')
    axes[2].plot(sazonalidade)
    axes[2].set_title('Sazonalidade')
    axes[3].plot(residuos)
    axes[3].set_title('Resíduo')
    plt.tight_layout()
    st.pyplot(fig)


# Streamlit app
st.title("Decomposição STL de Séries Temporais")

uploaded_file = st.file_uploader("Faça upload do arquivo CSV", type="csv")

if uploaded_file is not None:
    # Ler o arquivo com a codificação correta
    dados = pd.read_csv(uploaded_file, encoding='latin1', sep=';')
    st.write("Pré-visualização dos Dados:")
    st.dataframe(dados.head())

    # Inputs do usuário
    nome_unico = st.text_input(
        "Nome e ID do município ex: 110150 SERINGUEIRAS")
    mes_inicio = st.text_input(
        "Mês de Início (formato: 'AAAA-MM')", value="2008-01")
    mes_fim = st.text_input("Mês de Fim (formato: 'AAAA-MM')", value="2023-12")

    # Botão para executar a análise
    if st.button("Executar Decomposição STL"):
        try:
            decomposicao_stl(dados, nome_unico, mes_inicio, mes_fim)
        except Exception as e:
            st.error(f"Erro ao executar a decomposição: {e}")

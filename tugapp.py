import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar  # <-- corrigido
from processamento import balanceProcessing, jumpProcessing, tugProcessing, ytestProcessing, jointSenseProcessing
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
from textwrap import dedent

# --------- Config da página ---------
st.set_page_config(page_title="Momentum TUG", page_icon="⚡", layout="wide")

# --------- Estilo ---------
st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 40%, #e6e6e6 100%);
      }
      header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 40%, #e6e6e6 100%) !important;
      }
      .block-container { background: transparent; }
      section[data-testid="stSidebar"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Momentum TUG</h1>", unsafe_allow_html=True)

@st.cache_data
def carregar_dados_generico(arquivo):
    """
    Lê CSV com 4 ou 5 colunas.
    - 5 colunas: descarta a 1ª (ex.: índice/metadata) e usa as colunas 2..5
    - 4 colunas: usa todas
    Retorna DataFrame com colunas: ["Tempo", "X", "Y", "Z"] ou None em caso de erro.
    """
    try:
        df = pd.read_csv(arquivo, sep=None, engine='python')  # autodetecta separador

        if df.shape[1] == 5:
            dados = df.iloc[:, 1:5].copy()  # usa colunas 2..5
        elif df.shape[1] == 4:
            dados = df.iloc[:, 0:4].copy()  # usa todas
        else:
            st.error("O arquivo deve conter 4 ou 5 colunas com cabeçalhos.")
            return None

        dados.columns = ["Tempo", "X", "Y", "Z"]
        return dados

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

pagina = st.sidebar.radio("📂 Navegue pelas páginas", [ "🏠 Página Inicial", "⬆️ Importar Dados", "📈 Visualização Gráfica", "📤 Exportar Resultados", "📖 Referências bibliográficas" ])

# === Página Inicial ===
if pagina == "🏠 Página Inicial": # texto descritivo mais bonito
    html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;">
    <p><b>Bem-vindo ao Momentum TUG</b>,
    a aplicação Web para análise de dados de protocolos de avaliação do <i>iTUG</i>.
    </p>
    </div> """) 

    st.markdown(html, unsafe_allow_html=True)

# === Página de Importação ===
elif pagina == "⬆️ Importar Dados":
    st.title("⬆️ Importar Dados")
    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col1:
        tipo_teste = st.selectbox( "Qual teste você deseja analisar?", ["Selecione...","TUG"] )
        if tipo_teste != "Selecione...":
            st.session_state["tipo_teste"] = tipo_teste
            
        elif tipo_teste == "TUG":
             st.subheader("📱 Importar dados dos sensores")
             arquivo = st.file_uploader( "Selecione o arquivo do acelerômetro (CSV ou TXT)", type=["csv", "txt"])
             if arquivo is not None:
                 dados_acc = carregar_dados_generico(arquivo)
                 if dados_acc is not None:
                     st.success("Arquivo carregado com sucesso!")
                     st.dataframe(dados_acc.head())
                     st.session_state["dados_acc"] = dados_acc
                     st.session_state["dados"] = dados_acc
                     arquivo = st.file_uploader("Selecione o arquivo do giroscópico (CSV ou TXT)", type=["csv", "txt"])
                     if arquivo is not None:
                         dados_gyro = carregar_dados_generico(arquivo)
                         if dados_gyro is not None:
                             st.success("Arquivo carregado com sucesso!")
                             st.dataframe(dados_gyro.head())
                             st.session_state["dados_gyro"] = dados_gyro
                             st.session_state["tipo_teste"] = tipo_teste
                             


                    
                           

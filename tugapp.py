import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Normas Acc & Gyro", page_icon="📱", layout="centered")
st.title("📱 Normas do Acelerômetro e Giroscópio")

@st.cache_data
def carregar_dados(arquivo):
    """
    Lê CSV/TXT com 4 ou 5 colunas.
    - 5 colunas: usa colunas 2..5 (ignora a primeira)
    - 4 colunas: usa colunas 1..4
    Retorna DataFrame com colunas: Tempo, X, Y, Z.
    """
    df = pd.read_csv(arquivo, sep=None, engine="python")
    if df.shape[1] == 5:
        dados = df.iloc[:, 1:5].copy()
    elif df.shape[1] == 4:
        dados = df.iloc[:, 0:4].copy()
    else:
        raise ValueError("O arquivo deve ter 4 ou 5 colunas.")

    dados.columns = ["Tempo", "X", "Y", "Z"]
    return dados

col1, col2 = st.columns(2)

with col1:
    arq_acc = st.file_uploader("Arquivo do *acelerômetro*", type=["csv", "txt"], key="acc")
with col2:
    arq_gyro = st.file_uploader("Arquivo do *giroscópio*", type=["csv", "txt"], key="gyro")

if arq_acc is not None and arq_gyro is not None:
    try:
        df_acc = carregar_dados(arq_acc)
        df_gyro = carregar_dados(arq_gyro)

        # Calcula norma |v| = sqrt(x² + y² + z²)
        df_acc["Norma"] = np.sqrt(df_acc["X"]*2 + df_acc["Y"]2 + df_acc["Z"]*2)
        df_gyro["Norma"] = np.sqrt(df_gyro["X"]*2 + df_gyro["Y"]2 + df_gyro["Z"]*2)

        st.subheader("Prévia dos dados")
        st.write("Acelerômetro:")
        st.dataframe(df_acc.head())
        st.write("Giroscópio:")
        st.dataframe(df_gyro.head())

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        axes[0].plot(df_acc["Tempo"], df_acc["Norma"])
        axes[0].set_ylabel("‖a‖")
        axes[0].set_title("Norma do Acelerômetro")

        axes[1].plot(df_gyro["Tempo"], df_gyro["Norma"])
        axes[1].set_ylabel("‖ω‖")
        axes[1].set_xlabel("Tempo")
        axes[1].set_title("Norma do Giroscópio")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar arquivos: {e}")

else:
    st.info("Faça o upload dos dois arquivos para ver os gráficos.")
                             


                    
                           

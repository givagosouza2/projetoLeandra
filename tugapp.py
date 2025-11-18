import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="Normas Acc & Gyro", page_icon="📱", layout="centered")
st.title("📱 Normas do Acelerômetro e Giroscópio (com Filtro 2 Hz)")

# -------------------------
# Função de carregamento
# -------------------------
@st.cache_data
def carregar_dados(arquivo):
    df = pd.read_csv(arquivo, sep=None, engine="python")

    if df.shape[1] == 5:
        dados = df.iloc[:, 1:5].copy()
    elif df.shape[1] == 4:
        dados = df.iloc[:, 0:4].copy()
    else:
        raise ValueError("O arquivo deve ter 4 ou 5 colunas.")

    dados.columns = ["Tempo", "X", "Y", "Z"]
    return dados

# -------------------------
# Filtro passa-baixa
# -------------------------
def lowpass_filter(series, fs, cutoff=2, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, series)

# -------------------------
# Upload
# -------------------------
col1, col2 = st.columns(2)

with col1:
    arq_acc = st.file_uploader("Arquivo do acelerômetro", type=["csv", "txt"], key="acc")
with col2:
    arq_gyro = st.file_uploader("Arquivo do giroscópio", type=["csv", "txt"], key="gyro")

# -------------------------
# Processamento
# -------------------------
if arq_acc is not None and arq_gyro is not None:
    try:
        df_acc = carregar_dados(arq_acc)
        df_gyro = carregar_dados(arq_gyro)

        # ====== Estimar Fs ======
        dt_acc = np.diff(df_acc["Tempo"]).mean()
        dt_gyro = np.diff(df_gyro["Tempo"]).mean()
        fs_acc = 1 / dt_acc
        fs_gyro = 1 / dt_gyro

        st.write(f"*Fs acelerômetro estimado:* {fs_acc:.2f} Hz")
        st.write(f"*Fs giroscópio estimado:* {fs_gyro:.2f} Hz")

        # ====== Calcular norma ======
        df_acc["Norma_raw"] = np.sqrt(df_acc["X"]**2 + df_acc["Y"]**2 + df_acc["Z"]**2)
        df_gyro["Norma_raw"] = np.sqrt(df_gyro["X"]**2 + df_gyro["Y"]**2 + df_gyro["Z"]**2)

        # ====== Filtrar ======
        df_acc["Norma"] = lowpass_filter(df_acc["Norma_raw"], fs_acc)
        df_gyro["Norma"] = lowpass_filter(df_gyro["Norma_raw"], fs_gyro)

        # ====== Plot ======
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

        axes[0].plot(df_acc["Tempo"], df_acc["Norma_raw"], alpha=0.4, label="Bruto")
        axes[0].plot(df_acc["Tempo"], df_acc["Norma"], linewidth=2, label="Filtrado (2 Hz)")
        axes[0].set_ylabel("‖a‖")
        axes[0].set_title("Norma do Acelerômetro")
        axes[0].legend()

        axes[1].plot(df_gyro["Tempo"], df_gyro["Norma_raw"], alpha=0.4, label="Bruto")
        axes[1].plot(df_gyro["Tempo"], df_gyro["Norma"], linewidth=2, label="Filtrado (2 Hz)")
        axes[1].set_ylabel("‖ω‖")
        axes[1].set_xlabel("Tempo")
        axes[1].set_title("Norma do Giroscópio")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar arquivos: {e}")

else:
    st.info("Faça o upload dos dois arquivos para ver os gráficos.")
                             


                    
                           

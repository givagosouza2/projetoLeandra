import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans  # <= K-Means para discretizar os estados

st.set_page_config(page_title="Normas Acc & Gyro", page_icon="📱", layout="centered")
st.title("📱 Normas do Acelerômetro e Giroscópio (com Filtro 2 Hz e Detecção de Início de Movimento)")

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
# Detecção de transição (cadeia de estados)
# -------------------------
def detectar_inicio_movimento(labels, base_class=0, min_run=5):
    """
    Encontra o primeiro índice em que ocorre:
      [base_class]*min_run seguido de [classe > base_class]*min_run.
    Retorna o índice do primeiro elemento da segunda sequência (início do movimento),
    ou None se não encontrar.
    """
    labels = np.asarray(labels)
    n = len(labels)
    janela = 2 * min_run

    for i in range(0, n - janela + 1):
        bloco1 = labels[i : i + min_run]
        bloco2 = labels[i + min_run : i + janela]

        if np.all(bloco1 == base_class) and np.all(bloco2 > base_class):
            return i + min_run  # primeiro índice da nova classe
    return None

# -------------------------
# Upload
# -------------------------
col1, col2 = st.columns(2)

with col1:
    arq_acc = st.file_uploader("Arquivo do acelerômetro", type=["csv", "txt"], key="acc")
with col2:
    arq_gyro = st.file_uploader("Arquivo do giroscópio", type=["csv", "txt"], key="gyro")

# Parâmetros do K-Means
k_classes = st.sidebar.number_input(
    "Número de classes (K-Means – giroscópio)",
    min_value=2,
    max_value=6,
    value=3,
    step=1
)
min_run = st.sidebar.number_input(
    "Comprimento mínimo da sequência (amostras)",
    min_value=3,
    max_value=20,
    value=5,
    step=1
)

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

        # Se Tempo está em ms (muitos celulares), faz sentido usar 1000/dt
        fs_acc = 1000 / dt_acc
        fs_gyro = 1000 / dt_gyro

        st.write(f"Fs acelerômetro estimado: {fs_acc:.2f} Hz")
        st.write(f"Fs giroscópio estimado: {fs_gyro:.2f} Hz")

        # ====== Calcular norma ======
        df_acc["Norma_raw"] = np.sqrt(df_acc["X"]*2 + df_acc["Y"]2 + df_acc["Z"]*2)
        df_gyro["Norma_raw"] = np.sqrt(df_gyro["X"]*2 + df_gyro["Y"]2 + df_gyro["Z"]*2)

        # ====== Filtrar ======
        df_acc["Norma"] = lowpass_filter(df_acc["Norma_raw"], fs_acc)
        df_gyro["Norma"] = lowpass_filter(df_gyro["Norma_raw"], fs_gyro)

        # ----- K-Means na norma filtrada do giroscópio -----
        valores = df_gyro["Norma"].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=k_classes, n_init=10, random_state=42)
        labels_raw = kmeans.fit_predict(valores)
        centros = kmeans.cluster_centers_.flatten()

        # Ordenar classes pelos centros (classe 0 = menores valores)
        ordem = np.argsort(centros)           # índices dos clusters do menor pro maior
        mapa = {old_label: rank for rank, old_label in enumerate(ordem)}
        labels = np.array([mapa[l] for l in labels_raw])

        df_gyro["Classe"] = labels

        # Detectar início de movimento (transição da classe 0 para qualquer > 0)
        idx_inicio = detectar_inicio_movimento(df_gyro["Classe"], base_class=0, min_run=min_run)

        if idx_inicio is not None:
            tempo_inicio = df_gyro["Tempo"].iloc[idx_inicio]
            st.success(f"Início de movimento detectado em ~ *t = {tempo_inicio:.2f}* (unidades do seu eixo Tempo).")
        else:
            tempo_inicio = None
            st.warning("Nenhuma transição estável (classe 0 → classe > 0) com as condições definidas foi encontrada.")

        # ====== Plot ======
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)

        # Acelerômetro
        axes[0].plot(df_acc["Tempo"], df_acc["Norma_raw"], alpha=0.4, label="Bruto")
        axes[0].plot(df_acc["Tempo"], df_acc["Norma"], linewidth=2, label="Filtrado (2 Hz)")
        axes[0].set_ylabel("‖a‖")
        axes[0].set_title("Norma do Acelerômetro")
        axes[0].legend()

        # Giroscópio
        axes[1].plot(df_gyro["Tempo"], df_gyro["Norma_raw"], alpha=0.3, label="Bruto")
        axes[1].plot(df_gyro["Tempo"], df_gyro["Norma"], linewidth=2, label="Filtrado (2 Hz)")

        # Se achou o início, desenha linha vertical
        if tempo_inicio is not None:
            axes[1].axvline(tempo_inicio, linestyle="--", linewidth=2, label="Início do movimento (Markov+K-Means)")

        axes[1].set_ylabel("‖ω‖")
        axes[1].set_xlabel("Tempo")
        axes[1].set_title("Norma do Giroscópio + Detecção de Mudança de Classe")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Opcional: mostrar tabela resumida das classes
        with st.expander("Ver primeiros valores e classes do giroscópio"):
            st.dataframe(df_gyro[["Tempo", "Norma", "Classe"]].head(50))

    except Exception as e:
        st.error(f"Erro ao processar arquivos: {e}")

else:
    st.info("Faça o upload dos dois arquivos para ver os gráficos e a detecção do início de movimento.")


                    
                           

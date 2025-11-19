import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from sklearn.cluster import KMeans  # <= K-Means para discretizar os estados

st.set_page_config(page_title="Gyro ML – Markov", page_icon="📱", layout="centered")
st.title("📱 Giroscópio – Eixo Médio-Lateral (|ML|, 100 Hz + Detrend + Filtro 2 Hz + Cadeias de Markov)")

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
# Pré-processamento: interpola para 100 Hz, detrend, filtra, norma
# -------------------------
def preprocess_sensor(df, target_fs=100, cutoff=2):
    """
    df: DataFrame com colunas ["Tempo", "X", "Y", "Z"]
    Retorna df_proc com:
        Tempo (s, interpolado a 100 Hz),
        X_filt, Y_filt, Z_filt,
        Norma (a partir dos eixos filtrados),
        Norma_raw_interp (norma dos eixos interpolados sem filtro).
    """

    # Vetor de tempo original
    t_orig = df["Tempo"].values
    dt_orig = np.diff(t_orig).mean()

    # Heurística simples: se dt > 2, assumo Tempo em ms e converto para s
    if dt_orig > 2:  # algo do tipo 10, 20 ms...
        t_s = t_orig / 1000.0
    else:
        t_s = t_orig.astype(float)

    # Normaliza para começar em 0
    t_s = t_s - t_s[0]

    # Novo eixo de tempo uniforme a 100 Hz
    t_new = np.arange(t_s[0], t_s[-1], 1.0 / target_fs)

    # Interpolar cada eixo
    x_interp = np.interp(t_new, t_s, df["X"].values)
    y_interp = np.interp(t_new, t_s, df["Y"].values)
    z_interp = np.interp(t_new, t_s, df["Z"].values)

    # Detrend
    x_det = detrend(x_interp)
    y_det = detrend(y_interp)
    z_det = detrend(z_interp)

    # Filtro passa-baixa nos eixos detrended
    x_f = lowpass_filter(x_det, fs=target_fs, cutoff=cutoff)
    y_f = lowpass_filter(y_det, fs=target_fs, cutoff=cutoff)
    z_f = lowpass_filter(z_det, fs=target_fs, cutoff=cutoff)

    # Normas
    norma_raw_interp = np.sqrt(x_interp**2 + y_interp**2 + z_interp**2)
    norma_filt = np.sqrt(x_f**2 + y_f**2 + z_f**2)

    df_proc = pd.DataFrame({
        "Tempo": t_new,
        "X_filt": x_f,
        "Y_filt": y_f,
        "Z_filt": z_f,
        "Norma_raw_interp": norma_raw_interp,
        "Norma": norma_filt,
    })

    return df_proc, target_fs  # fs agora é exatamente target_fs

# -------------------------
# Detecção de início (cadeia de estados)
# -------------------------
def detectar_inicio_movimento(labels, base_class=0, min_run=5):
    """
    Início do movimento:
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
# Detecção de fim da ação usando o ESTADO DO FIM DO REGISTRO
# -------------------------
def detectar_fim_movimento(labels, min_run=5):
    """
    Determina o fim do movimento usando como estado de repouso
    o estado presente no FIM do registro (última amostra).

    Padrão procurado (de trás pra frente):
        [classe != estado_final]*min_run
        seguido de
        [estado_final]*min_run

    Retorna o índice do primeiro elemento da sequência de estado_final (fim do movimento),
    ou None se não encontrar.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2 * min_run:
        return None

    # Estado de repouso = estado da última amostra
    estado_final = labels[-1]

    janela = 2 * min_run

    for i in range(n - janela, -1, -1):
        bloco1 = labels[i : i + min_run]
        bloco2 = labels[i + min_run : i + janela]

        # movimento -> repouso (repouso = estado_final)
        if np.all(bloco1 != estado_final) and np.all(bloco2 == estado_final):
            return i + min_run  # primeiro índice da sequência em repouso

    return None

# -------------------------
# Upload
# -------------------------
col1, col2 = st.columns(2)

with col1:
    arq_acc = st.file_uploader("Arquivo do acelerômetro", type=["csv", "txt"], key="acc")
with col2:
    arq_gyro = st.file_uploader("Arquivo do giroscópio", type=["csv", "txt"], key="gyro")

# Parâmetros do K-Means (para |ML_gyro|)
k_classes = st.sidebar.number_input(
    "Número de classes (K-Means – |gyro ML|)",
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
        # ====== Carrega sinais crus ======
        df_acc_raw = carregar_dados(arq_acc)
        df_gyro_raw = carregar_dados(arq_gyro)

        # ====== 1) Inferir orientação pelos eixos X e Y do acelerômetro (média) ======
        mean_x = df_acc_raw["X"].mean()
        mean_y = df_acc_raw["Y"].mean()

        if abs(mean_x) >= abs(mean_y):
            eixo_vertical = "X"
            eixo_ml = "Y"
            g_est = mean_x
        else:
            eixo_vertical = "Y"
            eixo_ml = "X"
            g_est = mean_y

        st.subheader("📐 Orientação aproximada do smartphone")
        st.write(f"Média do eixo X (acc): {mean_x:.3f}")
        st.write(f"Média do eixo Y (acc): {mean_y:.3f}")
        st.success(
            f"Eixo **vertical** (gravidade) ≈ **{eixo_vertical}** "
            f"(|média| = {abs(g_est):.3f}); eixo **médio-lateral** ≈ **{eixo_ml}**."
        )
        st.caption("A detecção do movimento será feita no giroscópio **usando |ML_gyro|** (valor absoluto do eixo médio-lateral filtrado).")

        # ====== 2) Pré-processamento: interpola em 100 Hz, detrend, filtra, norma ======
        df_acc, fs_acc = preprocess_sensor(df_acc_raw, target_fs=100, cutoff=2)
        df_gyro, fs_gyro = preprocess_sensor(df_gyro_raw, target_fs=100, cutoff=2)

        st.write(f"Fs acelerômetro (após interpolação): {fs_acc:.2f} Hz")
        st.write(f"Fs giroscópio (após interpolação): {fs_gyro:.2f} Hz")

        # ====== 3) Definir eixos Vert/ML no acc e ML_gyro no gyro ======
        # Acelerômetro
        if eixo_vertical == "X":
            df_acc["Vert"] = df_acc["X_filt"]
            df_acc["ML"] = df_acc["Y_filt"]
        else:
            df_acc["Vert"] = df_acc["Y_filt"]
            df_acc["ML"] = df_acc["X_filt"]

        # Giroscópio – eixo médio-lateral e seu módulo
        if eixo_vertical == "X":
            df_gyro["ML_gyro"] = df_gyro["Y_filt"]
        else:
            df_gyro["ML_gyro"] = df_gyro["X_filt"]

        df_gyro["ML_gyro_abs"] = np.abs(df_gyro["ML_gyro"])

        # ====== 4) Aplicar K-Means em ML_gyro_abs (cadeias de Markov) ======
        ml_abs = df_gyro["ML_gyro_abs"].values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=k_classes, n_init=10, random_state=42)
        labels_raw = kmeans.fit_predict(ml_abs)
        centros = kmeans.cluster_centers_.flatten()

        # Ordenar classes pelos centros (classe 0 = menor |ML|, presumido repouso)
        ordem = np.argsort(centros)           # índices dos clusters do menor pro maior
        mapa = {old_label: rank for rank, old_label in enumerate(ordem)}
        labels = np.array([mapa[l] for l in labels_raw])

        df_gyro["Classe"] = labels

        st.write("Centros dos clusters em |ML_gyro| (ordenados):")
        for idx in range(k_classes):
            st.write(f"Classe {idx}: centro ≈ {np.sort(centros)[idx]:.4f}")

        # ====== 5) Detectar início e fim do movimento (usando classes em |ML_gyro|) ======
        idx_inicio = detectar_inicio_movimento(df_gyro["Classe"], base_class=0, min_run=min_run)
        idx_fim = detectar_fim_movimento(df_gyro["Classe"], min_run=min_run)

        tempo_inicio = None
        tempo_fim = None

        if idx_inicio is not None:
            tempo_inicio = df_gyro["Tempo"].iloc[idx_inicio]
            st.success(f"Início de movimento (|ML_gyro|) detectado em ~ *t = {tempo_inicio:.2f} s*.")
        else:
            st.warning("Nenhuma transição estável (classe 0 → classe > 0 em |ML_gyro|) foi encontrada para o INÍCIO.")

        if idx_fim is not None:
            tempo_fim = df_gyro["Tempo"].iloc[idx_fim]
            st.success(f"Fim de movimento (|ML_gyro|) detectado em ~ *t = {tempo_fim:.2f} s* (usando estado final do registro).")
        else:
            st.warning("Nenhuma transição estável para o estado final foi encontrada para o FIM da ação em |ML_gyro|.")

        # =========================
        # 6) PLOTS
        # =========================
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

        # Acelerômetro - norma
        axes[0].plot(df_acc["Tempo"], df_acc["Norma_raw_interp"], alpha=0.4, label="Norma interpolada (bruta)")
        axes[0].plot(df_acc["Tempo"], df_acc["Norma"], linewidth=2, label="Norma filtrada (detrend + 2 Hz)")
        axes[0].set_ylabel("‖a‖")
        axes[0].set_title("Norma do Acelerômetro (100 Hz)")
        axes[0].legend()

        # Acelerômetro - eixos Vert / ML
        axes[1].plot(df_acc["Tempo"], df_acc["Vert"], label=f"Eixo vertical acc ({eixo_vertical}_filt)")
        axes[1].plot(df_acc["Tempo"], df_acc["ML"], linestyle="--", label=f"Eixo médio-lateral acc ({eixo_ml}_filt)")
        axes[1].set_ylabel("a (filtrado)")
        axes[1].set_title("Componentes vertical e médio-lateral (acelerômetro)")
        axes[1].legend()

        # Giroscópio – |ML_gyro|
        axes[2].plot(df_gyro["Tempo"], df_gyro["ML_gyro_abs"], label="|Gyro médio-lateral| (|ML_gyro|)")

        if tempo_inicio is not None:
            axes[2].axvline(tempo_inicio, linestyle="--", linewidth=2,
                            label="Início movimento (|ML_gyro|)")
        if tempo_fim is not None:
            axes[2].axvline(tempo_fim, linestyle="--", linewidth=2,
                            label="Fim movimento (|ML_gyro|)")

        if (tempo_inicio is not None) and (tempo_fim is not None) and (tempo_fim > tempo_inicio):
            axes[2].axvspan(tempo_inicio, tempo_fim, alpha=0.15, label="Janela movimento (|ML_gyro|)")

        axes[2].set_ylabel("|ω ML| (filtrado)")
        axes[2].set_xlabel("Tempo (s)")
        axes[2].set_title("Giroscópio – Valor absoluto do eixo médio-lateral (|ML_gyro|) + Markov")
        axes[2].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # ====== Tabelas para inspeção ======
        with st.expander("Ver primeiros valores (gyro) com ML_gyro, |ML_gyro| e Classe"):
            st.dataframe(
                df_gyro[["Tempo", "ML_gyro", "ML_gyro_abs", "Classe"]].head(200)
            )

        with st.expander("Ver primeiros valores (acc) com Vert/ML"):
            st.dataframe(
                df_acc[["Tempo", "X_filt", "Y_filt", "Vert", "ML", "Norma"]].head(200)
            )

    except Exception as e:
        st.error(f"Erro ao processar arquivos: {e}")

else:
    st.info("Faça o upload dos dois arquivos para ver a orientação do smartphone e a detecção de início/fim do movimento em |ML_gyro|.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from sklearn.cluster import KMeans  # <= K-Means para discretizar os estados

st.set_page_config(page_title="Gyro ML & Z – Markov", page_icon="📱", layout="centered")
st.title("📱 Giroscópio – ML (|ML|) para movimento e Z para transientes (100 Hz + Detrend + Filtro 2 Hz)")

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

    # Normas (não usadas diretamente aqui, mas úteis se precisar depois)
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
# Detecção de TODOS os transientes dentro da janela de movimento
# (usando labels discretos de qualquer série – aqui, do eixo Z)
# -------------------------
def detectar_transientes(labels, idx_inicio, idx_fim, min_run=5):
    """
    Dentro da janela [idx_inicio, idx_fim], detecta múltiplos componentes transientes:

    - Estado de referência = labels[idx_inicio] (classe_inicial).
    - Um transiente é definido como:
        sequência de min_run amostras com classe > classe_inicial (início),
        seguida (em algum ponto) de sequência de min_run amostras com classe == classe_inicial (fim).

    Retorna uma lista de tuplas:
        [(idx_ini_1, idx_fim_1), (idx_ini_2, idx_fim_2), ...]
    Se nenhum transiente for encontrado, retorna lista vazia.
    """
    if idx_inicio is None or idx_fim is None:
        return []

    labels = np.asarray(labels)
    n = len(labels)

    idx_inicio = int(idx_inicio)
    idx_fim = int(idx_fim)

    if idx_fim <= idx_inicio + 2 * min_run:
        return []

    classe_inicial = labels[idx_inicio]
    transientes = []

    i = idx_inicio
    last_possible = min(idx_fim, n - min_run)

    while i <= last_possible:
        # 1) Procurar início de um transiente (classe > classe_inicial por min_run)
        if np.all(labels[i : i + min_run] > classe_inicial):
            idx_ini_trans = i
            # Avança pelo menos min_run
            i = i + min_run

            # 2) Procurar o fim do transiente (classe == classe_inicial por min_run)
            idx_fim_trans = None
            j_last_possible = min(idx_fim, n - min_run)
            for j in range(i, j_last_possible + 1):
                if np.all(labels[j : j + min_run] == classe_inicial):
                    idx_fim_trans = j
                    i = j + min_run  # próxima busca começa depois desse retorno
                    break

            if idx_fim_trans is None:
                # Não encontrou retorno estável ao estado inicial; considerar
                # o fim na borda da janela de movimento
                idx_fim_trans = idx_fim
                i = idx_fim + 1  # força saída do loop

            transientes.append((idx_ini_trans, idx_fim_trans))
        else:
            i += 1

    return transientes

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
    "Número de classes (K-Means – |gyro ML| e |gyro Z|)",
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

        st.subheader("📐 Orientação aproximada do smartphone (via acc)")
        st.write(f"Média do eixo X (acc): {mean_x:.3f}")
        st.write(f"Média do eixo Y (acc): {mean_y:.3f}")
        st.success(
            f"Eixo **vertical** (gravidade) ≈ **{eixo_vertical}** "
            f"(|média| = {abs(g_est):.3f}); eixo **médio-lateral** ≈ **{eixo_ml}**."
        )
        st.caption("Para o giroscópio, o eixo médio-lateral é X ou Y (conforme acima) e o eixo Z será usado como eixo 'vertical' para buscar transientes.")

        # ====== 2) Pré-processamento: interpola em 100 Hz, detrend, filtra ======
        df_acc, fs_acc = preprocess_sensor(df_acc_raw, target_fs=100, cutoff=2)
        df_gyro, fs_gyro = preprocess_sensor(df_gyro_raw, target_fs=100, cutoff=2)

        st.write(f"Fs giroscópio (após interpolação): {fs_gyro:.2f} Hz")

        # ====== 3) Definir ML_gyro e Z_gyro ======
        if eixo_vertical == "X":
            df_gyro["ML_gyro"] = df_gyro["Y_filt"]
        else:
            df_gyro["ML_gyro"] = df_gyro["X_filt"]

        df_gyro["ML_gyro_abs"] = np.abs(df_gyro["ML_gyro"])
        # Eixo vertical (Z) do giroscópio
        df_gyro["Z_gyro"] = df_gyro["Z_filt"]
        df_gyro["Z_gyro_abs"] = np.abs(df_gyro["Z_gyro"])

        # ====== 4) Markov em |ML_gyro| para início/fim do movimento ======
        ml_abs = df_gyro["ML_gyro_abs"].values.reshape(-1, 1)

        kmeans_ml = KMeans(n_clusters=k_classes, n_init=10, random_state=42)
        labels_ml_raw = kmeans_ml.fit_predict(ml_abs)
        centros_ml = kmeans_ml.cluster_centers_.flatten()

        # Ordenar classes (0 = menor |ML| => repouso)
        ordem_ml = np.argsort(centros_ml)
        mapa_ml = {old_label: rank for rank, old_label in enumerate(ordem_ml)}
        labels_ml = np.array([mapa_ml[l] for l in labels_ml_raw])
        df_gyro["Classe_ML"] = labels_ml

        st.write("Centros dos clusters em |ML_gyro| (ordenados):")
        for idx in range(k_classes):
            st.write(f"Classe_ML {idx}: centro ≈ {np.sort(centros_ml)[idx]:.4f}")

        # Início e fim do movimento (baseado em |ML_gyro|)
        idx_inicio = detectar_inicio_movimento(df_gyro["Classe_ML"], base_class=0, min_run=min_run)
        idx_fim = detectar_fim_movimento(df_gyro["Classe_ML"], min_run=min_run)

        tempo_inicio = None
        tempo_fim = None

        if idx_inicio is not None:
            tempo_inicio = df_gyro["Tempo"].iloc[idx_inicio]
            st.success(f"Início de movimento (|ML_gyro|) detectado em ~ *t = {tempo_inicio:.2f} s*.")
        else:
            st.warning("Nenhuma transição estável (Classe_ML 0 → Classe_ML > 0) foi encontrada para o INÍCIO.")

        if idx_fim is not None:
            tempo_fim = df_gyro["Tempo"].iloc[idx_fim]
            st.success(f"Fim de movimento (|ML_gyro|) detectado em ~ *t = {tempo_fim:.2f} s* (usando estado final do registro).")
        else:
            st.warning("Nenhuma transição estável para o estado final foi encontrada para o FIM do movimento em |ML_gyro|.")

        # ====== 5) Markov em |Z_gyro| para DETECTAR TRANSIENTES dentro da janela ======
        transientes_z = []
        if (idx_inicio is not None) and (idx_fim is not None) and (idx_fim > idx_inicio):
            z_abs = df_gyro["Z_gyro_abs"].values.reshape(-1, 1)

            kmeans_z = KMeans(n_clusters=k_classes, n_init=10, random_state=42)
            labels_z_raw = kmeans_z.fit_predict(z_abs)
            centros_z = kmeans_z.cluster_centers_.flatten()

            # Ordenar classes (0 = menor |Z|)
            ordem_z = np.argsort(centros_z)
            mapa_z = {old_label: rank for rank, old_label in enumerate(ordem_z)}
            labels_z = np.array([mapa_z[l] for l in labels_z_raw])
            df_gyro["Classe_Z"] = labels_z

            st.write("Centros dos clusters em |Z_gyro| (ordenados):")
            for idx in range(k_classes):
                st.write(f"Classe_Z {idx}: centro ≈ {np.sort(centros_z)[idx]:.4f}")

            # Detectar transientes da série temporal do giroscópio no eixo Z
            transientes_z = detectar_transientes(df_gyro["Classe_Z"], idx_inicio, idx_fim, min_run=min_run)

            if len(transientes_z) == 0:
                st.info("Nenhum transiente detectado no eixo Z dentro da janela de movimento.")
            else:
                st.info(f"Foram detectados {len(transientes_z)} transiente(s) no eixo Z dentro da janela de movimento.")
                for k, (i_ini, i_fim) in enumerate(transientes_z, start=1):
                    t_ini = df_gyro["Tempo"].iloc[i_ini]
                    t_fim = df_gyro["Tempo"].iloc[i_fim]
                    st.write(
                        f"Transiente Z {k}: início ~ *t = {t_ini:.2f} s*, fim ~ *t = {t_fim:.2f} s* "
                        f"(duração ≈ {(t_fim - t_ini):.2f} s)."
                    )
        else:
            st.info("Janela de movimento não definida (sem início ou fim válidos) – não foi possível buscar transientes em Z.")

        # =========================
        # 6) PLOTS – somente giroscópio (ML e Z)
        # =========================
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        # --- Giroscópio – |ML_gyro| ---
        axes[0].plot(df_gyro["Tempo"], df_gyro["ML_gyro_abs"], label="|Gyro médio-lateral| (|ML_gyro|)")

        if tempo_inicio is not None:
            axes[0].axvline(tempo_inicio, linestyle="--", linewidth=2,
                            label="Início movimento (|ML_gyro|)")
        if tempo_fim is not None:
            axes[0].axvline(tempo_fim, linestyle="--", linewidth=2,
                            label="Fim movimento (|ML_gyro|)")

        if (tempo_inicio is not None) and (tempo_fim is not None) and (tempo_fim > tempo_inicio):
            axes[0].axvspan(tempo_inicio, tempo_fim, alpha=0.15, label="Janela movimento")

        axes[0].set_ylabel("|ω ML| (filtrado)")
        axes[0].set_title("Giroscópio – Valor absoluto do eixo médio-lateral (|ML_gyro|)")
        axes[0].legend()

        # --- Giroscópio – eixo Z (série temporal + transientes) ---
        axes[1].plot(df_gyro["Tempo"], df_gyro["Z_gyro"], label="Gyro eixo Z (Z_gyro)")

        if tempo_inicio is not None and tempo_fim is not None and tempo_fim > tempo_inicio:
            axes[1].axvspan(tempo_inicio, tempo_fim, alpha=0.10, label="Janela movimento")

        # Sombrear transientes em Z
        if len(transientes_z) > 0:
            for k, (i_ini, i_fim) in enumerate(transientes_z, start=1):
                t_ini = df_gyro["Tempo"].iloc[i_ini]
                t_fim = df_gyro["Tempo"].iloc[i_fim]
                axes[1].axvspan(t_ini, t_fim, alpha=0.25,
                                label="Transiente Z" if k == 1 else None)

        axes[1].set_ylabel("ω Z (filtrado)")
        axes[1].set_xlabel("Tempo (s)")
        axes[1].set_title("Giroscópio – Eixo Z com transientes dentro da janela de movimento")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # ====== Tabelas para inspeção ======
        with st.expander("Ver primeiros valores (gyro) – ML e Z"):
            st.dataframe(
                df_gyro[["Tempo", "ML_gyro", "ML_gyro_abs",
                         "Z_gyro", "Z_gyro_abs",
                         "Classe_ML"] + (["Classe_Z"] if "Classe_Z" in df_gyro.columns else [])
                ].head(200)
            )

    except Exception as e:
        st.error(f"Erro ao processar arquivos: {e}")

else:
    st.info("Faça o upload dos dois arquivos para ver o giroscópio nos eixos médio-lateral e Z, com detecção de movimento e transientes.")

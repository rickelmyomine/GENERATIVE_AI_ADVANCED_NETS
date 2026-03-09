import os
import json
import time
import io
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==============================================
# 1. CONFIGURAÇÕES DE UI E ESTADO (SESSION STATE)
# ==============================================
st.set_page_config(page_title='Sentinel VAE Clinical v3.5', layout='wide')

# Custom CSS para cards e semântica
st.markdown("""
    <style>
    .report-card { padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #3e4259; }
    .metric-box { background-color: #1e2130; padding: 15px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state: st.session_state.history = []
if 'feedback_log' not in st.session_state: st.session_state.feedback_log = []
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'current_res' not in st.session_state: st.session_state.current_res = None
if 'gen_imgs' not in st.session_state: st.session_state.gen_imgs = None

# Callbacks de Reset
def reset_analysis_callback():
    st.session_state.analyzed = False
    st.session_state.current_res = None

def reset_lab_callback():
    st.session_state.gen_imgs = None

# ==============================================
# 2. MOTOR DE IA (ARQUITETURA VAE)
# ==============================================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder, self.decoder = encoder, decoder
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)
    def decode(self, z):
        return self.decoder(z)

@st.cache_resource
def load_vae_engine():
    """Carregamento pesado do modelo com arquitetura original restaurada."""
    base = os.path.dirname(__file__)
    weights_p = os.path.join(base, 'models', 'vae_pneumonia.weights.h5')
    latent_dim = 16 

    # Encoder
    enc_in = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(enc_in)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_m = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_v = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_m, z_v])
    encoder = tf.keras.Model(enc_in, [z_m, z_v, z])

    # Decoder
    dec_in = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(dec_in)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    out = tf.keras.layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
    decoder = tf.keras.Model(dec_in, out)

    model = VAE(encoder, decoder)
    model(tf.zeros((1, 28, 28, 1))) # Warmup
    if os.path.exists(weights_p):
        model.load_weights(weights_p)
    return model

@st.cache_data
def preprocess_img(bytes_data):
    """Processamento de dados com cache."""
    img = Image.open(io.BytesIO(bytes_data)).convert('L').resize((28,28))
    return np.array(img).astype('float32') / 255.0

# ==============================================
# 3. SIDEBAR (PAINEL DE CONTROLE)
# ==============================================
with st.sidebar:
    st.header("⚙️ Painel Operacional")
    vae = load_vae_engine()
    st.success("Modelo VAE Ativo")
    
    uploaded_file = st.file_uploader("Upload de Exame", type=['png', 'jpg'], key="sidebar_upload")
    
    col_act, col_rst = st.columns(2)
    with col_act:
        btn_run = st.button("🚀 ANALISAR", type="primary", use_container_width=True)
    with col_rst:
        st.button("🗑️ LIMPAR", on_click=reset_analysis_callback, use_container_width=True)

    st.divider()
    # Alerta de Degradação (Métrica de Confiabilidade Recente)
    if len(st.session_state.history) >= 3:
        avg_recent = np.mean([h['reliability'] for h in st.session_state.history[-3:]])
        if avg_recent < 70:
            st.error("🚨 ALERTA: Degradação de Confiabilidade detectada. Verifique os feedbacks humanos.")

# ==============================================
# 4. ÁREA PRINCIPAL (TABS POR CONTEXTO)
# ==============================================
tab_diag, tab_stats, tab_lab, tab_info = st.tabs([
    "🔍 Diagnóstico", "📊 Performance", "🧪 Laboratório", "📚 Info"
])

# --- TAB 1: TRIAGEM ---
with tab_diag:
    if not st.session_state.analyzed and not btn_run:
        st.info("### 📥 Sistema Sentinel Aguardando\nSuba um Raio-X na barra lateral para iniciar a triagem automatizada.")
    
    if btn_run and uploaded_file:
        with st.status("Executando Pipeline Clínico...", expanded=True) as status:
            st.write("🔧 Normalizando pixels...")
            img_np = preprocess_img(uploaded_file.getvalue())
            batch = np.expand_dims(img_np, (0, -1))
            st.progress(30)
            
            st.write("🧠 Reconstruindo imagem ideal...")
            recon = vae(batch).numpy()
            st.progress(70)
            
            st.write("📊 Calculando métricas de divergência...")
            mse = float(np.mean((batch - recon) ** 2))
            # Confiabilidade: quanto maior o MSE, menor a confiança de que é um pulmão normal
            rel = max(0, min(100, 100 - (mse * 3000))) 
            
            st.session_state.current_res = {
                "id": len(st.session_state.history) + 1,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "mse": mse, "reliability": round(rel, 1),
                "orig": img_np, "recon": recon[0]
            }
            st.session_state.history.append(st.session_state.current_res)
            st.session_state.analyzed = True
            st.progress(100)
            status.update(label="Análise Concluída!", state="complete", expanded=False)

    if st.session_state.analyzed:
        res = st.session_state.current_res
        mse, rel = res['mse'], res['reliability']
        
        # CLASSIFICAÇÃO SEMÂNTICA SOLICITADA
        if mse < 0.01:
            label, color, risk = "NORMAL", "green", "Baixo risco"
            st.success(f"### {label} ({risk})")
            st.write("✅ **Estimativa:** Padrão pulmonar dentro da normalidade estatística.")
        elif mse < 0.02:
            label, color, risk = "BORDERLINE", "orange", "Risco moderado"
            st.warning(f"### {label} ({risk})")
            st.write("⚠️ **Estimativa:** Detectadas divergências leves. Recomenda-se revisão radiológica.")
        else:
            label, color, risk = "POSSÍVEL PNEUMONIA", "red", "Alto risco"
            st.error(f"### {label} ({risk})")
            st.write("❗ **Estimativa:** Divergência crítica detectada. Encaminhar para avaliação urgente.")

        # Layout de Visualização
        c_vis, c_gau = st.columns([2, 1])
        with c_vis:
            v1, v2 = st.columns(2)
            v1.image(res['orig'], caption="Original (28x28)", use_container_width=True)
            v2.image(res['recon'], caption="Reconstrução (Ideal)", use_container_width=True)
        
        with c_gau:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=mse, 
                 title={'text': "Score MSE", 'font': {'size': 16}},
                 gauge={'axis':{'range':[0, 0.04]}, 'bar':{'color': color}}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Confiabilidade da Estimativa", f"{rel}%")

        # Feedback Humano (Persistência)
        st.divider()
        st.write("**O diagnóstico assistido parece correto?**")
        f1, f2, _ = st.columns([1, 1, 4])
        if f1.button("👍 Sim"): 
            st.session_state.feedback_log.append({"id": res['id'], "status": "Correto"})
            st.toast("Feedback positivo salvo!")
        if f2.button("👎 Não"): 
            st.session_state.feedback_log.append({"id": res['id'], "status": "Incorreto"})
            st.toast("Reportado como possível erro de modelo.", icon="🚨")

# --- TAB 2: PERFORMANCE ---
with tab_stats:
    if st.session_state.history:
        st.header("Monitoramento de Estabilidade")
        df = pd.DataFrame(st.session_state.history)
        
        # Gráfico de Evolução
        fig_ev = px.line(df, x='id', y='reliability', title="Confiabilidade por Interação", markers=True)
        fig_ev.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Limite de Segurança")
        st.plotly_chart(fig_ev, use_container_width=True)
        
        

        # DataFrame Operacional com Column Config
        st.subheader("Log de Triagem Assistida")
        st.dataframe(df[['ts', 'mse', 'reliability']], column_config={
            "reliability": st.column_config.ProgressColumn("Confiabilidade (%)", min_value=0, max_value=100, format="%f%%"),
            "mse": "Divergência", "ts": "Horário"
        }, use_container_width=True)
    else:
        st.info("Nenhum dado histórico para análise de performance.")

# --- TAB 3: LABORATÓRIO ---
with tab_lab:
    st.header("🧪 Gerador de Amostras de Controle")
    n_gen = st.slider("Quantidade de imagens", 1, 16, 4)
    
    cg1, cg2 = st.columns([1, 1])
    with cg1:
        if st.button("✨ GERAR IMAGENS", type="primary", use_container_width=True):
            with st.spinner("Decodificando vetores latentes..."):
                z = np.random.normal(0, 1, (n_gen, 16))
                st.session_state.gen_imgs = vae.decode(z).numpy()
    with cg2:
        st.button("🔄 RESETAR LAB", on_click=reset_lab_callback, use_container_width=True)

    if st.session_state.gen_imgs is not None:
        grid = st.columns(4)
        for i, img in enumerate(st.session_state.gen_imgs):
            grid[i % 4].image(img, use_container_width=True, caption=f"Sintética {i+1}")

# --- TAB 4: INFO ---
with tab_info:
    st.header("📚 Documentação Técnica")
    st.markdown("""
    ### Princípio de Funcionamento
    Este sistema utiliza a detecção por **anomalia de reconstrução**. O modelo VAE é treinado apenas com pulmões **normais**. 
    Ao receber uma patologia, o erro de reconstrução (MSE) aumenta proporcionalmente à gravidade da anomalia.
    """)
    
    
    
    st.markdown("""
    ### Tabela de Decisão Semântica
    | MSE | Classificação | Cor | Ação Recomendada |
    | :--- | :--- | :--- | :--- |
    | **< 0.010** | **NORMAL** | Verde | Liberação rápida |
    | **0.010 - 0.020** | **BORDERLINE** | Laranja | Revisão manual |
    | **> 0.020** | **PNEUMONIA** | Vermelho | Avaliação urgente |
    """)
    
    

st.markdown("---")
st.caption("Sentinel VAE Framework 2026 | Arquitetura de Diagnóstico Assistido por IA | Documentação Interna")
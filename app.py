import streamlit as st
import torch
import numpy as np
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Diagnóstico de Doença Cardíaca - TabM",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Carrega o modelo TabM completo"""
    try:
        checkpoint = torch.load('tabm_final_model.pth', map_location='cpu', weights_only=False)
        print("Chaves disponíveis:", checkpoint.keys())
        model = checkpoint['model']
        model.eval()
        return model, checkpoint
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None, None

def predict(model, oldpeak, cp, exang, slope, ca, thal):
    """Faz a predição usando o modelo TabM"""
    # Preparar dados numéricos
    X_num = np.array([[oldpeak]], dtype=np.float32)
    
    # Preparar dados categóricos
    X_cat = np.array([[cp, exang, slope, ca, thal]], dtype=np.int64)
    
    # Converter para tensores
    X_num_tensor = torch.tensor(X_num)
    X_cat_tensor = torch.tensor(X_cat)
    
    # Predição
    with torch.no_grad():
        logits = model(X_num_tensor, X_cat_tensor)
        
        # Verificar dimensão do output
        if logits.dim() > 1:
            # Se for (batch, features) ou (batch, ensemble, features), fazer média
            logits = logits.mean(dim=-1) if logits.dim() == 2 else logits.mean(dim=(1, 2))
        
        # Se ainda tiver mais de 1 elemento, pegar o primeiro
        if logits.numel() > 1:
            logits = logits[0]
        
        prob = torch.sigmoid(logits).item()
    
    return prob

# Título e descrição
st.title("🏥 Diagnóstico de Doença Cardíaca com TabM")
st.markdown("""
Este aplicativo utiliza um modelo **TabM** (Transformer para Dados Tabulares) treinado 
no dataset Cleveland para auxiliar no diagnóstico de doença cardíaca coronariana.

**NÃO** substitui avaliação médica profissional.
""")

# Carregar modelo
model, model_info = load_model()

if model is None:
    st.stop()

# Criar formulário
st.header("📋 Dados do Paciente")

col1, col2, col3 = st.columns(3)

with col1:
    oldpeak = st.number_input(
        "**Oldpeak** (ST depression)",
        min_value=-3.0,
        max_value=7.0,
        value=0.0,
        step=0.1,
        help="Depressão do segmento ST induzida por exercício relativo ao repouso"
    )
    
    cp = st.selectbox(
        "**Tipo de Dor (cp)**",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Assintomática",
            1: "1 - Angina atípica",
            2: "2 - Dor não-anginosa",
            3: "3 - Angina típica"
        }[x],
        help="Tipo de dor torácica reportada"
    )

with col2:
    exang = st.selectbox(
        "**Angina por Exercício (exang)**",
        options=[0, 1],
        format_func=lambda x: "0 - Não" if x == 0 else "1 - Sim",
        help="Angina induzida por exercício"
    )
    
    slope = st.selectbox(
        "**Inclinação ST (slope)**",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Descendente",
            1: "1 - Plana",
            2: "2 - Ascendente"
        }[x],
        help="Inclinação do segmento ST durante exercício"
    )

with col3:
    ca = st.selectbox(
        "**Vasos Principais (ca)**",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} - {x} vaso(s)",
        help="Número de vasos principais coloridos por fluoroscopia (0-3)"
    )
    
    thal = st.selectbox(
        "**Talassemia (thal)**",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Normal",
            1: "1 - Defeito fixo",
            2: "2 - Defeito reversível",
            3: "3 - Outro"
        }[x],
        help="Resultado do teste de talassemia"
    )

# Botão de predição
st.markdown("---")
if st.button("🔍 Realizar Diagnóstico", type="primary", use_container_width=True):
    with st.spinner("Analisando dados..."):
        # Fazer predição
        probability = predict(model, oldpeak, cp, exang, slope, ca, thal)
        
        # Determinar diagnóstico
        threshold = 0.5  # Você pode ajustar baseado no threshold otimizado
        diagnosis = "Doença Cardíaca" if probability >= threshold else "Saudável"
        
        # Mostrar resultados
        st.header("📊 Resultados")
        
        # Criar colunas para resultados
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(
                label="Probabilidade de Doença",
                value=f"{probability*100:.1f}%"
            )
        
        with res_col2:
            st.metric(
                label="Diagnóstico",
                value=diagnosis
            )
        
        with res_col3:
            confidence = abs(probability - 0.5) * 2
            st.metric(
                label="Confiança",
                value=f"{confidence*100:.1f}%"
            )
        
        # Barra de progresso visual
        st.markdown("### Nível de Risco")
        
        # Definir cor baseada na probabilidade
        if probability < 0.3:
            color = "🟢"
            risk_level = "Baixo Risco"
            bar_color = "green"
        elif probability < 0.7:
            color = "🟡"
            risk_level = "Risco Moderado"
            bar_color = "orange"
        else:
            color = "🔴"
            risk_level = "Alto Risco"
            bar_color = "red"
        
        st.progress(probability)
        st.markdown(f"{color} **{risk_level}** - Probabilidade: {probability*100:.1f}%")
        
        # Interpretação
        st.markdown("### 💡 Interpretação")
        if probability >= 0.7:
            st.error("""
            **⚠️ Alta probabilidade de doença cardíaca detectada**
            
            É **fortemente recomendado** buscar avaliação cardiológica imediata.
            Este resultado sugere que há sinais significativos que requerem atenção médica.
            """)
        elif probability >= 0.5:
            st.warning("""
            **⚠️ Probabilidade moderada-alta de doença cardíaca**
            
            Recomenda-se **consulta com cardiologista** para avaliação detalhada.
            Exames complementares podem ser necessários para um diagnóstico preciso.
            """)
        elif probability >= 0.3:
            st.info("""
            **ℹ️ Probabilidade moderada-baixa de doença cardíaca**
            
            Considere avaliação preventiva com médico.
            Mantenha acompanhamento regular e hábitos de vida saudáveis.
            """)
        else:
            st.success("""
            **✅ Baixa probabilidade de doença cardíaca**
            
            Os indicadores sugerem baixo risco, mas mantenha:
            - Check-ups médicos regulares
            - Hábitos de vida saudáveis
            - Atenção a quaisquer novos sintomas
            """)
        
        # Resumo dos dados
        with st.expander("📝 Resumo dos Dados Inseridos"):
            data_summary = pd.DataFrame({
                'Feature': [
                    'Oldpeak (ST depression)',
                    'Tipo de Dor no Peito (cp)',
                    'Angina por Exercício (exang)',
                    'Inclinação ST (slope)',
                    'Vasos Principais (ca)',
                    'Talassemia (thal)'
                ],
                'Valor Inserido': [
                    f"{oldpeak:.1f}",
                    f"{cp}",
                    f"{exang}",
                    f"{slope}",
                    f"{ca}",
                    f"{thal}"
                ],
                'Descrição': [
                    "Depressão ST induzida por exercício",
                    ["Assintomática", "Angina atípica", "Dor não-anginosa", "Angina típica"][cp],
                    "Não" if exang == 0 else "Sim",
                    ["Descendente", "Plana", "Ascendente"][slope],
                    f"{ca} vaso(s) colorido(s)",
                    ["Normal", "Defeito fixo", "Defeito reversível", "Outro"][thal]
                ]
            })
            st.dataframe(data_summary, use_container_width=True, hide_index=True)
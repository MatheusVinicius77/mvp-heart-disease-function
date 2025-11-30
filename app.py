import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --------------------- Configuração da página ---------------------
st.set_page_config(page_title="Predição Doença Cardíaca - Cleveland", layout="centered")
st.title("🫀 Predição de Doença Cardíaca (Dataset Cleveland)")
st.markdown("Preencha todos os campos abaixo com os dados do paciente.")

# --------------------- Carregar o modelo treinado ---------------------
# Salve seu modelo como "heart_model.pkl" na mesma pasta
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------- Função para coletar inputs ---------------------
def user_input_features():
    age      = st.number_input("Idade (anos)", min_value=20, max_value=100, value=50)
    sex      = st.selectbox("Sexo", options=[1, 0], format_func=lambda x: "Masculino" if x==1 else "Feminino")
    cp       = st.selectbox("Tipo de dor torácica (cp)", 
                            options=[1,2,3,4], 
                            format_func=lambda x: {1:"Angina típica", 2:"Angina atípica", 3:"Dor não-angina", 4:"Assintomático"}[x])
    trestbps = st.number_input("Pressão arterial em repouso (mm Hg)", min_value=80, max_value=220, value=120)
    chol     = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=200)
    fbs      = st.selectbox("Glicemia de jejum > 120 mg/dl", options=[1, 0], format_func=lambda x: "Sim" if x==1 else "Não")
    restecg  = st.selectbox("Resultado ECG em repouso", 
                            options=[0,1,2], 
                            format_func=lambda x: {0:"Normal", 1:"Anormalidade onda ST-T", 2:"Hipertrofia ventricular"}[x])
    thalach  = st.number_input("Frequência cardíaca máxima (bpm)", min_value=60, max_value=220, value=150)
    exang    = st.selectbox("Angina induzida por exercício", options=[1, 0], format_func=lambda x: "Sim" if x==1 else "Não")
    oldpeak  = st.slider("Depressão ST induzida por exercício", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope    = st.selectbox("Inclinação do segmento ST no pico do exercício", 
                            options=[1,2,3], 
                            format_func=lambda x: {1:"Ascendente", 2:"Plano", 3:"Descendente"}[x])
    ca       = st.selectbox("Número de vasos principais coloridos por fluoroscopia", options=[0,1,2,3,4], index=0)
    thal     = st.selectbox("Thal", 
                            options=[3,6,7], 
                            format_func=lambda x: {3:"Normal", 6:"Defeito fixo", 7:"Defeito reversível"}[x])

    data = {
        'age'     : age,
        'sex'     : sex,
        'cp'      : cp,
        'trestbps': trestbps,
        'chol'    : chol,
        'fbs'     : fbs,
        'restecg' : restecg,
        'thalach' : thalach,
        'exang'   : exang,
        'oldpeak' : oldpeak,
        'slope'   : slope,
        'ca'      : ca,
        'thal'    : thal
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --------------------- Mostrar o dataframe preenchido ---------------------
st.subheader("Dados inseridos pelo usuário")
st.write(input_df)

# --------------------- Fazer predição ---------------------
if st.button("🔍 Fazer predição"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Resultado da predição")
    
    if prediction[0] == 1:
        st.error("⚠️ O modelo indica **ALTA PROBABILIDADE** de doença cardíaca.")
    else:
        st.success("✅ O modelo indica **BAIXA PROBABILIDADE** de doença cardíaca.")
    
    st.write(f"Probabilidade de doença cardíaca: **{prediction_proba[0][1]:.2%}**")
    st.write(f"Probabilidade de ausência de doença: **{prediction_proba[0][0]:.2%}**")

st.caption("Modelo treinado com o dataset Cleveland processed (UCI). "
           "Salve seu modelo treinado como `heart_model.pkl` na mesma pasta do app.py")
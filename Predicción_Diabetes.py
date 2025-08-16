#Importamos nuestras herramientas para nuestra app web de prediccion de diabetes

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Título
st.title("Predicción de Diabetes")
#Subtítulo
st.markdown("""Esta aplicación predice la probabilidad de que una persona tenga **diabetes** 
en base a ciertos datos médicos.""")

# Cargar los datos modelo
df1 = pd.read_csv("C:\\Users\\USUARIO\\OneDrive\\Documentos\\ciencia de datos\\proyectos de ciencia de datos\\MLregresion\\Predicción de Diabetes\\Data\\diabetes.csv")
df = df1.rename(columns={
    'Pregnancies': 'Embarazos',
    'Glucose': 'Glucosa',
    'BloodPressure': 'Presión',
    'SkinThickness': 'Espesor de Piel',
    'Insulin': 'Insulina',
    'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Historial Familiar',
    'Age': 'Edad'
})

# Entrenar el modelo
X_data = df.drop("Outcome", axis=1)
y_data = df["Outcome"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.25, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo
modelo_logistico = LogisticRegression(max_iter=1000, random_state=42)
modelo_logistico.fit(X_train_scaled, y_train)


# Widgets para la entrada de datos del usuario
st.sidebar.header("Ingresa los datos requeridos")
def user_input_features():
    embarazos = st.sidebar.number_input("Embarazos", min_value=0, max_value=20, value=1)
    glucosa = st.sidebar.number_input("Glucosa", min_value=0, max_value=200, value=90)
    presión = st.sidebar.number_input("Presión Arterial", min_value=0, max_value=150, value=70)
    piel = st.sidebar.number_input("Espesor de Piel", min_value=0, max_value=100, value=20)
    insulina = st.sidebar.number_input("Insulina", min_value=0, max_value=900, value=80)
    imc = st.sidebar.number_input("IMC", min_value=0.0, max_value=70.0, value=25.0)
    historial = st.sidebar.number_input("Historial Familiar", min_value=0.0, max_value=2.5, value=0.5)
    edad = st.sidebar.number_input("Edad", min_value=0, max_value=120, value=30)

    data = {
        "Embarazos": embarazos,
        "Glucosa": glucosa,
        "Presión": presión,
        "Espesor de Piel": piel,
        "Insulina": insulina,
        "IMC": imc,
        "Historial Familiar": historial,
        "Edad": edad
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

#Hacemos la predicción 
if st.sidebar.button("Predecir"):
    input_scaled = scaler.transform(input_df)
    prediccion = modelo_logistico.predict(input_scaled)[0]
    probabilidad = modelo_logistico.predict_proba(input_scaled)[0][1]

    if prediccion == 1:
        st.error(f"Resultado: Positivo para Diabetes (Probabilidad: {probabilidad*100:.2f}%)")
    else:
        st.success(f"Resultado: Negativo para Diabetes (Probabilidad: {probabilidad*100:.2f}%)")

# Mostrar los datos originales
st.subheader("Datos Originales")
st.write(input_df)


# Mostramos los coeficientes del modelo 
st.subheader("Datos del Dataset (Muestra)")
st.write(df)

st.subheader("Coeficientes del Modelo")
coef_df = pd.DataFrame({
    "Variable": X_data.columns,
    "Coeficiente": modelo_logistico.coef_[0],
    "Odds Ratio": np.exp(modelo_logistico.coef_[0])
}).sort_values("Odds Ratio", ascending=False)
st.write(coef_df)


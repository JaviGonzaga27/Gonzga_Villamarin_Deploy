import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential 
from keras.layers import Dense
import tensorflow as tf


def leer_datos(ruta):
    df = pd.read_csv(ruta,sep=',')
    return df


@st.cache_data
def load_and_preprocess_data():
    ingresos = leer_datos("data_evaluacion.csv")
    

    nuevos_nombres = [
        'Age', 'Workclass', 'Fnlwgt', 'Education', 'Education_Num', 
        'Marital_Status', 'Occupation', 'Relationship', 'Race', 
        'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_per_Week', 
        'Native_Country', 'Income'
    ]
    ingresos.columns = nuevos_nombres


    ingresos['Sex'] = ingresos['Sex'].map({'Male': 1, 'Female': 0})
    ingresos['Income'] = ingresos['Income'].map({'<=50K': 0, '>50K': 1})
    ingresos['Marital_Status'] = ingresos['Marital_Status'].replace(['Never-married','Divorced','Married-spouse-absent','Separated','Widowed'],'Soltero')
    ingresos['Marital_Status'] = ingresos['Marital_Status'].replace(['Married-civ-spouse', 'Married-AF-spouse'],'Pareja')
    ingresos['Marital_Status'] = ingresos['Marital_Status'].map({'Soltero': 1, 'Pareja': 0})


    le = LabelEncoder()
    for col in ['Race', 'Relationship', 'Education', 'Workclass', 'Occupation', 'Native_Country']:
        ingresos[col] = le.fit_transform(ingresos[col].astype(str))

    return ingresos


ingresos = load_and_preprocess_data()


X = ingresos.drop(['Income'], axis=1)
y = ingresos['Income']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo
@st.cache_resource
def create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()


st.title('Predicción de Ingresos')

if st.sidebar.button('Entrenar Modelo'):
    with st.spinner('Entrenando el modelo...'):
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    st.sidebar.success('Modelo entrenado!')
    
    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.sidebar.write(f'Precisión del modelo: {accuracy:.2f}')
    
    # Gráfico de pérdida
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

# Formulario para predicción
st.header('Ingrese los datos para la predicción:')
age = st.slider('Edad', 18, 90, 30)
workclass = st.selectbox('Clase de trabajo', options=ingresos['Workclass'].unique())
education = st.selectbox('Educación', options=ingresos['Education'].unique())
marital_status = st.selectbox('Estado civil', options=['Soltero', 'Pareja'])
occupation = st.selectbox('Ocupación', options=ingresos['Occupation'].unique())
relationship = st.selectbox('Relación', options=ingresos['Relationship'].unique())
race = st.selectbox('Raza', options=ingresos['Race'].unique())
sex = st.selectbox('Sexo', options=['Male', 'Female'])
capital_gain = st.number_input('Ganancia de capital', value=0)
capital_loss = st.number_input('Pérdida de capital', value=0)
hours_per_week = st.slider('Horas por semana', 1, 100, 40)
native_country = st.selectbox('País de origen', options=ingresos['Native_Country'].unique())

# Botón para predecir
if st.button('Predecir'):
    # Preparar datos de entrada
    input_data = np.array([[age, workclass, 0, education, 0, 
                            1 if marital_status == 'Soltero' else 0, 
                            occupation, relationship, race, 
                            0 if sex == 'Female' else 1,  # Cambio aquí
                            capital_gain, capital_loss, hours_per_week, native_country]])
    
    # Escalar datos de entrada
    input_scaled = scaler.transform(input_data)
    
    # Realizar predicción
    prediction = model.predict(input_scaled)
    
    # Mostrar resultado
    st.write('Probabilidad de ingresos >50K:', prediction[0][0])
    st.write('Predicción:', '>50K' if prediction[0][0] > 0.5 else '<=50K')
    
    # Mostrar los datos de entrada para verificación
    st.write("Datos de entrada:", input_data)
    st.write("Datos escalados:", input_scaled)
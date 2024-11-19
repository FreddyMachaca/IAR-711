from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from main import modelo_rf  # Importa el modelo entrenado desde main.py

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')  # Página principal con el formulario

# Ruta para predecir el precio
@app.route('/predecir', methods=['POST'])
def predecir():
    # Recibir los datos del formulario
    data = request.form
    nueva_propiedad = pd.DataFrame({
        'Superficie_m2': [float(data['superficie'])],
        'Dormitorios': [int(data['dormitorios'])],
        'Baños': [int(data['banos'])],
        'Garaje': [1 if data['garaje'] == 'Si' else 0],
        'Cercania_Escuelas_km': [float(data['escuelas'])],
        'Cercania_Hospitales_km': [float(data['hospitales'])],
        'Cercania_Teleferico_km': [float(data['teleferico'])],
        'Edad_Propiedad_años': [int(data['edad'])]
    })

    # Realizar la predicción
    precio_predicho = modelo_rf.predecir(nueva_propiedad)[0]
    
    # Retornar la predicción al frontend
    return render_template('index.html', prediccion=f"Precio Predicho: ${precio_predicho:.2f}")

# Ejecutar el servidor Flask
if __name__ == '__main__':
    app.run(debug=True)

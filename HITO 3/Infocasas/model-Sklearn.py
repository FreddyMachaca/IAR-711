import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cargar los datos
df = pd.read_csv('infocasas_dataset.csv')

# Seleccionar características para el modelo
caracteristicas = ['Superficie_Terreno_m2', 'Superficie_Construida_m2', 'Dormitorios', 'Baños', 'Tipo', 'Ubicación']
objetivo = 'Precio_USD'

# Dividir los datos
X = df[caracteristicas]
y = df[objetivo]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear preprocesador
caracteristicas_numericas = ['Superficie_Terreno_m2', 'Superficie_Construida_m2', 'Dormitorios', 'Baños']
caracteristicas_categoricas = ['Tipo', 'Ubicación']

preprocesador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), caracteristicas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), caracteristicas_categoricas)
    ])

# Imprimir formas para verificar
print("Forma del conjunto de entrenamiento:", X_train.shape)
print("Forma del conjunto de prueba:", X_test.shape)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Crear el pipeline del modelo
modelo = Pipeline([
    ('preprocesador', preprocesador),
    ('regresor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# Entrenar el modelo
modelo.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio: {mse}")
print(f"Puntuación R-cuadrado: {r2}")


from sklearn.model_selection import GridSearchCV

# Definir la cuadrícula de parámetros
param_grid = {
    'regresor__n_estimators': [50, 100, 200],
    'regresor__max_depth': [None, 10, 20, 30],
    'regresor__min_samples_split': [2, 5, 10]
}

# Crear el objeto de búsqueda en cuadrícula
grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Ajustar la búsqueda en cuadrícula
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros y puntuación
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor puntuación de validación cruzada:", -grid_search.best_score_)

# Usar el mejor modelo para la evaluación final
mejor_modelo = grid_search.best_estimator_
y_pred_mejor = mejor_modelo.predict(X_test)
mse_mejor = mean_squared_error(y_test, y_pred_mejor)
r2_mejor = r2_score(y_test, y_pred_mejor)

print(f"Mejor Modelo - Error Cuadrático Medio: {mse_mejor}")
print(f"Mejor Modelo - Puntuación R-cuadrado: {r2_mejor}")


def predecir_precio(superficie_terreno, superficie_construida, dormitorios, banos, tipo, ubicacion):
    # Crear un DataFrame con la entrada
    datos_entrada = pd.DataFrame({
        'Superficie_Terreno_m2': [superficie_terreno],
        'Superficie_Construida_m2': [superficie_construida],
        'Dormitorios': [dormitorios],
        'Baños': [banos],
        'Tipo': [tipo],
        'Ubicación': [ubicacion]
    })
    
    # Hacer predicción
    prediccion_usd = mejor_modelo.predict(datos_entrada)[0]
    
    # Convertir a Bolivianos (asumiendo 1 USD = 6.96 BOB a partir de 2023)
    prediccion_bob = prediccion_usd * 6.96
    
    return prediccion_usd, prediccion_bob

# Ejemplo de uso:
precio_usd, precio_bob = predecir_precio(110, 60, 2, 1, 'Casa', 'La Paz, Sopocachi')
print(f"Precio predicho: ${precio_usd:.2f} USD / {precio_bob:.2f} BOB")
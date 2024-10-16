import pandas as pd
import seaborn as sns
import random
import numpy as np
import matplotlib.pyplot as plt

# Configurar semilla aleatoria para reproducibilidad
random.seed(42)
np.random.seed(42)

# 1. Preparación de datos
# Cargar el conjunto de datos
datos = pd.read_csv('infocasas_dataset.csv')

# Seleccionar características relevantes
caracteristicas_relevantes = ['Superficie_Terreno_m2', 'Superficie_Construida_m2', 
                              'Dormitorios', 'Baños', 'Cocina', 'Garage', 'Galpones']
objetivo = 'Precio_USD'

# Crear un nuevo DataFrame con características relevantes y objetivo
df = datos[caracteristicas_relevantes + [objetivo]].copy()

# Manejar variables categóricas
df['Cocina'] = df['Cocina'].map({'Sí': 1, 'No': 0})
df['Garage'] = df['Garage'].map({'Sí': 1, 'No': 0})
df['Galpones'] = df['Galpones'].map({'Sí': 1, 'No': 0})

# Dividir los datos en características (X) y objetivo (y)
X = df.drop(objetivo, axis=1)
y = df[objetivo]

# Función para dividir los datos en conjuntos de entrenamiento y prueba
def train_test_split(X, y, test_size=0.2):
    n = len(X)
    test_n = int(n * test_size)
    indices = list(range(n))
    random.shuffle(indices)
    test_indices = indices[:test_n]
    train_indices = indices[test_n:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Ingeniería de modelos
class ArbolDecision:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
    
    def dividir_nodo(self, X, y, profundidad):
        m = X.shape[1]
        if len(y) <= 3 or profundidad >= self.max_depth:
            return np.mean(y)
        
        mejor_ganancia = 0
        mejor_pregunta = None
        mejor_izquierda = None
        mejor_derecha = None
        
        for col in range(m):
            valores_unicos = np.unique(X[:, col])
            for valor in valores_unicos:
                pregunta = (col, valor)
                izquierda, derecha = self.partir(X, y, pregunta)
                if len(izquierda[1]) > 0 and len(derecha[1]) > 0:
                    ganancia = self.ganancia_informacion(y, izquierda[1], derecha[1])
                    if ganancia > mejor_ganancia:
                        mejor_ganancia = ganancia
                        mejor_pregunta = pregunta
                        mejor_izquierda = izquierda
                        mejor_derecha = derecha
        
        if mejor_ganancia > 0:
            izquierda = self.dividir_nodo(mejor_izquierda[0], mejor_izquierda[1], profundidad + 1)
            derecha = self.dividir_nodo(mejor_derecha[0], mejor_derecha[1], profundidad + 1)
            return (mejor_pregunta, izquierda, derecha)
        
        return np.mean(y)
    
    def partir(self, X, y, pregunta):
        col, valor = pregunta
        mascara = X[:, col] >= valor
        return (X[~mascara], y[~mascara]), (X[mascara], y[mascara])
    
    def ganancia_informacion(self, padre, izquierda, derecha):
        p = len(izquierda) / len(padre)
        return self.varianza(padre) - p * self.varianza(izquierda) - (1 - p) * self.varianza(derecha)
    
    def varianza(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)
    
    def ajustar(self, X, y):
        self.raiz = self.dividir_nodo(X.values, y.values, 0)
    
    def predecir_uno(self, x, nodo):
        if isinstance(nodo, tuple):
            pregunta, izquierda, derecha = nodo
            if x[pregunta[0]] >= pregunta[1]:
                return self.predecir_uno(x, derecha)
            else:
                return self.predecir_uno(x, izquierda)
        else:
            return nodo
    
    def predecir(self, X):
        return [self.predecir_uno(x, self.raiz) for x in X.values]

class BosqueAleatorio:
    def __init__(self, n_arboles=10, max_depth=5):
        self.n_arboles = n_arboles
        self.max_depth = max_depth
        self.arboles = []
    
    def ajustar(self, X, y):
        for _ in range(self.n_arboles):
            arbol = ArbolDecision(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_muestra = X.iloc[indices]
            y_muestra = y.iloc[indices]
            arbol.ajustar(X_muestra, y_muestra)
            self.arboles.append(arbol)
    
    def predecir(self, X):
        predicciones = np.array([arbol.predecir(X) for arbol in self.arboles])
        return np.mean(predicciones, axis=0)

# 3. Evaluación del modelo
modelo_rf = BosqueAleatorio(n_arboles=10, max_depth=5)
modelo_rf.ajustar(X_train, y_train)

y_pred = modelo_rf.predecir(X_test)

# Calcular métricas de evaluación
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f"Error Cuadrático Medio: {mse}")
print(f"Raíz del Error Cuadrático Medio: {rmse}")
print(f"Puntuación R-cuadrado: {r2}")

# 4. Implementación del modelo
def predecir_precio(nuevos_datos):
    return modelo_rf.predecir(nuevos_datos)[0]

# Ejemplo de uso
nueva_propiedad = pd.DataFrame({
    'Superficie_Terreno_m2': [200],
    'Superficie_Construida_m2': [110],
    'Dormitorios': [5],
    'Baños': [1],
    'Cocina': [1],
    'Garage': [1],
    'Galpones': [0]
})

precio_predicho = predecir_precio(nueva_propiedad)
print(f"Precio Predicho: ${precio_predicho:.2f}")


"""
# Visualización de la importancia de las características
importancia_caracteristicas = pd.DataFrame({
    'caracteristica': X.columns,
    'importancia': np.random.rand(len(X.columns))  # Simulamos la importancia aleatoriamente
})
importancia_caracteristicas = importancia_caracteristicas.sort_values('importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importancia', y='caracteristica', data=importancia_caracteristicas)
plt.title('Importancia de las Características')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.show()
"""

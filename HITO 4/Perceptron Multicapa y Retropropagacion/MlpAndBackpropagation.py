import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Inicialización de pesos y sesgos
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Propagación hacia adelante
        self.layer1 = X
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.layer2, self.weights2) + self.bias2)
        return self.output
    
    def backward(self, X, y, output):
        # Retropropagación
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.layer2_error = np.dot(self.output_delta, self.weights2.T)
        self.layer2_delta = self.layer2_error * self.sigmoid_derivative(self.layer2)
        
        # Actualización de pesos y sesgos
        self.weights2 += self.learning_rate * np.dot(self.layer2.T, self.output_delta)
        self.bias2 += self.learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        self.weights1 += self.learning_rate * np.dot(X.T, self.layer2_delta)
        self.bias1 += self.learning_rate * np.sum(self.layer2_delta, axis=0, keepdims=True)
    
    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)

# Normalización de datos
def normalize_data(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

# Preparación de datos
# [asistencia%, nota_parcial1, nota_parcial2, nota_trabajos]
X = np.array([
    [95, 80, 75, 90],  # Estudiante 1 - Alta asistencia
    [65, 60, 65, 70],  # Estudiante 2 - Baja asistencia
    [90, 85, 80, 95],  # Estudiante 3 - Alta asistencia
    [60, 50, 55, 60],  # Estudiante 4 - Baja asistencia
    [100, 90, 85, 100], # Estudiante 5 - Asistencia perfecta
    [85, 70, 75, 80],  # Estudiante 6 - Buena asistencia
    [55, 40, 45, 50],  # Estudiante 7 - Muy baja asistencia
    [95, 95, 90, 85],  # Estudiante 8 - Alta asistencia
])

# Etiquetas (0: no aprobado, 1: aprobado)
y = np.array([[1], [0], [1], [0], [1], [1], [0], [1]])

# Normalizar los datos
X_normalized = normalize_data(X)

# División de datos en entrenamiento y prueba
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X_normalized[train_idx], X_normalized[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Crear y entrenar el modelo
input_size = X.shape[1]  # 4 características
hidden_size = 5
output_size = 1

mlp = MLP(input_size, hidden_size, output_size, learning_rate=0.1)
mlp.train(X_train, y_train, epochs=1000)

# Evaluar el modelo
y_pred = mlp.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Precisión del modelo: {accuracy:.2f}')

# Ejemplo de predicción con nuevos datos
nuevos_datos = np.array([
    [90, 85, 80, 90],  # Estudiante con buena asistencia y buenas notas
    [60, 55, 60, 65]   # Estudiante con baja asistencia y notas regulares
])
nuevos_datos_normalized = (nuevos_datos - X.mean(axis=0)) / X.std(axis=0)
predicciones = mlp.predict(nuevos_datos_normalized)
print('Predicciones para nuevos datos:', predicciones.flatten())
print('Probabilidades:', mlp.forward(nuevos_datos_normalized).flatten())
print('Pesos de la capa oculta:', mlp.weights1)
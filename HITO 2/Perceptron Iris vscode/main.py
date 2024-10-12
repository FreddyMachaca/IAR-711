import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

def generar_pesos_aleatorios(n):
    pesos = [random.random() * 10 - 5 for _ in range(n)]
    print("Pesos iniciales:{}".format(pesos))
    return pesos

def ajustar_pesos(clase_real, datum):
    pesos[0] = pesos[0] + clase_real
    for i in range(1, 5):
        pesos[i] = pesos[i] + clase_real * datum[i-1]

def predecir(datum):
    pesos_sin_bias = pesos[1:5]
    pesos_bias = pesos[0]
    valores_atributo = datum[:4]
    activation = sum([i*j for i,j in zip(pesos_sin_bias, valores_atributo)]) + pesos_bias
    return 1 if activation > 0 else -1

def entrenar(data, epochs):
    for epoch in range(epochs):
        for _, row in data.iterrows():
            clase_real = dict_classes[row['Species']]
            clase_a_predecir = predecir(row.iloc[:4].tolist())
            if clase_real != clase_a_predecir:
                ajustar_pesos(clase_real, row.iloc[:4].tolist())
        print(f"Pesos finales para el epoch {epoch}: {pesos}")

def verificar(data):
    contador = sum(1 for _, row in data.iterrows() if dict_classes[row['Species']] != predecir(row.iloc[:4].tolist()))
    return (1 - contador / len(data)) * 100

# Load the dataset (adjust the path as needed)
dataset = pd.read_csv("Iris.csv")

# Display the dataset
print(dataset)

# Create and display pairplot
sns.pairplot(dataset, hue="Species")
plt.show()

# Create and display violinplot
sns.violinplot(data=dataset, x="Species", y="SepalLengthCm")
plt.show()

# Display dataset info
dataset.info()

# Split the dataset by species
data_iris_setosa = dataset[0:50]
data_iris_versicolor = dataset[50:100]
data_iris_virginica = dataset[100:150]

print(data_iris_setosa)
print(data_iris_versicolor)
print(data_iris_virginica)

# Define classes
classes = ['Iris-setosa', 'Iris-versicolor']
print(classes)

dict_classes = {classes[0]: 1, classes[1]: -1}
print(dict_classes)

# Combine setosa and versicolor data
data_white_two_species = pd.concat([data_iris_setosa, data_iris_versicolor])
print(data_white_two_species)

# Randomly shuffle the data
data_white_two_species = data_white_two_species.sample(frac=1).reset_index(drop=True)
print(data_white_two_species)

# Generate initial weights
pesos = generar_pesos_aleatorios(5)

# Split data into training and verification sets
data_training = data_white_two_species[0:80]
data_verification = data_white_two_species[80:100]

# Train the model
entrenar(data_training, epochs=10)

# Verify the model
accuracy = verificar(data_verification)
print("Error calculando el modelo de clasificacion: {}%".format(100-accuracy))
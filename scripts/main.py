import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("./data/tic_tac_toe.csv")
dataset.replace(["x","o", "b", "positivo", "negativo"], [1,-1, 0, 1, -1], inplace = True)

tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()

x = dataset.iloc[:,0:9]
y = dataset.iloc[:,9]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

tree.fit(x_train, y_train)
knn.fit(x_train, y_train)

tree_prediction = tree.predict(x_test)
knn_prediction = knn.predict(x_test)

tree_score = accuracy_score(y_test, tree_prediction)
knn_score = accuracy_score(y_test, knn_prediction)

input_vector = []
for i in range (9):
    valor = input()
    valor = 1 if valor == "x" else -1 if valor == "o" else 0
    input_vector.append(valor)

input_vector = np.array(input_vector).reshape(1, -1)

if tree_score > knn_score:
    resultado = tree.predict(input_vector)
    model = "Árvore de decisão"
    accuracy = tree_score
else:
    resultado = knn.predict(input_vector)
    model = "KNN"
    accuracy = knn_score

resultado = "x venceu" if resultado[0] == 1 else "x não venceu"

print(f"Modelo selecionado: {model}")
print(f"Acurácia do modelo {accuracy}")
print(f"Resultado {resultado}")

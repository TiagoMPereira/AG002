# Bibliotecas para manipulação de dados
import numpy as np
import pandas as pd

# Criação e avaliação de modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

# Função auxiliar para imprimir tabuleiro
from print_board import print_board

# Lendo dataset e alterando valores
dataset = pd.read_csv("./data/tic_tac_toe.csv")
dataset.replace(["x", "o", "b", "positivo", "negativo"], [1, -1, 0, 1, -1], inplace = True)

# Separando variáveis X e Y
x = dataset.iloc[:,0:9]
y = dataset.iloc[:,9]

# Dividindo conjunto de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Criando modelos de classificação
tree = DecisionTreeClassifier()
rf = RandomForestClassifier()
svc = SVC()

# Treinando modelos
tree.fit(x_train, y_train)
rf.fit(x_train, y_train)
svc.fit(x_train, y_train)

# Fazendo predições com os modelos
tree_prediction = tree.predict(x_test)
rf_prediction = rf.predict(x_test)
svc_prediction = svc.predict(x_test)

# Avaliando acurácia dos modelos
tree_score = accuracy_score(y_test, tree_prediction)
rf_score = accuracy_score(y_test, rf_prediction)
svc_score = accuracy_score(y_test, svc_prediction)

# Pedindo entrada do usuário
print("Informe sua entrada conforme a sequência descrita: (cada valor deve ser separado por ENTER)")
print_board()
input_vector = []
input_raw = []
for i in range (9):
    valor = input()
    input_raw.append(valor)
    valor = 1 if valor == "x" else -1 if valor == "o" else 0
    input_vector.append(valor)

input_vector = np.array(input_vector).reshape(1, -1)

print_board(input_raw)

# Avaliando a qual modelo submeter os dados do usuário e fazendo a predição dos mesmos
if tree_score > rf_score and tree_score > svc_score:
    resultado = tree.predict(input_vector)
    model = "Árvore de decisão"
    accuracy = tree_score

elif rf_score > tree_score and rf_score > svc_score:
    resultado = rf.predict(input_vector)
    model = "Random Forest"
    accuracy = rf_score

else:
    resultado = svc.predict(input_vector)
    model = "SVC"
    accuracy = svc_score

resultado = "x venceu" if resultado[0] == 1 else "x não venceu"

# Apresentando resultados
print(f"Modelo selecionado: {model}")
print(f"Acurácia do modelo: {accuracy}")
print(f"Resultado: {resultado}")
# =============================
# RANDOM FOREST CON DATOS REALES
# =============================

# Paso 1: Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

# Paso 2: Cargar datos
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Mostrar los primeros datos
print("Primeras filas del dataset:")
print(X.head())
print("\nEtiquetas:", iris.target_names)

# Paso 3: Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 4: Crear y entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Paso 5: Hacer predicciones
y_pred = modelo.predict(X_test)

# Paso 6: Evaluar el modelo
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nPrecisión del modelo:", accuracy_score(y_test, y_pred))

# Paso 7: Importancia de las variables
importancias = modelo.feature_importances_
indices = np.argsort(importancias)[::-1]

# Mostrar importancias
print("\nImportancia de cada variable:")
for i in range(X.shape[1]):
    print(f"{X.columns[indices[i]]}: {importancias[indices[i]]:.4f}")

# Función: Matriz de Confusión Visual
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusión")
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()

# Función: Importancia de Variables
def plot_feature_importances(modelo, feature_names):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importancia de Variables")
    plt.bar(range(len(importancias)), importancias[indices], align="center")
    plt.xticks(range(len(importancias)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel("Variables")
    plt.ylabel("Importancia")
    plt.tight_layout()
    plt.show()

# Función: Error vs Número de Árboles
def plot_error_vs_trees(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    errores = []
    arboles = list(range(1, 101))  # De 1 a 100 árboles

    for n in arboles:
        modelo = RandomForestClassifier(n_estimators=n, random_state=42)
        modelo.fit(X_train, y_train)
        error = 1 - modelo.score(X_test, y_test)
        errores.append(error)

    plt.figure(figsize=(8, 6))
    plt.plot(arboles, errores, marker='o')
    plt.title('Error de Test vs Número de Árboles')
    plt.xlabel('Número de Árboles')
    plt.ylabel('Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Función: Visualizar un Árbol Individual
def plot_single_tree(modelo, feature_names, tree_index=0):
    plt.figure(figsize=(20, 10))
    plot_tree(modelo.estimators_[tree_index], 
              feature_names=feature_names, 
              class_names=iris.target_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title(f"Árbol {tree_index} del Random Forest")
    plt.show()

# Paso 8: Gráficos
plot_confusion_matrix(y_test, y_pred, labels=iris.target_names)
plot_feature_importances(modelo, X.columns)
plot_error_vs_trees(X, y)
plot_single_tree(modelo, X.columns, tree_index=0)

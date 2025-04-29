# ================================
# RANDOM FOREST PARA REGRESIÓN
# ================================

# Paso 1: Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paso 2: Crear datos artificiales
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Tamaño del terreno (0 a 10)
y = 2.5 * X.squeeze() + np.random.randn(100) * 2  # Precio de casa + algo de ruido

# Paso 3: Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Crear y entrenar el modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Paso 5: Hacer predicciones
y_pred = modelo.predict(X_test)

# Paso 6: Evaluar el modelo
print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred))
print("Error absoluto medio (MAE):", mean_absolute_error(y_test, y_pred))
print("R² (coeficiente de determinación):", r2_score(y_test, y_pred))

# Paso 7: Graficar resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.scatter(X_test, y_pred, color='red', label='Predicciones', marker='x')
plt.title('Random Forest - Regresión')
plt.xlabel('Tamaño del terreno')
plt.ylabel('Precio de la casa')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

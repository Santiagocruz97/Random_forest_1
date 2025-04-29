from sklearn.ensemble import RandomForestClassifier  # Para clasificación
# o
from sklearn.ensemble import RandomForestRegressor   # Para regresión
import pandas as pd

# Ejemplo de datos
X = pd.DataFrame({
    'humedad': [30, 45, 60, 80],
    'temperatura': [22, 20, 18, 16],
    'viento': [5, 7, 10, 15]
})

y = ['No llueve', 'No llueve', 'Llueve', 'Llueve']  # etiquetas de clasificación

# Crear el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
modelo.fit(X, y)

# Nuevos datos para predecir
X_nuevos = pd.DataFrame({
    'humedad': [35],
    'temperatura': [9],
    'viento': [8]
})

# Predicción
prediccion = modelo.predict(X_nuevos)
print(prediccion)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar con el conjunto de entrenamiento
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Probar con el conjunto de prueba
y_pred = modelo.predict(X_test)

# Ver precisión
print("Precisión del modelo:", accuracy_score(y_test, y_pred))

















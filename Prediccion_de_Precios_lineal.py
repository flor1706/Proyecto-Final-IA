# Paso 1: Importar bibliotecas
import numpy as np                         # Importamos la librería NumPy para cálculos numéricos.
import pandas as pd                        # Importamos Pandas para manipulación de datos.
import matplotlib.pyplot as plt             # Importamos Matplotlib para crear gráficos.
from sklearn.preprocessing import MinMaxScaler  # Importamos MinMaxScaler para escalar los datos.
from sklearn.linear_model import LinearRegression  # Importamos el modelo de regresión lineal.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Importamos métricas.

# Paso 2: Cargar y preprocesar los datos
data = pd.read_csv('TSLA.csv')          # Cargamos los datos desde un archivo CSV.
prices = data['Close'].values.reshape(-1, 1)  # Seleccionamos los precios de cierre y los transformamos.

scaler = MinMaxScaler(feature_range=(0, 1))  # Creamos un escalador para normalizar los datos.
prices_scaled = scaler.fit_transform(prices)  # Escalamos los precios entre 0 y 1.

train_size = int(len(prices) * 0.8)     # Definimos el tamaño del conjunto de entrenamiento.
train_data = prices_scaled[:train_size]  # Datos de entrenamiento.
test_data = prices_scaled[train_size:]   # Datos de prueba.

X_train, y_train = train_data[:-1], train_data[1:]  # Creamos conjuntos de entrada y salida para entrenamiento.
X_test, y_test = test_data[:-1], test_data[1:]     # Conjuntos de entrada y salida para prueba.

# Paso 3: Construir y entrenar el modelo de regresión lineal
model = LinearRegression()    # Creamos un modelo de regresión lineal.
model.fit(X_train, y_train)    # Entrenamos el modelo con los datos de entrenamiento.

# Paso 4: Hacer predicciones
y_pred = model.predict(X_test)  # Realizamos predicciones en los datos de prueba.

# Desescalar las predicciones para obtener los valores reales
y_pred = scaler.inverse_transform(y_pred)  # Volvemos a la escala original.
y_test = scaler.inverse_transform(y_test)

# Paso 5: Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)    # Calculamos el error cuadrático medio.
mae = mean_absolute_error(y_test, y_pred)   # Calculamos el error absoluto medio.
r2 = r2_score(y_test, y_pred)              # Calculamos el coeficiente de determinación.

print(f'MSE: {mse}')    # Imprimimos el error cuadrático medio.
print(f'MAE: {mae}')    # Imprimimos el error absoluto medio.
print(f'R-squared: {r2}')  # Imprimimos el coeficiente de determinación.

# Paso 6: Visualizar las predicciones
plt.figure(figsize=(12, 6))  # Creamos una figura para el gráfico.
plt.plot(y_test, label='Datos reales')    # Mostramos los datos reales en el gráfico.
plt.plot(y_pred, label='Predicciones', color='red')  # Mostramos las predicciones en rojo.
plt.legend()             # Mostramos una leyenda en el gráfico.
plt.title('Predicciones de Precios de Acciones (Regresión Lineal)')  # Añadimos un título al gráfico.
plt.xlabel('Días')        # Etiqueta para el eje x.
plt.ylabel('Precio de Cierre')  # Etiqueta para el eje y.
plt.show()                # Mostramos el gráfico.


# Paso 1: Importar bibliotecas
import tensorflow as tf                 # Importamos TensorFlow, una librería de aprendizaje automático.
from tensorflow import keras             # Importamos la parte de Keras de TensorFlow para construir modelos.
import numpy as np                       # Importamos NumPy, una librería para cálculos numéricos eficientes.
import pandas as pd                      # Importamos Pandas, una librería para manipulación de datos.
import matplotlib.pyplot as plt          # Importamos Matplotlib para visualización de datos.
from sklearn.preprocessing import MinMaxScaler  # Importamos MinMaxScaler para escalar los datos.
from tensorflow.keras.models import Sequential   # Importamos el modelo secuencial de Keras.
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Importamos las capas LSTM, Densa y Dropout.

# Paso 2: Cargar y preprocesar los datos
data = pd.read_csv('TSLA.csv')            # Cargamos los datos del archivo CSV.
prices = data['Close'].values            # Seleccionamos la columna 'Close' como los precios.

scaler = MinMaxScaler(feature_range=(0, 1))   # Creamos un escalador para normalizar los precios entre 0 y 1.
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

train_size = int(len(prices) * 0.8)     # Definimos el tamaño del conjunto de entrenamiento.
train_data = prices_scaled[:train_size]  # Datos de entrenamiento.
test_data = prices_scaled[train_size:]   # Datos de prueba.

# Paso 3: Crear secuencias de datos para el modelo LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])   # Creamos secuencias de datos.
        y.append(data[i+sequence_length])     # La etiqueta es el siguiente valor después de la secuencia.
    return np.array(X), np.array(y)

sequence_length = 7   # Definimos la longitud de las secuencias.
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Paso 4: Construir y entrenar el modelo LSTM
input_shape = (X_train.shape[1], X_train.shape[2])  # Definimos la forma de entrada para el modelo.

model = Sequential()                    # Creamos un modelo secuencial.
model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))   # Agregamos una capa LSTM.
model.add(Dropout(0.2))                 # Agregamos una capa de Dropout para regularización.
model.add(LSTM(units=50))               # Agregamos otra capa LSTM.
model.add(Dense(1))                     # Agregamos una capa densa con una sola neurona.

model.compile(optimizer='adam', loss='mean_squared_error')   # Compilamos el modelo.

model.fit(X_train, y_train, epochs=50, batch_size=64)   # Entrenamos el modelo.

# Paso 5: Hacer predicciones y evaluar el modelo
y_pred = model.predict(X_test)   # Realizamos predicciones en los datos de prueba.
y_pred = scaler.inverse_transform(y_pred)  # Desescalamos las predicciones.
y_test = scaler.inverse_transform(y_test)    # Desescalamos los datos de prueba.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   # Importamos métricas.

mse = mean_squared_error(y_test, y_pred)   # Calculamos el error cuadrático medio.
mae = mean_absolute_error(y_test, y_pred)  # Calculamos el error absoluto medio.
r2 = r2_score(y_test, y_pred)             # Calculamos el coeficiente de determinación.

print(f'MSE: {mse}')   # Imprimimos el error cuadrático medio.
print(f'MAE: {mae}')   # Imprimimos el error absoluto medio.
print(f'R-squared: {r2}')   # Imprimimos el coeficiente de determinación.

# Paso 6: Visualizar las predicciones
plt.figure(figsize=(12, 6))               # Creamos una figura para mostrar el gráfico.
plt.plot(y_test, label='Datos reales')     # Graficamos los datos reales.
plt.plot(y_pred, label='Predicciones', color='red')  # Graficamos las predicciones en rojo.
plt.legend()  # Mostramos la leyenda en el gráfico.
plt.title('Predicciones de Precios de Acciones')   # Añadimos un título al gráfico.
plt.xlabel('Días')                       # Etiqueta del eje x.
plt.ylabel('Precio de Cierre')           # Etiqueta del eje y.
plt.show()                               # Mostramos el gráfico.


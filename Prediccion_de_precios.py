# Paso 1: Importar bibliotecas
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = keras.models.Sequential()
model.add(keras.layers.LSTM(50))
model.add(keras.layers.Dense(1))

# Paso 2: Cargar y preprocesar los datos

# Cargar los datos desde un archivo CSV
data = pd.read_csv('TSLA.csv')

# Seleccionar la variable objetivo (el precio de cierre)
prices = data['Close'].values

# Escalar los precios a un rango entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(prices) * 0.8)
train_data = prices_scaled[:train_size]
test_data = prices_scaled[train_size:]
print(train_data)
# Paso 3: Crear secuencias de datos para el modelo LSTM

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 12  # Longitud de la secuencia
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)
print("Forma de X_train:", X_train.shape)

# Paso 4: Construir y entrenar el modelo LSTM
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape= input_shape))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=64)


# Paso 5: Hacer predicciones y evaluar el modelo

y_pred = model.predict(X_test)

# Desescalar las predicciones para obtener los valores reales
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Evaluar el modelo (por ejemplo, calcular el error cuadrático medio)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Paso 6: Visualizar las predicciones

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Datos reales')
plt.plot(y_pred, label='Predicciones', color='red')
plt.legend()
plt.title('Predicciones de Precios de Acciones')
plt.xlabel('Días')
plt.ylabel('Precio de Cierre')
plt.show()


# Paso 1: Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paso 2: Cargar y preprocesar los datos
data = pd.read_csv('TSLA.csv')
prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

train_size = int(len(prices) * 0.8)
train_data = prices_scaled[:train_size]
test_data = prices_scaled[train_size:]

X_train, y_train = train_data[:-1], train_data[1:]
X_test, y_test = test_data[:-1], test_data[1:]

# Paso 3: Construir y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Paso 4: Hacer predicciones
y_pred = model.predict(X_test)

# Desescalar las predicciones para obtener los valores reales
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Paso 5: Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R-squared: {r2}')

# Paso 6: Visualizar las predicciones
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Datos reales')
plt.plot(y_pred, label='Predicciones', color='red')
plt.legend()
plt.title('Predicciones de Precios de Acciones (Regresión Lineal)')
plt.xlabel('Días')
plt.ylabel('Precio de Cierre')
plt.show()

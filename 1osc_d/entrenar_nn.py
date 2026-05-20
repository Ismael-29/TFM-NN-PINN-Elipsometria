"""
Entrenamiento de la red neuronal para el caso de 1 oscilador Tauc-Lorentz
con espesor variable.

Entrada:  202 valores espectrales (101 tan(Ψ) + 101 cos(Δ))
Salida:   6 parámetros (A, E₀, Eg, C, ε∞, d)

Arquitectura: 256 → 256 → 128 → 128 → 64 → 32 → 6 (todas ReLU, salida lineal)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Cargar dataset ---
df = pd.read_csv('1osc_d/dataset_1osc_d.csv')

col_params = ['A', 'E_0', 'E_g', 'C', 'eps_inf', 'd']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

# --- Preparar datos ---
# Entrada: espectros (tan(Ψ), cos(Δ)), Salida: parámetros TL + espesor
X = df[col_cos + col_tan].values    # (40000, 202)
Y = df[col_params].values           # (40000, 6)

# Normalización: StandardScaler para entrada y salida
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# División 75% entrenamiento, 25% test
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.25, random_state=42)

# --- Definir modelo ---
# Arquitectura wider: capa adicional de 256 respecto al caso sin espesor,
# para compensar la complejidad añadida por el parámetro d
model = keras.Sequential([
    layers.Input(shape=(202,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(6)     # salida: A, E₀, Eg, C, ε∞, d
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# --- Entrenar ---
# EarlyStopping: detiene si val_loss no mejora en 20 épocas
early_stopper = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stopper]
)

# --- Evaluar sobre test ---
Y_pred_norm = model.predict(X_test)
Y_pred = scaler_y.inverse_transform(Y_pred_norm)
Y_real = scaler_y.inverse_transform(Y_test)

print("\nResultados sobre test:")
for i, param in enumerate(col_params):
    mae = mean_absolute_error(Y_real[:, i], Y_pred[:, i])
    mse = mean_squared_error(Y_real[:, i], Y_pred[:, i])
    print(f'  {param}: MAE = {mae:.4f}, MSE = {mse:.4f}')

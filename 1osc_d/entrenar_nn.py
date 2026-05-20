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

from modelo_tl import eps_im, eps_re, indice_refraccion, tmm_psi_delta

# =============================================================================
# Constantes físicas (necesarias para la evaluación espectral)
# =============================================================================

theta0 = 70 * np.pi / 180              # ángulo de incidencia
E = np.linspace(0.75, 3.65, 101)       # malla de energías (eV)
lambda_ = 1239.8 / E                   # conversión eV → nm

# --- Constantes ópticas del silicio (sustrato) ---
data_si = pd.read_csv('datos/Si.clc',
    skiprows=64, sep=r'\s+', header=None, encoding='latin-1', nrows=305)

E_si = data_si.iloc[:, 3].values
n_si_tab = data_si.iloc[:, 8].values
k_si_tab = data_si.iloc[:, 9].values

# =============================================================================
# Cargar dataset
# =============================================================================

df = pd.read_csv('1osc_d/dataset_1osc_d.csv')

col_params = ['A', 'E_0', 'E_g', 'C', 'eps_inf', 'd']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

# =============================================================================
# Preparar datos
# =============================================================================

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

# =============================================================================
# Definir modelo
# =============================================================================
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

# =============================================================================
# Entrenar
# =============================================================================
# EarlyStopping: detiene si val_loss no mejora en 20 épocas

early_stopper = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stopper]
)

# =============================================================================
# Evaluar sobre test: error por parámetros y error espectral
# =============================================================================

# --- Error por parámetros ---
Y_pred_norm = model.predict(X_test)
Y_pred = scaler_y.inverse_transform(Y_pred_norm)
Y_real = scaler_y.inverse_transform(Y_test)

print("\nError por parámetros (test):")
print(f"{'Parámetro':<10} {'MAE':>10} {'MSE':>12}")
print("-" * 34)
for i, param in enumerate(col_params):
    mae = mean_absolute_error(Y_real[:, i], Y_pred[:, i])
    mse = mean_squared_error(Y_real[:, i], Y_pred[:, i])
    print(f"  {param:<10} {mae:>10.4f} {mse:>12.4f}")

# --- Error espectral ---
# Reconstruye los espectros a partir de los parámetros predichos
# y los compara con los espectros reales del test set.
# Esto mide la calidad de la inversión en el espacio observable.

n_aire = np.ones(len(E)) * (1.0 + 0j)
n_si = np.interp(E, E_si, n_si_tab) + 1j * np.interp(E, E_si, k_si_tab)
X_test_real = scaler_x.inverse_transform(X_test)

errores_tan_psi = []
errores_cos_delta = []

for idx in range(len(Y_pred)):
    A_, E0_, Eg_, C_, einf_, d_ = Y_pred[idx]

    # Reconstruir espectro a partir de parámetros predichos
    eps2 = eps_im(E, A_, E0_, Eg_, C_)
    eps1 = eps_re(E, A_, E0_, Eg_, C_, einf_)
    n_, k_ = indice_refraccion(eps1, eps2)
    n_capa = n_ + 1j * k_

    psi, delta = tmm_psi_delta([n_aire, n_capa, n_si], [d_], theta0, lambda_)
    tan_psi_pred = np.tan(psi * np.pi / 180)
    cos_delta_pred = np.cos(delta * np.pi / 180)

    # Espectros reales (desnormalizados)
    tan_psi_real = X_test_real[idx, :101]
    cos_delta_real = X_test_real[idx, 101:]

    # Comprobar NaN (puede ocurrir si Eg ≥ E0 en la predicción)
    if np.any(np.isnan(tan_psi_pred)) or np.any(np.isnan(cos_delta_pred)):
        continue

    errores_tan_psi.append(np.mean(np.abs(tan_psi_pred - tan_psi_real)))
    errores_cos_delta.append(np.mean(np.abs(cos_delta_pred - cos_delta_real)))

print(f"\nError espectral (test, {len(errores_tan_psi)}/{len(Y_pred)} muestras válidas):")
print(f"  MAE tan(Ψ):  {np.mean(errores_tan_psi):.6f}")
print(f"  MAE cos(Δ):  {np.mean(errores_cos_delta):.6f}")

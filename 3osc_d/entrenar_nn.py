"""
Entrenamiento de la red neuronal para el caso multioscilador Tauc-Lorentz
con espesor variable.

Red multitarea con dos salidas:
    - Clasificación: número de osciladores (1, 2 o 3)
    - Regresión: 14 parámetros (4 por oscilador × 3 + ε∞ + d)

Los osciladores inactivos se enmascaran a cero tras la predicción,
según la clase predicha.

Entrada:  202 valores espectrales (101 tan(Ψ) + 101 cos(Δ))
Arquitectura: 256 → 128 → 128 → 64 → bifurcación (clasificación + regresión)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from modelo_tl import eps_multi, indice_refraccion, tmm_psi_delta

# =============================================================================
# Constantes físicas
# =============================================================================

THETA0 = 70 * np.pi / 180              # ángulo de incidencia
E = np.linspace(0.75, 3.65, 101)       # malla de energías (eV)
LAMBDA = 1239.8 / E                    # conversión eV → nm

# --- Constantes ópticas del silicio (sustrato) ---
data_si = pd.read_csv('datos/Si.clc',
    skiprows=64, sep=r'\s+', header=None, encoding='latin-1', nrows=305)

E_si = data_si.iloc[:, 3].values
n_si_tab = data_si.iloc[:, 8].values
k_si_tab = data_si.iloc[:, 9].values

n_si = np.interp(E, E_si, n_si_tab) + 1j * np.interp(E, E_si, k_si_tab)
n_aire = np.ones(len(E)) * (1.0 + 0j)

# =============================================================================
# Cargar dataset
# =============================================================================

df = pd.read_csv('3osc_d/dataset_3osc_d.csv')

col_params = ['A1', 'E0_1', 'Eg_1', 'C1',
              'A2', 'E0_2', 'Eg_2', 'C2',
              'A3', 'E0_3', 'Eg_3', 'C3',
              'eps_inf', 'd']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

X = df[col_tan + col_cos].values       # (60000, 202) - espectros
Y_reg = df[col_params].values          # (60000, 14)  - parámetros
Y_clf = df['n_osc'].values             # (60000,)     - clase (1, 2 o 3)

# =============================================================================
# Preparar datos
# =============================================================================

Y_clf_idx = Y_clf - 1

# División 75% entrenamiento, 25% test
X_train, X_test, Yreg_train, Yreg_test, Yclf_train, Yclf_test = train_test_split(
    X, Y_reg, Y_clf_idx, test_size=0.25, random_state=42, stratify=Y_clf_idx)

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)

scaler_Y = MinMaxScaler()
Yreg_train_s = scaler_Y.fit_transform(Yreg_train)
Yreg_test_s = scaler_Y.transform(Yreg_test)

# =============================================================================
# Definir modelo multitarea
# =============================================================================

tf.random.set_seed(42)

inputs = layers.Input(shape=(202,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64,  activation='relu')(x)

out_clf = layers.Dense(3,  activation='softmax', name='n_osc')(x)
out_reg = layers.Dense(14, activation='linear',  name='params')(x)

model = Model(inputs, [out_clf, out_reg])

model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss=['sparse_categorical_crossentropy', 'mse'],
    loss_weights=[1.0, 1.0],
    metrics=[['accuracy'], []]
)

# =============================================================================
# Entrenar
# =============================================================================

history = model.fit(
    X_train_s, [Yclf_train, Yreg_train_s],
    validation_split=0.2,
    epochs=300, batch_size=32,
    callbacks=[EarlyStopping(patience=20, restore_best_weights=True)]
)

# =============================================================================
# Evaluar sobre test: clasificación
# =============================================================================

y_clf_pred_prob, y_reg_pred_s = model.predict(X_test_s, verbose=0)
y_clf_pred = np.argmax(y_clf_pred_prob, axis=1)

print("\nMatriz de confusión (test):")
print("         pred=1  pred=2  pred=3")
cm = confusion_matrix(Yclf_test, y_clf_pred)
for i, row in enumerate(cm):
    print(f"real={i+1}:   " + "  ".join(f"{v:5d}" for v in row))

print("\nAccuracy por clase:")
print(f"  1 oscilador:   {cm[0,0]/cm[0].sum():.3f}")
print(f"  2 osciladores: {cm[1,1]/cm[1].sum():.3f}")
print(f"  3 osciladores: {cm[2,2]/cm[2].sum():.3f}")

# =============================================================================
# Evaluar sobre test: error por parámetros
# =============================================================================

y_reg_pred = scaler_Y.inverse_transform(y_reg_pred_s)
y_reg_true = scaler_Y.inverse_transform(Yreg_test_s)

for i in range(len(y_reg_pred)):
    n_osc_i = y_clf_pred[i] + 1
    if n_osc_i < 2:
        y_reg_pred[i, 4:8] = 0.0
    if n_osc_i < 3:
        y_reg_pred[i, 8:12] = 0.0

nombres_params = ['A1', 'E0_1', 'Eg_1', 'C1',
                  'A2', 'E0_2', 'Eg_2', 'C2',
                  'A3', 'E0_3', 'Eg_3', 'C3', 'eps_inf', 'd']

print("\nError por parámetros (test):")
print(f"{'Parámetro':<10} {'MAE':>10} {'MSE':>12}")
print("-" * 34)
for j, name in enumerate(nombres_params):
    diff = y_reg_true[:, j] - y_reg_pred[:, j]
    print(f"  {name:<10} {np.mean(np.abs(diff)):>10.4f} {np.mean(diff**2):>12.4f}")

# =============================================================================
# Evaluar sobre test: error espectral
# =============================================================================

X_test_real = scaler_X.inverse_transform(X_test_s)

def reconstruir_espectro(params, n_osc):
    """
    Reconstruye tan(Ψ) y cos(Δ) a partir de un vector de 14 parámetros
    y el número de osciladores activos.
    """
    osciladores = []
    for i in range(n_osc):
        A  = params[4*i]
        E0 = params[4*i + 1]
        Eg = params[4*i + 2]
        C  = params[4*i + 3]
        osciladores.append((A, E0, Eg, C))

    eps_inf = params[12]
    d_capa = params[13]

    eps1, eps2 = eps_multi(E, osciladores, eps_inf)
    n, k = indice_refraccion(eps1, eps2)
    n_capa = n + 1j * k

    psi, delta = tmm_psi_delta([n_aire, n_capa, n_si], [d_capa], THETA0, LAMBDA)
    tan_psi = np.tan(psi * np.pi / 180)
    cos_delta = np.cos(delta * np.pi / 180)
    return tan_psi, cos_delta

mse_tan_list, mse_cos_list = [], []
mae_tan_list, mae_cos_list = [], []
descartadas = 0

for i in range(len(X_test_s)):
    n_osc_i = y_clf_pred[i] + 1
    tan_psi_real = X_test_real[i, :101]
    cos_delta_real = X_test_real[i, 101:]

    try:
        tan_psi_p, cos_delta_p = reconstruir_espectro(y_reg_pred[i], n_osc_i)

        if not (np.all(np.isfinite(tan_psi_p)) and np.all(np.isfinite(cos_delta_p))):
            descartadas += 1
            continue

        mse_tan_list.append(np.mean((tan_psi_real - tan_psi_p)**2))
        mse_cos_list.append(np.mean((cos_delta_real - cos_delta_p)**2))
        mae_tan_list.append(np.mean(np.abs(tan_psi_real - tan_psi_p)))
        mae_cos_list.append(np.mean(np.abs(cos_delta_real - cos_delta_p)))
    except:
        descartadas += 1

print(f"\nError espectral (test, {len(mse_tan_list)}/{len(X_test_s)} muestras válidas):")
print(f"  MSE tan(Ψ):           {np.mean(mse_tan_list):.6f}")
print(f"  MSE cos(Δ):           {np.mean(mse_cos_list):.6f}")
print(f"  MSE espectral agregado: {(np.mean(mse_tan_list) + np.mean(mse_cos_list)) / 2:.6f}")
print(f"  MAE tan(Ψ):           {np.mean(mae_tan_list):.6f}")
print(f"  MAE cos(Δ):           {np.mean(mae_cos_list):.6f}")
print(f"  MAE espectral agregado: {(np.mean(mae_tan_list) + np.mean(mae_cos_list)) / 2:.6f}")

"""
Entrenamiento de la PINN para el caso de 1 oscilador Tauc-Lorentz
con espesor variable.

Combina una loss de datos (MSE sobre parámetros normalizados) con una
loss física basada en la reconstrucción completa de espectros (tan(Ψ), cos(Δ))
mediante el TMM en TensorFlow nativo.

A diferencia de la PINN de 1osc (que solo usa ε₂), aquí la physics loss
reconstruye los espectros elipsométricos completos y los compara con la
entrada de la red.

Entrada:  202 valores espectrales (101 tan(Ψ) + 101 cos(Δ))
Salida:   6 parámetros (A, E₀, Eg, C, ε∞, d)

loss_total = loss_data + λ · loss_physics   (λ = 0.005)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modelo_tl_tf import eps_im_tf, eps_re_tf, indice_refraccion_tf, tmm_psi_delta_tf

# =============================================================================
# Constantes físicas del experimento
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
n_si_complex = np.interp(E, E_si, n_si_tab) + 1j * np.interp(E, E_si, k_si_tab)

# --- Versiones TensorFlow (constantes globales para la PINN) ---
E_tf = tf.constant(E, dtype=tf.float32)
lambda_tf = tf.constant(lambda_, dtype=tf.float32)

n_aire_tf = tf.complex(
    tf.ones(len(E), dtype=tf.float32),
    tf.zeros(len(E), dtype=tf.float32)
)
n_si_tf = tf.complex(
    tf.constant(np.real(n_si_complex), dtype=tf.float32),
    tf.constant(np.imag(n_si_complex), dtype=tf.float32)
)

# =============================================================================
# Cargar dataset y preparar datos
# =============================================================================

df = pd.read_csv('1osc_d/dataset_1osc_d.csv')

col_params = ['A', 'E_0', 'E_g', 'C', 'eps_inf', 'd']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

# Entrada: espectros, Salida: parámetros TL + espesor
X = df[col_tan + col_cos].values   # (40000, 202)
Y = df[col_params].values          # (40000, 6)

# Normalización
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# División 75% entrenamiento, 25% test
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.25, random_state=42)

# =============================================================================
# Constantes para la physics loss
# =============================================================================

# Media y desviación de los scalers para desnormalizar dentro del grafo de TF
Y_mean = tf.constant(scaler_y.mean_, dtype=tf.float32)
Y_std = tf.constant(scaler_y.scale_, dtype=tf.float32)
X_mean = tf.constant(scaler_x.mean_, dtype=tf.float32)
X_std = tf.constant(scaler_x.scale_, dtype=tf.float32)

LAMBDA_PHYS = 0.005    # peso de la loss física (menor que en 1osc porque
                        # la loss TMM tiene mayor magnitud que la de ε₂)
BS = 32

tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# Physics loss: reconstrucción completa de espectros via TMM
# =============================================================================

@tf.function
def physics_loss(Y_pred_real, X_batch_real):
    """
    Reconstruye los espectros (tan(Ψ), cos(Δ)) a partir de los parámetros
    predichos por la red, usando el modelo directo completo:
        parámetros → ε₁,ε₂ (Tauc-Lorentz) → n,k → TMM → tan(Ψ), cos(Δ)
    
    Compara los espectros reconstruidos con los espectros de entrada
    (desnormalizados) y devuelve el residuo.
    
    Aplica clamping físico a los parámetros predichos para garantizar
    que se mantengan en rangos válidos durante el entrenamiento.
    """
    # Separar parámetros predichos
    A       = Y_pred_real[:, 0]
    E0      = Y_pred_real[:, 1]
    Eg      = Y_pred_real[:, 2]
    C       = Y_pred_real[:, 3]
    eps_inf = Y_pred_real[:, 4]
    d       = Y_pred_real[:, 5]

    # Clamping físico — mismos rangos que el generador del dataset
    A       = tf.clip_by_value(A, 50.0, 200.0)
    E0      = tf.clip_by_value(E0, 1.0, 5.0)
    Eg      = tf.clip_by_value(Eg, 0.5, 5.0)
    C       = tf.clip_by_value(C, 0.5, 5.0)
    eps_inf = tf.clip_by_value(eps_inf, 1.0, 3.0)
    d       = tf.clip_by_value(d, 20.0, 300.0)

    # Restricciones físicas del modelo Tauc-Lorentz
    Eg = tf.minimum(Eg, E0 - 0.1)      # Eg < E0
    C = tf.minimum(C, 1.4 * E0)        # E0 > C/√2 → C < √2·E0 ≈ 1.4·E0

    # Forward físico: parámetros → función dieléctrica → índice → TMM → espectros
    eps2 = eps_im_tf(E_tf, A, E0, Eg, C)
    eps1 = eps_re_tf(E_tf, A, E0, Eg, C, eps_inf)
    n_capa, k_capa = indice_refraccion_tf(eps1, eps2)
    n_capa_complex = tf.complex(n_capa, k_capa)

    tan_psi_pred, cos_delta_pred = tmm_psi_delta_tf(
        [n_aire_tf, n_capa_complex, n_si_tf],
        [d],
        theta0,
        lambda_tf
    )

    # Comparar espectros reconstruidos con los de entrada
    espectro_pred = tf.concat([tan_psi_pred, cos_delta_pred], axis=1)
    residuo = espectro_pred - X_batch_real

    # Reemplazar NaN/Inf por ceros (red de seguridad por combinaciones límite)
    residuo = tf.where(tf.math.is_finite(residuo), residuo, tf.zeros_like(residuo))

    return residuo

# =============================================================================
# Definir modelo
# =============================================================================
# Arquitectura más ancha que la NN estándar: capa inicial de 512 y dos capas
# de 256, para manejar la complejidad de la inversión con espesor variable

model_pinn = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(202,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(6)     # salida: A, E₀, Eg, C, ε∞, d (normalizados)
])

optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)   # clipnorm para estabilidad

# =============================================================================
# Paso de entrenamiento
# =============================================================================

@tf.function
def train_step(X_batch, Y_batch):
    """
    Un paso de entrenamiento de la PINN:
    1. Forward pass → predicción normalizada
    2. loss_data: MSE entre predicción y target (espacio normalizado)
    3. Desnormalizar predicción y entrada → espacio físico
    4. loss_phys: MSE del residuo espectral (TMM completo)
    5. loss_total = loss_data + λ · loss_phys
    6. Backpropagation y actualización de pesos
    """
    with tf.GradientTape() as tape:
        Y_pred_norm = model_pinn(X_batch, training=True)
        loss_data = tf.reduce_mean((Y_pred_norm - Y_batch)**2)

        # Desnormalizar para la physics loss
        Y_pred_real = Y_pred_norm * Y_std + Y_mean
        X_batch_real = X_batch * X_std + X_mean

        residuo = physics_loss(Y_pred_real, X_batch_real)
        loss_phys = tf.reduce_mean(residuo**2)
        loss_phys = tf.where(tf.math.is_nan(loss_phys), tf.constant(0.0), loss_phys)

        loss_total = loss_data + LAMBDA_PHYS * loss_phys

    gradients = tape.gradient(loss_total, model_pinn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_pinn.trainable_variables))
    return loss_data, loss_phys

# =============================================================================
# Preparar datasets de TensorFlow
# =============================================================================

# Separar validación (20%) del conjunto de entrenamiento
n_val = int(len(X_train) * 0.2)
idx_perm = np.random.RandomState(42).permutation(len(X_train))
val_idx, train_idx = idx_perm[:n_val], idx_perm[n_val:]

X_tr = tf.cast(X_train[train_idx], tf.float32)
Y_tr = tf.cast(Y_train[train_idx], tf.float32)
X_val = tf.cast(X_train[val_idx], tf.float32)
Y_val = tf.cast(Y_train[val_idx], tf.float32)

dataset_train = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr)).shuffle(10000, seed=42).batch(BS)
dataset_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BS)

# =============================================================================
# Bucle de entrenamiento con early stopping manual
# =============================================================================

EPOCHS = 300
PATIENCE = 20
best_val_loss = float('inf')
best_weights = None
patience_counter = 0

hist_ld_train, hist_lp_train, hist_total_train = [], [], []
hist_ld_val, hist_lp_val, hist_total_val = [], [], []

for epoch in range(EPOCHS):
    # --- Entrenamiento ---
    ep_ld_tr, ep_lp_tr = [], []
    for X_batch, Y_batch in dataset_train:
        ld, lp = train_step(X_batch, Y_batch)
        ep_ld_tr.append(ld.numpy())
        ep_lp_tr.append(lp.numpy())

    # --- Validación (sin gradientes) ---
    ep_ld_val, ep_lp_val = [], []
    for X_batch, Y_batch in dataset_val:
        Y_pred_norm = model_pinn(X_batch, training=False)
        ld = tf.reduce_mean((Y_pred_norm - Y_batch)**2)
        Y_pred_real = Y_pred_norm * Y_std + Y_mean
        X_batch_real = X_batch * X_std + X_mean
        residuo = physics_loss(Y_pred_real, X_batch_real)
        lp = tf.reduce_mean(residuo**2)
        lp = tf.where(tf.math.is_nan(lp), tf.constant(0.0), lp)
        ep_ld_val.append(ld.numpy())
        ep_lp_val.append(lp.numpy())

    # Promediar losses de la época
    mean_ld_tr = np.mean(ep_ld_tr)
    mean_lp_tr = np.mean(ep_lp_tr)
    mean_ld_val = np.mean(ep_ld_val)
    mean_lp_val = np.mean(ep_lp_val)

    hist_ld_train.append(mean_ld_tr)
    hist_lp_train.append(mean_lp_tr)
    hist_total_train.append(mean_ld_tr + LAMBDA_PHYS * mean_lp_tr)
    hist_ld_val.append(mean_ld_val)
    hist_lp_val.append(mean_lp_val)
    hist_total_val.append(mean_ld_val + LAMBDA_PHYS * mean_lp_val)

    # Early stopping sobre loss total de validación
    val_loss = mean_ld_val + LAMBDA_PHYS * mean_lp_val
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model_pinn.get_weights()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping en época {epoch+1}')
            break

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Época {epoch+1:3d} | train: ld={mean_ld_tr:.4f} lp={mean_lp_tr:.4f} | '
              f'val: ld={mean_ld_val:.4f} lp={mean_lp_val:.4f}')

# Restaurar los mejores pesos
model_pinn.set_weights(best_weights)
print('PINN entrenada')

# =============================================================================
# Evaluar sobre test: error por parámetros y error espectral
# =============================================================================

from modelo_tl import eps_im, eps_re, indice_refraccion, tmm_psi_delta

# --- Error por parámetros ---
Y_pred_norm = model_pinn.predict(tf.cast(X_test, tf.float32))
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
# Para cada muestra del test, reconstruye los espectros (tan(Ψ), cos(Δ))
# a partir de los parámetros predichos y los compara con los espectros reales.
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

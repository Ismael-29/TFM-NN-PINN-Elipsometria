"""
Entrenamiento de la PINN para el caso de 1 oscilador Tauc-Lorentz
con espesor fijo.

Combina una loss de datos (MSE sobre parámetros normalizados) con una
loss física basada en la diferencia de ε₂ entre parámetros predichos
y reales, calculada con el modelo Tauc-Lorentz en TensorFlow nativo.

Entrada:  202 valores espectrales (101 tan(Ψ) + 101 cos(Δ))
Salida:   5 parámetros (A, E₀, Eg, C, ε∞)

loss_total = loss_data + λ · loss_physics   (λ = 0.01)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# =============================================================================
# Cargar dataset
# =============================================================================

df = pd.read_csv('1osc/dataset_1osc.csv')

col_params = ['A', 'E_0', 'E_g', 'C', 'eps_inf']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

# =============================================================================
# Preparar datos
# =============================================================================

# Entrada: espectros (tan(Ψ), cos(Δ)), Salida: parámetros TL
X = df[col_cos + col_tan].values
Y = df[col_params].values

# Normalización: StandardScaler para entrada y salida
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

# Malla de energías sobre la que se evalúa ε₂ en la loss física
E_grid = tf.constant(np.linspace(0.75, 3.65, 101), dtype=tf.float32)

# Media y desviación del scaler para desnormalizar dentro del grafo de TF
Y_mean = tf.constant(scaler_y.mean_, dtype=tf.float32)
Y_std = tf.constant(scaler_y.scale_, dtype=tf.float32)

LAMBDA_PHYS = 0.01     # peso de la loss física (valores mayores interfieren
                        # con la recuperación de parámetros por degeneración espectral)
BS = 32                 # batch size

# =============================================================================
# Physics loss: ε₂ del Tauc-Lorentz en TensorFlow
# =============================================================================

def eps_im_tf(E, A, E0, Eg, C):
    """
    Calcula ε₂(E) vectorizado sobre batch y energías.
    Implementación en TF nativo para que el gradiente fluya.
    
    E:  (N_E,)      →  se expande a (1, N_E)
    A, E0, Eg, C: (batch,) → se expanden a (batch, 1)
    Retorna: (batch, N_E)
    """
    E = E[tf.newaxis, :]
    A = A[:, tf.newaxis]
    E0 = E0[:, tf.newaxis]
    Eg = Eg[:, tf.newaxis]
    C = C[:, tf.newaxis]

    numerador = A * E0 * C * (E - Eg)**2
    denominador = ((E**2 - E0**2)**2 + (C * E)**2) * E
    eps2 = numerador / (denominador + 1e-10)    # epsilon para evitar div/0
    eps2 = tf.where(E > Eg, eps2, tf.zeros_like(eps2))  # ε₂ = 0 si E ≤ Eg
    return eps2

def net_physics(y_pred_real, y_true_real):
    """
    Calcula el residuo físico: diferencia de ε₂ entre los parámetros
    predichos por la red y los parámetros reales del dataset.
    
    Los parámetros predichos se clippean a sus rangos válidos para
    evitar valores no físicos durante el entrenamiento.
    """
    # Clippear parámetros predichos a rangos válidos
    A_pred = tf.clip_by_value(y_pred_real[:, 0], 50.0, 200.0)
    E0_pred = tf.clip_by_value(y_pred_real[:, 1], 1.0, 5.0)
    Eg_pred = tf.clip_by_value(y_pred_real[:, 2], 0.5, 5.0)
    C_pred = tf.clip_by_value(y_pred_real[:, 3], 0.5, 5.0)
    Eg_pred = tf.minimum(Eg_pred, E0_pred - 0.01)  # garantizar Eg < E0

    # Parámetros reales (no necesitan clipping)
    A_true = y_true_real[:, 0]
    E0_true = y_true_real[:, 1]
    Eg_true = y_true_real[:, 2]
    C_true = y_true_real[:, 3]

    # ε₂ predicho vs ε₂ real sobre la malla de energías
    eps2_pred = eps_im_tf(E_grid, A_pred, E0_pred, Eg_pred, C_pred)
    eps2_true = eps_im_tf(E_grid, A_true, E0_true, Eg_true, C_true)

    return eps2_pred - eps2_true

# =============================================================================
# Definir modelo (misma arquitectura que la NN estándar)
# =============================================================================

model_pinn = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(202,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(5)     # salida: A, E₀, Eg, C, ε∞ (normalizados)
])

optimizer = Adam(learning_rate=0.0005)

# =============================================================================
# Paso de entrenamiento
# =============================================================================

@tf.function
def train_step(X_batch, Y_batch):
    """
    Un paso de entrenamiento de la PINN:
    1. Forward pass → predicción normalizada
    2. loss_data: MSE entre predicción y target (espacio normalizado)
    3. Desnormalizar predicción y target → espacio físico
    4. loss_phys: MSE del residuo de ε₂ (espacio físico)
    5. loss_total = loss_data + λ · loss_phys
    6. Backpropagation y actualización de pesos
    """
    with tf.GradientTape() as tape:
        Y_pred_norm = model_pinn(X_batch, training=True)
        loss_data = tf.reduce_mean((Y_pred_norm - Y_batch)**2)

        # Desnormalizar para calcular la loss física
        Y_pred_real = Y_pred_norm * Y_std + Y_mean
        Y_true_real = Y_batch * Y_std + Y_mean

        residuo = net_physics(Y_pred_real, Y_true_real)
        loss_phys = tf.reduce_mean(residuo**2)
        # Protección contra NaN (puede ocurrir con parámetros extremos)
        loss_phys = tf.where(tf.math.is_nan(loss_phys), tf.constant(0.0), loss_phys)

        loss_total = loss_data + LAMBDA_PHYS * loss_phys

    gradients = tape.gradient(loss_total, model_pinn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_pinn.trainable_variables))
    return loss_data, loss_phys, loss_total

# =============================================================================
# Preparar datasets de TensorFlow
# =============================================================================

# Separar validación (20%) del conjunto de entrenamiento
n_val = int(len(X_train) * 0.2)
indices = np.random.RandomState(42).permutation(len(X_train))
val_idx, train_idx = indices[:n_val], indices[n_val:]

X_tr = tf.cast(X_train[train_idx], tf.float32)
Y_tr = tf.cast(Y_train[train_idx], tf.float32)
X_val = tf.cast(X_train[val_idx], tf.float32)
Y_val = tf.cast(Y_train[val_idx], tf.float32)

# Crear datasets con shuffle y batching
dataset_train = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr)).shuffle(10000).batch(BS)
dataset_val = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BS)

# =============================================================================
# Bucle de entrenamiento con early stopping manual
# =============================================================================
# Se implementa manualmente (en lugar de callbacks de Keras) porque el
# train_step personalizado con GradientTape no es compatible con model.fit()

EPOCHS = 300
PATIENCE = 20           # épocas sin mejora antes de parar
best_val_loss = float('inf')
best_weights = None
patience_counter = 0

history_data_train, history_data_val = [], []
history_phys_train, history_phys_val = [], []

for epoch in range(EPOCHS):
    # --- Entrenamiento ---
    epoch_ld_train, epoch_lp_train = [], []
    for X_batch, Y_batch in dataset_train:
        ld, lp, lt = train_step(X_batch, Y_batch)
        epoch_ld_train.append(ld.numpy())
        epoch_lp_train.append(lp.numpy())

    # --- Validación (sin gradientes) ---
    epoch_ld_val, epoch_lp_val = [], []
    for X_batch, Y_batch in dataset_val:
        Y_pred_norm = model_pinn(X_batch, training=False)
        ld = tf.reduce_mean((Y_pred_norm - Y_batch)**2)
        Y_pred_real = Y_pred_norm * Y_std + Y_mean
        Y_true_real = Y_batch * Y_std + Y_mean
        residuo = net_physics(Y_pred_real, Y_true_real)
        lp = tf.reduce_mean(residuo**2)
        lp = tf.where(tf.math.is_nan(lp), tf.constant(0.0), lp)
        epoch_ld_val.append(ld.numpy())
        epoch_lp_val.append(lp.numpy())

    # Promediar losses de la época
    mean_ld_train = np.mean(epoch_ld_train)
    mean_lp_train = np.mean(epoch_lp_train)
    mean_ld_val = np.mean(epoch_ld_val)
    mean_lp_val = np.mean(epoch_lp_val)

    history_data_train.append(mean_ld_train)
    history_phys_train.append(mean_lp_train)
    history_data_val.append(mean_ld_val)
    history_phys_val.append(mean_lp_val)

    # Early stopping sobre loss total de validación
    val_loss_total = mean_ld_val + LAMBDA_PHYS * mean_lp_val
    if val_loss_total < best_val_loss:
        best_val_loss = val_loss_total
        best_weights = model_pinn.get_weights()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping en época {epoch+1}')
            break

    if (epoch + 1) % 50 == 0:
        print(f'Época {epoch+1}: loss_data={mean_ld_train:.4f}, '
              f'val_loss_data={mean_ld_val:.4f}, loss_phys={mean_lp_train:.4f}')

# Restaurar los mejores pesos
model_pinn.set_weights(best_weights)

# =============================================================================
# Evaluar sobre test
# =============================================================================

Y_pred_norm = model_pinn.predict(tf.cast(X_test, tf.float32))
Y_pred = scaler_y.inverse_transform(Y_pred_norm)
Y_real = scaler_y.inverse_transform(Y_test)

print("\nResultados sobre test:")
for i, param in enumerate(col_params):
    mae = mean_absolute_error(Y_real[:, i], Y_pred[:, i])
    mse = mean_squared_error(Y_real[:, i], Y_pred[:, i])
    print(f'  {param}: MAE = {mae:.4f}, MSE = {mse:.4f}')

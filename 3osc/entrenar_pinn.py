"""
Entrenamiento de la PINN para el caso multioscilador Tauc-Lorentz
con espesor fijo (d = 100 nm).

Red multitarea con dos salidas:
    - Clasificación: número de osciladores (1, 2 o 3)
    - Regresión: 13 parámetros (4 por oscilador × 3 + ε∞)

Incorpora una loss física basada en la reconstrucción espectral mediante
el modelo directo TMM implementado en TensorFlow nativo.

Entrada:  202 valores espectrales (101 tan(Ψ) + 101 cos(Δ))
Arquitectura: 256 → 128 → 128 → 64 → bifurcación (clasificación + regresión)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from modelo_tl import eps_multi, indice_refraccion, tmm_psi_delta
from modelo_tl_tf import eps_im_tf, eps_re_tf, indice_refraccion_tf, tmm_psi_delta_tf

# =============================================================================
# Constantes físicas
# =============================================================================

THETA0 = 70 * np.pi / 180              # ángulo de incidencia
D_CAPA = 100.0                         # espesor fijo en nm
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

# Constantes TF para la physics loss
E_tf = tf.constant(E, dtype=tf.float32)
lambda_tf = tf.constant(LAMBDA, dtype=tf.float32)
n_aire_tf = tf.constant(n_aire, dtype=tf.complex64)
n_si_tf = tf.constant(n_si, dtype=tf.complex64)
d_capa_tf = tf.constant(D_CAPA, shape=(1,))

# =============================================================================
# Cargar dataset
# =============================================================================

df = pd.read_csv('3osc/dataset_3osc.csv')

col_params = ['A1', 'E0_1', 'Eg_1', 'C1',
              'A2', 'E0_2', 'Eg_2', 'C2',
              'A3', 'E0_3', 'Eg_3', 'C3',
              'eps_inf']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

X = df[col_tan + col_cos].values       # (72000, 202) - espectros
Y_reg = df[col_params].values          # (72000, 13)  - parámetros
Y_clf = df['n_osc'].values             # (72000,)     - clase (1, 2 o 3)

# =============================================================================
# Preparar datos
# =============================================================================

Y_clf_idx = Y_clf - 1

X_temp, X_test, Yreg_temp, Yreg_test, Yclf_temp, Yclf_test = train_test_split(
    X, Y_reg, Y_clf_idx, test_size=0.15, random_state=42, stratify=Y_clf_idx)
X_train, X_val, Yreg_train, Yreg_val, Yclf_train, Yclf_val = train_test_split(
    X_temp, Yreg_temp, Yclf_temp, test_size=0.15/0.85, random_state=42, stratify=Yclf_temp)

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)

scaler_Y = MinMaxScaler()
Yreg_train_s = scaler_Y.fit_transform(Yreg_train)
Yreg_val_s = scaler_Y.transform(Yreg_val)
Yreg_test_s = scaler_Y.transform(Yreg_test)

# =============================================================================
# Physics loss
# =============================================================================

@tf.function
def physics_loss_multi(Y_pred_real, X_batch_real):
    """
    Calcula el residuo espectral entre los espectros reconstruidos a partir
    de los parámetros predichos y los espectros reales de entrada.

    Parámetros:
        Y_pred_real  : (batch, 13) - parámetros TL desescalados
        X_batch_real : (batch, 202) - espectros desescalados

    Retorna:
        residuo : (batch, 202) - diferencia espectral
    """
    A1, E0_1, Eg_1, C1 = Y_pred_real[:,0], Y_pred_real[:,1], Y_pred_real[:,2], Y_pred_real[:,3]
    A2, E0_2, Eg_2, C2 = Y_pred_real[:,4], Y_pred_real[:,5], Y_pred_real[:,6], Y_pred_real[:,7]
    A3, E0_3, Eg_3, C3 = Y_pred_real[:,8], Y_pred_real[:,9], Y_pred_real[:,10], Y_pred_real[:,11]
    eps_inf = Y_pred_real[:, 12]

    A1   = tf.clip_by_value(A1,  50.0, 200.0)
    E0_1 = tf.clip_by_value(E0_1, 1.0, 5.0)
    Eg_1 = tf.clip_by_value(Eg_1, 0.5, 5.0)
    C1   = tf.clip_by_value(C1,  0.5, 5.0)
    A2   = tf.clip_by_value(A2,  50.0, 200.0)
    E0_2 = tf.clip_by_value(E0_2, 1.0, 5.0)
    Eg_2 = tf.clip_by_value(Eg_2, 0.5, 5.0)
    C2   = tf.clip_by_value(C2,  0.5, 5.0)
    A3   = tf.clip_by_value(A3,  50.0, 200.0)
    E0_3 = tf.clip_by_value(E0_3, 1.0, 5.0)
    Eg_3 = tf.clip_by_value(Eg_3, 0.5, 5.0)
    C3   = tf.clip_by_value(C3,  0.5, 5.0)
    eps_inf = tf.clip_by_value(eps_inf, 1.0, 3.0)

    Eg_1 = tf.minimum(Eg_1, E0_1 - 0.1)
    C1   = tf.minimum(C1, 1.4 * E0_1)
    Eg_2 = tf.minimum(Eg_2, E0_2 - 0.1)
    C2   = tf.minimum(C2, 1.4 * E0_2)
    Eg_3 = tf.minimum(Eg_3, E0_3 - 0.1)
    C3   = tf.minimum(C3, 1.4 * E0_3)

    eps2 = (eps_im_tf(E_tf, A1, E0_1, Eg_1, C1)
          + eps_im_tf(E_tf, A2, E0_2, Eg_2, C2)
          + eps_im_tf(E_tf, A3, E0_3, Eg_3, C3))

    eps1 = (eps_re_tf(E_tf, A1, E0_1, Eg_1, C1, tf.zeros_like(eps_inf))
          + eps_re_tf(E_tf, A2, E0_2, Eg_2, C2, tf.zeros_like(eps_inf))
          + eps_re_tf(E_tf, A3, E0_3, Eg_3, C3, tf.zeros_like(eps_inf))
          + eps_inf[:, tf.newaxis])

    n_capa, k_capa = indice_refraccion_tf(eps1, eps2)
    n_capa_complex = tf.complex(n_capa, k_capa)

    tan_psi_pred, cos_delta_pred = tmm_psi_delta_tf(
        [n_aire_tf, n_capa_complex, n_si_tf],
        [d_capa_tf], THETA0, lambda_tf)

    espectro_pred = tf.concat([tan_psi_pred, cos_delta_pred], axis=1)
    residuo = espectro_pred - X_batch_real
    residuo = tf.where(tf.math.is_finite(residuo), residuo, tf.zeros_like(residuo))

    return residuo

# =============================================================================
# Definir modelo PINN
# =============================================================================

tf.random.set_seed(42)
np.random.seed(42)

Y_range = tf.constant(scaler_Y.data_range_, dtype=tf.float32)
Y_min   = tf.constant(scaler_Y.data_min_,   dtype=tf.float32)
X_mean  = tf.constant(scaler_X.mean_,        dtype=tf.float32)
X_std   = tf.constant(scaler_X.scale_,       dtype=tf.float32)

lambda_phys = 0.005

inputs = layers.Input(shape=(202,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64,  activation='relu')(x)

out_clf = layers.Dense(3,  activation='softmax', name='n_osc')(x)
out_reg = layers.Dense(13, activation='linear',  name='params')(x)

model_pinn = Model(inputs, [out_clf, out_reg])

optimizer = Adam(learning_rate=5e-4, clipnorm=1.0)

# =============================================================================
# Bucle de entrenamiento
# =============================================================================

@tf.function
def train_step(X_batch, Y_reg_batch, Y_clf_batch):
    with tf.GradientTape() as tape:
        y_clf_pred, y_reg_pred = model_pinn(X_batch, training=True)

        loss_reg = tf.reduce_mean(tf.square(Y_reg_batch - y_reg_pred))
        loss_clf = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(Y_clf_batch, y_clf_pred))

        Y_pred_real  = y_reg_pred * Y_range + Y_min
        X_batch_real = X_batch * X_std + X_mean

        residuo = physics_loss_multi(Y_pred_real, X_batch_real)
        loss_phys = tf.reduce_mean(residuo**2)
        loss_phys = tf.where(tf.math.is_nan(loss_phys), tf.constant(0.0), loss_phys)

        loss_total = loss_clf + loss_reg + lambda_phys * loss_phys

    gradients = tape.gradient(loss_total, model_pinn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_pinn.trainable_variables))
    return loss_reg, loss_phys, loss_clf

dataset_train = tf.data.Dataset.from_tensor_slices((
    tf.cast(X_train_s, tf.float32),
    tf.cast(Yreg_train_s, tf.float32),
    tf.cast(Yclf_train, tf.int32)
)).shuffle(10000, seed=42).batch(32)

dataset_val = tf.data.Dataset.from_tensor_slices((
    tf.cast(X_val_s, tf.float32),
    tf.cast(Yreg_val_s, tf.float32),
    tf.cast(Yclf_val, tf.int32)
)).batch(32)

hist = {'reg_tr':[], 'phys_tr':[], 'clf_tr':[],
        'reg_val':[], 'phys_val':[], 'clf_val':[]}

best_val_loss = float('inf')
best_weights = None
patience_counter = 0

for epoch in range(300):
    ep_reg, ep_phys, ep_clf = [], [], []
    for X_b, Y_b, clf_b in dataset_train:
        lr, lp, lc = train_step(X_b, Y_b, clf_b)
        ep_reg.append(lr.numpy())
        ep_phys.append(lp.numpy())
        ep_clf.append(lc.numpy())

    ep_reg_v, ep_phys_v, ep_clf_v = [], [], []
    for X_b, Y_b, clf_b in dataset_val:
        y_clf_p, y_reg_p = model_pinn(X_b, training=False)

        lr = tf.reduce_mean(tf.square(Y_b - y_reg_p))
        lc = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(clf_b, y_clf_p))

        Y_pred_real  = y_reg_p * Y_range + Y_min
        X_batch_real = X_b * X_std + X_mean

        residuo = physics_loss_multi(Y_pred_real, X_batch_real)
        lp = tf.reduce_mean(residuo**2)
        lp = tf.where(tf.math.is_nan(lp), tf.constant(0.0), lp)

        ep_reg_v.append(lr.numpy())
        ep_phys_v.append(lp.numpy())
        ep_clf_v.append(lc.numpy())

    m_reg = np.mean(ep_reg);     m_phys = np.mean(ep_phys);     m_clf = np.mean(ep_clf)
    m_reg_v = np.mean(ep_reg_v); m_phys_v = np.mean(ep_phys_v); m_clf_v = np.mean(ep_clf_v)

    hist['reg_tr'].append(m_reg);     hist['phys_tr'].append(m_phys);     hist['clf_tr'].append(m_clf)
    hist['reg_val'].append(m_reg_v);  hist['phys_val'].append(m_phys_v);  hist['clf_val'].append(m_clf_v)

    val_total = m_clf_v + m_reg_v + lambda_phys * m_phys_v

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época {epoch+1:3d} | "
              f"train: clf={m_clf:.4f} reg={m_reg:.4f} phys={m_phys:.4f} | "
              f"val: clf={m_clf_v:.4f} reg={m_reg_v:.4f} phys={m_phys_v:.4f}")

    if val_total < best_val_loss:
        best_val_loss = val_total
        best_weights = model_pinn.get_weights()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 20:
            print(f'Early stopping en época {epoch+1}')
            break

model_pinn.set_weights(best_weights)
print('PINN multi-oscilador entrenada')

# =============================================================================
# Evaluar sobre test: clasificación
# =============================================================================

y_clf_pred_prob, y_reg_pred_s = model_pinn.predict(X_test_s, verbose=0)
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
                  'A3', 'E0_3', 'Eg_3', 'C3', 'eps_inf']

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
    Reconstruye tan(Ψ) y cos(Δ) a partir de un vector de 13 parámetros
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

    eps1, eps2 = eps_multi(E, osciladores, eps_inf)
    n, k = indice_refraccion(eps1, eps2)
    n_capa = n + 1j * k

    psi, delta = tmm_psi_delta([n_aire, n_capa, n_si], [D_CAPA], THETA0, LAMBDA)
    tan_psi = np.tan(psi * np.pi / 180)
    cos_delta = np.cos(delta * np.pi / 180)
    return tan_psi, cos_delta

errores_tan_psi = []
errores_cos_delta = []
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

        errores_tan_psi.append(np.mean(np.abs(tan_psi_real - tan_psi_p)))
        errores_cos_delta.append(np.mean(np.abs(cos_delta_real - cos_delta_p)))
    except:
        descartadas += 1

print(f"\nError espectral (test, {len(errores_tan_psi)}/{len(X_test_s)} muestras válidas):")
print(f"  MAE tan(Ψ):  {np.mean(errores_tan_psi):.6f}")
print(f"  MAE cos(Δ):  {np.mean(errores_cos_delta):.6f}")

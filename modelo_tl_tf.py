import tensorflow as tf
import numpy as np

"""
Modelo de Tauc-Lorentz y TMM implementados en TensorFlow nativo.

Versión diferenciable de las funciones de modelo_tl.py, necesaria para
que el gradiente fluya a través de la physics loss en la PINN.
Todas las operaciones usan tf.Tensor en lugar de NumPy.
"""

PI = tf.constant(np.pi, dtype=tf.float32)          # π
EPS = tf.constant(1e-8, dtype=tf.float32)           # epsilon numérico para evitar divisiones por cero

def eps_im_tf(E, A, E0, Eg, C):
    """
    ε₂(E) del Tauc-Lorentz, vectorizado sobre batch y energías.
    
    E:  tensor (N_E,)        - malla de energías
    A:  tensor (batch,)      - amplitud
    E0: tensor (batch,)      - energía de resonancia
    Eg: tensor (batch,)      - gap
    C:  tensor (batch,)      - ensanchamiento
    
    Retorna: tensor (batch, N_E)
    """
    # Broadcasting: E → (1, N_E), parámetros → (batch, 1)
    E = E[tf.newaxis, :]
    A = A[:, tf.newaxis]
    E0 = E0[:, tf.newaxis]
    Eg = Eg[:, tf.newaxis]
    C = C[:, tf.newaxis]
    
    numerador = A * E0 * C * (E - Eg)**2
    denominador = ((E**2 - E0**2)**2 + (C * E)**2) * E + EPS
    
    eps2 = numerador / denominador
    eps2 = tf.where(E > Eg, eps2, tf.zeros_like(eps2))
    
    return eps2

def eps_re_tf(E, A, E0, Eg, C, eps_inf):
    """
    ε₁(E) del Tauc-Lorentz (forma analítica), vectorizado.
    
    E:       tensor (N_E,)
    A,E0,Eg,C,eps_inf: tensores (batch,)
    
    Retorna: tensor (batch, N_E)
    """
    # Broadcasting
    E = E[tf.newaxis, :]
    A = A[:, tf.newaxis]
    E0 = E0[:, tf.newaxis]
    Eg = Eg[:, tf.newaxis]
    C = C[:, tf.newaxis]
    eps_inf = eps_inf[:, tf.newaxis]
    
    # Parámetros auxiliares (con clamp para estabilidad numérica)
    alpha = tf.sqrt(tf.maximum(4.0 * E0**2 - C**2, EPS))
    gamma = tf.sqrt(tf.maximum(E0**2 - 0.5 * C**2, EPS))
    zeta_4 = (E**2 - gamma**2)**2 + (alpha * C / 2.0)**2 + EPS
    
    a_ln = (Eg**2 - E0**2) * E**2 + Eg**2 * C**2 - E0**2 * (E0**2 + 3.0 * Eg**2)
    a_atan = (E**2 - E0**2) * (E0**2 + Eg**2) + Eg**2 * C**2
    
    # term1
    log_arg1 = (E0**2 + Eg**2 + alpha * Eg) / (E0**2 + Eg**2 - alpha * Eg + EPS)
    term1 = A * C * a_ln / (2.0 * PI * zeta_4 * E0 * alpha) * tf.math.log(tf.abs(log_arg1) + EPS)
    
    # term2
    term2 = A * a_atan / (PI * zeta_4 * E0) * (
        PI - tf.math.atan((2.0 * Eg + alpha) / C) + tf.math.atan((alpha - 2.0 * Eg) / C)
    )
    
    # term3
    term3 = 4.0 * A * E0 * Eg * (E**2 - gamma**2) / (PI * zeta_4 * alpha) * (
        tf.math.atan((alpha + 2.0 * Eg) / C) + tf.math.atan((alpha - 2.0 * Eg) / C)
    )
    
    # term4
    log_arg4 = tf.abs((E - Eg) / (E + Eg + EPS)) + EPS
    term4 = A * E0 * C * (E**2 + Eg**2) / (PI * zeta_4 * E) * tf.math.log(log_arg4)
    
    # term5
    log_arg5 = tf.abs((E - Eg) * (E + Eg)) / (tf.sqrt((E0**2 - Eg**2)**2 + C**2 * Eg**2) + EPS) + EPS
    term5 = 2.0 * A * E0 * C * Eg / (PI * zeta_4) * tf.math.log(log_arg5)
    
    eps1 = eps_inf + term1 - term2 + term3 - term4 + term5
    
    return eps1

def indice_refraccion_tf(eps1, eps2):
    """
    n + ik a partir de ε₁ y ε₂.
    eps1, eps2: tensores (batch, N_E)
    Retorna: n, k tensores (batch, N_E)
    """
    modulo = tf.sqrt(eps1**2 + eps2**2 + EPS)
    n = tf.sqrt(tf.maximum((eps1 + modulo) / 2.0, EPS))
    k = tf.sqrt(tf.maximum((-eps1 + modulo) / 2.0, EPS))
    return n, k



def tmm_tf(lista_n, lista_d, theta0, lambda_, polarization='s'):
    """
    TMM en TensorFlow, vectorizado sobre batch y longitudes de onda.
    
    Parámetros:
        lista_n  : list de tensores complejos. Cada elemento tiene shape:
                   - (N_lambda,) si el índice no depende del batch (aire, Si)
                   - (batch, N_lambda) si depende del batch (capa TL predicha)
        lista_d  : list de tensores reales. Cada elemento tiene shape:
                   - (batch,) — espesor predicho por la red, en nm
        theta0   : float - ángulo de incidencia en radianes
        lambda_  : tensor (N_lambda,) - longitudes de onda en nm
        polarization : 's' o 'p'
    
    Retorna:
        r : tensor complejo (batch, N_lambda) - coeficiente de reflexión
    """
    n_capas = len(lista_n)
    
    # --- Homogeneizar shapes a (batch, N_lambda) ---
    # Detectamos el batch_size mirando la capa TL (la única con dim de batch)
    # Asumimos que al menos una capa tiene shape (batch, N_lambda)
    batch_size = None
    for n in lista_n:
        if len(n.shape) == 2:
            batch_size = tf.shape(n)[0]
            break
    
    # Promovemos las capas que sean (N_lambda,) a (batch, N_lambda)
    lista_n_b = []
    for n in lista_n:
        if len(n.shape) == 1:
            n = tf.broadcast_to(n[tf.newaxis, :], [batch_size, tf.shape(n)[0]])
        lista_n_b.append(n)
    
    # λ → (1, N_lambda) para broadcasting con batch
    lambda_b = tf.cast(lambda_[tf.newaxis, :], tf.complex64)
    
    # sin(θ₀) como complejo
    sin_theta0 = tf.complex(tf.constant(np.sin(theta0), dtype=tf.float32), 
                            tf.constant(0.0, dtype=tf.float32))
    
    # --- Ley de Snell para cada capa: cos(θ_j) ---
    n0 = lista_n_b[0]  # índice del medio incidente (aire)
    cos_theta_list = []
    for n in lista_n_b:
        sin_th = (n0 * sin_theta0) / n
        cos_th = tf.sqrt(tf.complex(1.0, 0.0) - sin_th**2)
        cos_theta_list.append(cos_th)
    
    # --- Matriz total: empezar con identidad (batch, N_lambda, 2, 2) ---
    eye = tf.eye(2, dtype=tf.complex64)
    M_total = tf.broadcast_to(eye, [batch_size, tf.shape(lambda_)[0], 2, 2])
    
    # --- Bucle sobre interfaces ---
    for i in range(n_capas - 1):
        n_i = lista_n_b[i]
        n_j = lista_n_b[i + 1]
        cos_i = cos_theta_list[i]
        cos_j = cos_theta_list[i + 1]
        
        if polarization == 's':
            r_ij = (n_i * cos_i - n_j * cos_j) / (n_i * cos_i + n_j * cos_j)
            t_ij = (2.0 * n_i * cos_i) / (n_i * cos_i + n_j * cos_j)
        else:
            r_ij = (n_j * cos_i - n_i * cos_j) / (n_j * cos_i + n_i * cos_j)
            t_ij = (2.0 * n_i * cos_i) / (n_j * cos_i + n_i * cos_j)
        
        # Construir F_ij con shape (batch, N_lambda, 2, 2)
        ones = tf.ones_like(r_ij)
        F_ij = tf.stack([
            tf.stack([ones, r_ij], axis=-1),
            tf.stack([r_ij, ones], axis=-1)
        ], axis=-2) / t_ij[..., tf.newaxis, tf.newaxis]
        
        M_total = tf.linalg.matmul(M_total, F_ij)
        
        # Si hay capa intermedia tras la interfaz, multiplicar por P_j
        if i < n_capas - 2:
            d_j = tf.cast(lista_d[i], tf.complex64)[:, tf.newaxis]  # (batch, 1)
            beta = 2.0 * tf.cast(PI, tf.complex64) * n_j * d_j * cos_j / lambda_b
            
            exp_neg = tf.exp(-1j * beta)
            exp_pos = tf.exp(1j * beta)
            zeros = tf.zeros_like(beta)
            
            P_j = tf.stack([
                tf.stack([exp_neg, zeros], axis=-1),
                tf.stack([zeros, exp_pos], axis=-1)
            ], axis=-2)
            
            M_total = tf.linalg.matmul(M_total, P_j)
    
    # r = M[1,0] / M[0,0]
    r = M_total[..., 1, 0] / M_total[..., 0, 0]
    return r


def tmm_psi_delta_tf(n_list, d_list, theta0, lambda_):
    """
    Calcula tan(Ψ) y cos(Δ) a partir del TMM en TF.
    Devuelve directamente las cantidades que usa el dataset (no grados).
    
    Retorna:
        tan_psi   : tensor real (batch, N_lambda)
        cos_delta : tensor real (batch, N_lambda)
    """
    rs = tmm_tf(n_list, d_list, theta0, lambda_, 's')
    rp = tmm_tf(n_list, d_list, theta0, lambda_, 'p')
    
    # ρ = rp / rs, con protección contra rs ≈ 0
    rs_safe = tf.where(tf.abs(rs) > 1e-12, rs, tf.complex(1e-12, 0.0))
    rho = rp / rs_safe
    
    # |ρ| y arg(ρ) en radianes
    abs_rho = tf.abs(rho)              # ya es real
    arg_rho = tf.math.angle(rho)       # ya es real, en radianes
    
    # Ψ = arctan(|ρ|)  →  tan(Ψ) = |ρ|  directamente
    tan_psi = abs_rho
    
    # Δ = arg(ρ)  →  cos(Δ) = cos(arg(ρ))
    cos_delta = tf.cos(arg_rho)
    
    return tan_psi, cos_delta
    

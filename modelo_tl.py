import numpy as np


# =============================================================================
# Modelo dieléctrico de Tauc-Lorentz
# =============================================================================

def eps_im(E, A, E_0, E_g, C):
    """
    Calcula la parte imaginaria de la función dieléctrica (ε₂) 
    usando el modelo de Tauc-Lorentz.
    
    ε₂(E) = (A · E₀ · C · (E - Eg)²) / (((E² - E₀²)² + (C·E)²) · E)  para E > Eg
    ε₂(E) = 0 para E ≤ Eg
    
    Parámetros:
        E     : array - Energías en eV
        A     : float - Amplitud del oscilador (eV)
        E_0   : float - Energía de resonancia del oscilador (eV)
        E_g   : float - Gap óptico (eV). Debe ser E_g < E_0
        C     : float - Ensanchamiento del oscilador (eV)
    
    Retorna:
        eps2  : array - Parte imaginaria de ε para cada energía
    """
    E = np.array(E)
    eps2 = np.zeros_like(E, dtype=float)
    
    for i, Ei in enumerate(E):
        if Ei > E_g:
            numerador = A * E_0 * C * (Ei - E_g)**2
            denominador = ((Ei**2 - E_0**2)**2 + (C * Ei)**2) * Ei
            eps2[i] = numerador / denominador
        else:
            eps2[i] = 0.0
    
    return eps2


def eps_re(E, A, E_0, E_g, C, eps_inf):
    """
    Calcula la parte real de la función dieléctrica (ε₁) usando el modelo 
    de Tauc-Lorentz mediante la solución analítica de la integral de 
    Kramers-Kronig.

    ε₁(E) = ε∞ + (2/π) · P ∫ (ξ · ε₂(ξ)) / (ξ² - E²) dξ

    La solución analítica se descompone en 5 términos que involucran
    logaritmos y arcotangentes, con los parámetros auxiliares:
        α = √(4E₀² - C²)
        γ = √(E₀² - C²/2)
        ζ⁴ = (E² - γ²)² + (αC/2)²

    Parámetros:
        E       : array - Energías en eV
        A       : float - Amplitud del oscilador (eV)
        E_0     : float - Energía de resonancia del oscilador (eV)
        E_g     : float - Gap óptico (eV). Debe ser E_g < E_0
        C       : float - Ensanchamiento del oscilador (eV)
        eps_inf : float - Contribución dieléctrica a alta frecuencia (adimensional)

    Retorna:
        eps1    : array - Parte real de ε para cada energía

    Nota:
        Requiere 4E₀² > C² y E₀² > C²/2 para que α y γ sean reales.
        Si no se cumple, se fuerzan a 0 con max(..., 0).
    """
    E = np.array(E)
    eps1 = np.full_like(E, eps_inf, dtype=float)

    for i, Ei in enumerate(E):
        alpha = np.sqrt(max(4 * E_0**2 - C**2, 0))
        gamma = np.sqrt(max(E_0**2 - 0.5 * C**2, 0))
        zeta_4 = (Ei**2 - gamma**2)**2 + (alpha * C / 2)**2

        a_ln = (E_g**2 - E_0**2) * Ei**2 + E_g**2 * C**2 - E_0**2 * (E_0**2 + 3 * E_g**2)
        a_atan = (Ei**2 - E_0**2) * (E_0**2 + E_g**2) + E_g**2 * C**2

        term1 = A * C * a_ln / (2 * np.pi * zeta_4 * E_0 * alpha) * np.log((E_0**2 + E_g**2 + alpha * E_g) / (E_0**2 + E_g**2 - alpha * E_g))
        term2 = A * a_atan / (np.pi * zeta_4 * E_0) * (np.pi - np.arctan((2 * E_g + alpha) / C) + np.arctan((alpha - 2 * E_g) / C))
        term3 = 4 * A * E_0 * E_g * (Ei**2 - gamma**2) / (np.pi * zeta_4 * alpha) * (np.arctan((alpha + 2 * E_g) / C) + np.arctan((alpha - 2 * E_g) / C))
        term4 = A * E_0 * C * (Ei**2 + E_g**2) / (np.pi * zeta_4 * Ei) * np.log(abs((Ei - E_g) / (Ei + E_g)))
        term5 = 2 * A * E_0 * C * E_g / (np.pi * zeta_4) * np.log(abs((Ei - E_g) * (Ei + E_g) / np.sqrt((E_0**2 - E_g**2)**2 + C**2 * E_g**2)))

        eps1[i] = eps_inf + term1 - term2 + term3 - term4 + term5

    return eps1


def indice_refraccion(eps1, eps2):
    """
    Calcula el índice de refracción complejo (n, k) a partir de las partes
    real e imaginaria de la función dieléctrica.

    n = √[(ε₁ + √(ε₁² + ε₂²)) / 2]
    k = √[(-ε₁ + √(ε₁² + ε₂²)) / 2]

    Donde n es el índice de refracción y k es el coeficiente de extinción.
    El índice complejo se expresa como: ñ = n + ik

    Parámetros:
        eps1 : array - Parte real de la función dieléctrica (ε₁)
        eps2 : array - Parte imaginaria de la función dieléctrica (ε₂)

    Retorna:
        n    : array - Índice de refracción
        k    : array - Coeficiente de extinción
    """
    n = np.sqrt((eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    k = np.sqrt((-eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    return n, k


def eps_multi(E, osciladores, eps_inf):
    """
    Calcula ε₁ y ε₂ totales para un material con n osciladores Tauc-Lorentz
    que comparten un mismo eps_inf.
    
    ε₂_total(E) = Σ ε₂_i(E)
    ε₁_total(E) = eps_inf + Σ [ε₁_i(E) - eps_inf_i]   (con eps_inf_i = 0 al llamar)
    
    Parámetros:
        E           : array - Energías en eV
        osciladores : list of tuples - [(A, E_0, E_g, C), ...] para cada oscilador
        eps_inf     : float - Contribución dieléctrica a alta frecuencia (global)
    
    Retorna:
        eps1, eps2 : arrays - Partes real e imaginaria totales
    """
    E = np.array(E)
    eps1 = np.full_like(E, eps_inf, dtype=float)
    eps2 = np.zeros_like(E, dtype=float)
    
    for (A, E_0, E_g, C) in osciladores:
        eps2 += eps_im(E, A, E_0, E_g, C)
        eps1 += eps_re(E, A, E_0, E_g, C, eps_inf=0.0)
    
    return eps1, eps2


# =============================================================================
# Método de la Matriz de Transferencia (TMM)
# =============================================================================

def tmm(lista_n, lista_d, theta0, lambda_, polarization='s'):
    """
    Método de la Matriz de Transferencia (TMM) para calcular el coeficiente
    de reflexión de un sistema multicapa con índices dependientes de λ.

    Para cada longitud de onda:
    1. Aplica la ley de Snell generalizada en cada capa
    2. Construye la matriz de interfaz (Fresnel) entre capas adyacentes:
       F_ij = (1/t_ij) · [[1, r_ij], [r_ij, 1]]
    3. Construye la matriz de propagación dentro de cada capa:
       P_j = [[exp(-iβ), 0], [0, exp(iβ)]]   con β = 2π·n_j·d_j·cos(θ_j)/λ
    4. Multiplica todas las matrices: M = F_01 · P_1 · F_12 · P_2 · ... · F_(N-1,N)
    5. Extrae el coeficiente de reflexión: r = M₁₀ / M₀₀

    Parámetros:
        lista_n      : list of arrays - Índice complejo de cada capa para cada λ.
                       [n_aire, n_capa1, ..., n_sustrato]. Cada elemento es un
                       array de longitud len(lambda_).
        lista_d      : list of float - Espesores de las capas intermedias en nm.
                       No incluye medio ambiente ni sustrato. len(lista_d) = len(lista_n) - 2
        theta0       : float - Ángulo de incidencia en radianes
        lambda_      : array - Longitudes de onda en nm
        polarization : str - Polarización: 's' (TE) o 'p' (TM). Por defecto 's'.

    Retorna:
        r            : array complex - Coeficiente de reflexión para cada λ
    """
    n_capas = len(lista_n)
    n_wavelengths = len(lambda_)
    sin_theta0 = np.sin(theta0)

    r_list = []

    for wl_idx in range(n_wavelengths):
        n_at_wl = [lista_n[j][wl_idx] for j in range(n_capas)]

        cos_theta_list = []
        for n in n_at_wl:
            sin_th = (n_at_wl[0] * sin_theta0) / n
            cos_th = np.sqrt(1 - sin_th**2 + 0j)
            cos_theta_list.append(cos_th)

        M_total = np.array([[1, 0], [0, 1]], dtype=complex)

        for i in range(n_capas - 1):
            cos_theta_i = cos_theta_list[i]
            cos_theta_j = cos_theta_list[i + 1]
            n_i = n_at_wl[i]
            n_j = n_at_wl[i + 1]

            if polarization == 's':
                r_ij = (n_i * cos_theta_i - n_j * cos_theta_j) / (n_i * cos_theta_i + n_j * cos_theta_j)
                t_ij = (2 * n_i * cos_theta_i) / (n_i * cos_theta_i + n_j * cos_theta_j)
            else:
                r_ij = (n_j * cos_theta_i - n_i * cos_theta_j) / (n_j * cos_theta_i + n_i * cos_theta_j)
                t_ij = (2 * n_i * cos_theta_i) / (n_j * cos_theta_i + n_i * cos_theta_j)

            F_ij = (1 / t_ij) * np.array([[1, r_ij], [r_ij, 1]], dtype=complex)

            if i < n_capas - 2:
                beta = 2 * np.pi * n_j * lista_d[i] * cos_theta_j / lambda_[wl_idx]
                P_j = np.array([[np.exp(-1j * beta), 0], [0, np.exp(1j * beta)]], dtype=complex)
                M_total = M_total @ F_ij @ P_j
            else:
                M_total = M_total @ F_ij

        r_list.append(M_total[1, 0] / M_total[0, 0])

    return np.array(r_list)


def tmm_psi_delta(n_list, d_list, theta0, wavelength):
    """
    Calcula los ángulos elipsométricos Ψ y Δ a partir del TMM.

    Ejecuta el TMM para ambas polarizaciones (s y p) y obtiene Ψ y Δ
    a partir de la relación fundamental de la elipsometría:

        ρ = rp / rs = tan(Ψ) · exp(iΔ)

    Donde:
        Ψ = arctan(|ρ|)
        Δ = arg(ρ)

    Parámetros:
        n_list     : list of arrays - Índice complejo de cada capa para cada λ.
                     [n_aire, n_capa1, ..., n_sustrato]
        d_list     : list of float - Espesores de las capas intermedias en nm
        theta0     : float - Ángulo de incidencia en radianes
        wavelength : array - Longitudes de onda en nm

    Retorna:
        psi        : array - Ángulo Ψ en grados
        delta      : array - Ángulo Δ en grados
    """
    rs = tmm(n_list, d_list, theta0, wavelength, 's')
    rp = tmm(n_list, d_list, theta0, wavelength, 'p')

    rho = np.where(rs != 0, rp / rs, 0)
    psi = np.arctan(np.abs(rho)) * 180 / np.pi
    delta = np.angle(rho) * 180 / np.pi

    return psi, delta


# =============================================================================
# Generación de parámetros aleatorios
# =============================================================================

def generar_parametros_tl():
    """
    Genera un conjunto aleatorio de parámetros para el modelo de Tauc-Lorentz
    con validaciones físicas.

    Rangos de muestreo:
        A       : [50, 200]   eV  - Amplitud del oscilador
        E_0     : [1, 5]      eV  - Energía de resonancia
        E_g     : [0.5, E_0]  eV  - Gap óptico (siempre E_g < E_0)
        C       : [0.5, 5]    eV  - Ensanchamiento
        eps_inf : [1, 3]           - Contribución dieléctrica a alta frecuencia

    Validaciones físicas (descarta combinaciones inválidas):
        - 4·E₀² > C²     → garantiza que α = √(4E₀² - C²) sea real
        - E₀ > C/√2      → garantiza que γ = √(E₀² - C²/2) sea real

    Retorna:
        A, E_0, E_g, C, eps_inf : floats - Parámetros TL válidos
    """
    while True:
        A = np.random.uniform(50, 200)
        E_0 = np.random.uniform(1, 5)
        E_g = np.random.uniform(0.5, E_0)
        C = np.random.uniform(0.5, 5)
        eps_inf = np.random.uniform(1, 3)

        if 4 * E_0**2 <= C**2:
            continue
        if E_0 <= C / np.sqrt(2):
            continue
        break

    return A, E_0, E_g, C, eps_inf


def generar_parametros_tl_espesor(d_min=20.0, d_max=300.0):
    """
    Genera un conjunto aleatorio de parámetros para el modelo de Tauc-Lorentz
    junto con el espesor de la capa, aplicando validaciones físicas.

    Rangos de muestreo:
        A       : [50, 200]    eV  - Amplitud del oscilador
        E_0     : [1, 5]       eV  - Energía de resonancia
        E_g     : [0.5, E_0]   eV  - Gap óptico (siempre E_g < E_0)
        C       : [0.5, 5]     eV  - Ensanchamiento
        eps_inf : [1, 3]            - Contribución dieléctrica a alta frecuencia
        d       : [d_min, d_max] nm - Espesor de la capa dieléctrica

    Validaciones físicas (descarta combinaciones inválidas):
        - 4·E₀² > C²     → garantiza que α = √(4E₀² - C²) sea real
        - E₀ > C/√2      → garantiza que γ = √(E₀² - C²/2) sea real

    Parámetros:
        d_min : float - Espesor mínimo en nm (por defecto 20)
        d_max : float - Espesor máximo en nm (por defecto 300)

    Retorna:
        A, E_0, E_g, C, eps_inf, d : floats - Parámetros TL válidos y espesor
    """
    while True:
        A = np.random.uniform(50, 200)
        E_0 = np.random.uniform(1, 5)
        E_g = np.random.uniform(0.5, E_0)
        C = np.random.uniform(0.5, 5)
        eps_inf = np.random.uniform(1, 3)

        if 4 * E_0**2 <= C**2:
            continue
        if E_0 <= C / np.sqrt(2):
            continue
        break

    d = np.random.uniform(d_min, d_max)

    return A, E_0, E_g, C, eps_inf, d


def generar_parametros_tl_multi(n_osc, d_min=20.0, d_max=300.0):
    """
    Genera parámetros TL para n_osc osciladores (1, 2 o 3) más espesor.
    
    Rangos por oscilador:
        A       : [50, 200]    eV
        E_0     : [1, 5]       eV
        E_g     : [0.5, E_0]   eV
        C       : [0.5, 5]     eV
    Globales:
        eps_inf : [1, 3]
        d       : [d_min, d_max] nm
    
    Retorna:
        osciladores : lista de n_osc tuplas (A, E_0, E_g, C)
        eps_inf, d  : floats
    """
    osciladores = []
    for _ in range(n_osc):
        while True:
            A = np.random.uniform(50, 200)
            E_0 = np.random.uniform(1, 5)
            E_g = np.random.uniform(0.5, E_0)
            C = np.random.uniform(0.5, 5)
            
            if 4 * E_0**2 <= C**2:
                continue
            if E_0 <= C / np.sqrt(2):
                continue
            break
        osciladores.append((A, E_0, E_g, C))
    
    eps_inf = np.random.uniform(1, 3)
    d = np.random.uniform(d_min, d_max)
    
    return osciladores, eps_inf, d

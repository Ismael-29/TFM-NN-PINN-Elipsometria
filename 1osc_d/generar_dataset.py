"""
Generación del dataset sintético para el caso de 1 oscilador Tauc-Lorentz
con espesor variable (d ∈ [20, 300] nm).

Genera 40.000 muestras de espectros (tan(Ψ), cos(Δ)) mediante TMM
para una estructura aire / capa TL / Si, con parámetros TL y espesor aleatorios.

A diferencia del caso 1osc, aquí el espesor d es un parámetro más a recuperar,
lo que añade una sexta salida al dataset.
"""

import numpy as np
import pandas as pd
from modelo_tl import (eps_im, eps_re, indice_refraccion,
                        tmm_psi_delta, generar_parametros_tl_espesor)

# --- Configuración ---
MUESTRAS = 40000
THETA0 = 70 * np.pi / 180          # ángulo de incidencia
E = np.linspace(0.75, 3.65, 101)   # malla de energías (eV)
LAMBDA = 1239.8 / E                # conversión eV → nm

# --- Constantes ópticas del silicio (sustrato) ---
# Se leen de un archivo tabulado y se interpolan a la malla de energías E
data_si = pd.read_csv('datos/Si.clc',
    skiprows=64, sep=r'\s+', header=None, encoding='latin-1', nrows=305)

E_si = data_si.iloc[:, 3].values       # energías tabuladas
n_si_tab = data_si.iloc[:, 8].values   # índice de refracción n
k_si_tab = data_si.iloc[:, 9].values   # coeficiente de extinción k

# Índice complejo del Si interpolado: ñ = n + ik
n_si = np.interp(E, E_si, n_si_tab) + 1j * np.interp(E, E_si, k_si_tab)
n_aire = np.ones(len(E)) * (1.0 + 0j)  # índice del aire (n=1, k=0)

# --- Generación de datos ---
# Para cada muestra:
#   1. Genera parámetros TL aleatorios (A, E₀, Eg, C, ε∞) y espesor d
#   2. Calcula la función dieléctrica (ε₁, ε₂) con el modelo Tauc-Lorentz
#   3. Obtiene el índice de refracción complejo de la capa
#   4. Simula los espectros elipsométricos (Ψ, Δ) con TMM
#      para la estructura: aire / capa TL (d variable) / Si
#   5. Almacena los 6 parámetros y los espectros como tan(Ψ) y cos(Δ)

resultados = []

for i in range(MUESTRAS):
    A, E_0, E_g, C, eps_inf, d_capa = generar_parametros_tl_espesor()

    eps2 = eps_im(E, A, E_0, E_g, C)
    eps1 = eps_re(E, A, E_0, E_g, C, eps_inf)
    n, k = indice_refraccion(eps1, eps2)
    n_capa = n + 1j * k

    psi, delta = tmm_psi_delta([n_aire, n_capa, n_si], [d_capa], THETA0, LAMBDA)

    tan_psi = np.tan(psi * np.pi / 180)
    cos_delta = np.cos(delta * np.pi / 180)

    resultados.append([A, E_0, E_g, C, eps_inf, d_capa] + list(tan_psi) + list(cos_delta))

# --- Crear DataFrame y guardar ---
# Columnas: 6 parámetros + 101 tan(Ψ) + 101 cos(Δ) = 208 columnas por muestra
col_params = ['A', 'E_0', 'E_g', 'C', 'eps_inf', 'd']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

df = pd.DataFrame(resultados, columns=col_params + col_tan + col_cos)
df.to_csv('1osc_d/dataset_1osc_d.csv', index=False)

print(f"Dataset generado: {len(df)} muestras")

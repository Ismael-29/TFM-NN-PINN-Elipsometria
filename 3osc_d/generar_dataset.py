"""
Generación del dataset sintético multioscilador Tauc-Lorentz
con espesor variable.

Genera 20.000 muestras por clase (1, 2 y 3 osciladores) = 60.000 total.
Cada muestra contiene:
    - Clase (n_osc): 1, 2 o 3
    - 12 parámetros de oscilador (4 × 3, rellenados con 0 si inactivos)
    - ε∞, d (espesor variable entre 20 y 300 nm)
    - 202 valores espectrales (101 tan(Ψ) + 101 cos(Δ))

Los osciladores se ordenan por E₀ creciente para reducir la degeneración
por permutación.
"""

import numpy as np
import pandas as pd

from modelo_tl import (generar_parametros_tl_multi, eps_multi,
                       indice_refraccion, tmm_psi_delta)

# =============================================================================
# Configuración
# =============================================================================

muestras_por_clase = 20000
n_osc_lista = [1, 2, 3]
theta0 = 70 * np.pi / 180
E = np.linspace(0.75, 3.65, 101)
lambda_ = 1239.8 / E

# --- Constantes ópticas del silicio (sustrato) ---
data_si = pd.read_csv('datos/Si.clc',
    skiprows=64, sep=r'\s+', header=None, encoding='latin-1', nrows=305)

E_si = data_si.iloc[:, 3].values
n_si_tab = data_si.iloc[:, 8].values
k_si_tab = data_si.iloc[:, 9].values

n_si_complex = np.interp(E, E_si, n_si_tab) + 1j * np.interp(E, E_si, k_si_tab)
n_aire = np.ones(len(E)) * (1.0 + 0j)

# =============================================================================
# Generar muestras
# =============================================================================

resultados = []

for n_osc in n_osc_lista:
    print(f"Generando {muestras_por_clase} muestras con {n_osc} oscilador(es)...")
    for _ in range(muestras_por_clase):
        osciladores, eps_inf, d_capa = generar_parametros_tl_multi(n_osc)
        osciladores.sort(key=lambda x: x[1])

        eps1, eps2 = eps_multi(E, osciladores, eps_inf)
        n, k = indice_refraccion(eps1, eps2)
        n_capa = n + 1j * k

        lista_n = [n_aire, n_capa, n_si_complex]
        psi, delta = tmm_psi_delta(lista_n, [d_capa], theta0, lambda_)

        tan_psi = np.tan(psi * np.pi / 180)
        cos_delta = np.cos(delta * np.pi / 180)

        params = []
        for i in range(3):
            if i < n_osc:
                A, E_0, E_g, C = osciladores[i]
                params.extend([A, E_0, E_g, C])
            else:
                params.extend([0.0, 0.0, 0.0, 0.0])

        resultados.append([n_osc] + params + [eps_inf, d_capa]
                          + list(tan_psi) + list(cos_delta))

# =============================================================================
# Guardar dataset
# =============================================================================

col_params = ['n_osc',
              'A1', 'E0_1', 'Eg_1', 'C1',
              'A2', 'E0_2', 'Eg_2', 'C2',
              'A3', 'E0_3', 'Eg_3', 'C3',
              'eps_inf', 'd']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

df = pd.DataFrame(resultados, columns=col_params + col_tan + col_cos)
df.to_csv('3osc_d/dataset_3osc_d.csv', index=False)

print(f"\nDataset generado: {df.shape}")
print(df['n_osc'].value_counts().sort_index())

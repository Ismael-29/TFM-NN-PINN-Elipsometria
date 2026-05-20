"""
Generación del dataset sintético para el caso multioscilador Tauc-Lorentz
con espesor fijo (d = 100 nm).

Genera 36.000 muestras (12.000 por clase) de espectros (tan(Ψ), cos(Δ))
mediante TMM para una estructura aire / capa TL / Si, con 1, 2 o 3
osciladores Tauc-Lorentz.

Los osciladores se ordenan por E₀ creciente para romper la degeneración
por permutación. Los osciladores inactivos se rellenan con ceros.

Columnas del dataset:
    - n_osc: número de osciladores (1, 2 o 3)
    - A1, E0_1, Eg_1, C1, ..., A3, E0_3, Eg_3, C3: parámetros por oscilador
    - eps_inf: contribución dieléctrica a alta frecuencia (global)
    - tanPsi_0 ... tanPsi_100: 101 valores de tan(Ψ)
    - cosDelta_0 ... cosDelta_100: 101 valores de cos(Δ)
"""

import numpy as np
import pandas as pd
from modelo_tl import (eps_multi, indice_refraccion, tmm_psi_delta,
                        generar_parametros_tl_multi)

# =============================================================================
# Configuración
# =============================================================================

MUESTRAS_POR_CLASE = 12000          # 12.000 × 3 clases = 36.000 muestras
N_OSC_LISTA = [1, 2, 3]            # clases de osciladores
THETA0 = 70 * np.pi / 180          # ángulo de incidencia
D_CAPA = 100.0                     # espesor fijo en nm
E = np.linspace(0.75, 3.65, 101)   # malla de energías (eV)
LAMBDA = 1239.8 / E                # conversión eV → nm

# =============================================================================
# Constantes ópticas del silicio (sustrato)
# =============================================================================

data_si = pd.read_csv('datos/Si.clc',
    skiprows=64, sep=r'\s+', header=None, encoding='latin-1', nrows=305)

E_si = data_si.iloc[:, 3].values       # energías tabuladas
n_si_tab = data_si.iloc[:, 8].values   # índice de refracción n
k_si_tab = data_si.iloc[:, 9].values   # coeficiente de extinción k

# Índice complejo del Si interpolado: ñ = n + ik
n_si = np.interp(E, E_si, n_si_tab) + 1j * np.interp(E, E_si, k_si_tab)
n_aire = np.ones(len(E)) * (1.0 + 0j)  # índice del aire (n=1, k=0)

# =============================================================================
# Generación de datos
# =============================================================================
# Para cada clase (1, 2 o 3 osciladores) y cada muestra:
#   1. Genera parámetros TL aleatorios para n_osc osciladores + eps_inf
#   2. Ordena los osciladores por E₀ creciente (rompe degeneración por permutación)
#   3. Calcula ε₁, ε₂ totales sumando las contribuciones de cada oscilador
#   4. Obtiene el índice de refracción complejo de la capa
#   5. Simula los espectros (Ψ, Δ) con TMM: aire / capa TL (100 nm) / Si
#   6. Rellena con ceros los parámetros de osciladores inactivos (hasta 3)

resultados = []

for n_osc in N_OSC_LISTA:
    print(f"Generando {MUESTRAS_POR_CLASE} muestras con {n_osc} oscilador(es)...")
    for _ in range(MUESTRAS_POR_CLASE):
        osciladores, eps_inf, _ = generar_parametros_tl_multi(n_osc)
        osciladores.sort(key=lambda x: x[1])  # ordenar por E₀ creciente

        eps1, eps2 = eps_multi(E, osciladores, eps_inf)
        n, k = indice_refraccion(eps1, eps2)
        n_capa = n + 1j * k

        psi, delta = tmm_psi_delta([n_aire, n_capa, n_si], [D_CAPA], THETA0, LAMBDA)

        tan_psi = np.tan(psi * np.pi / 180)
        cos_delta = np.cos(delta * np.pi / 180)

        # Empaquetar parámetros: 4 por oscilador, rellenando con 0 los inactivos
        params = []
        for i in range(3):
            if i < n_osc:
                A, E_0, E_g, C = osciladores[i]
                params.extend([A, E_0, E_g, C])
            else:
                params.extend([0.0, 0.0, 0.0, 0.0])

        resultados.append([n_osc] + params + [eps_inf] + list(tan_psi) + list(cos_delta))

# =============================================================================
# Crear DataFrame y guardar
# =============================================================================
# 1 (n_osc) + 12 (3×4 parámetros) + 1 (eps_inf) + 101 (tan Ψ) + 101 (cos Δ) = 216 columnas

col_params = ['n_osc',
              'A1', 'E0_1', 'Eg_1', 'C1',
              'A2', 'E0_2', 'Eg_2', 'C2',
              'A3', 'E0_3', 'Eg_3', 'C3',
              'eps_inf']
col_tan = [f'tanPsi_{i}' for i in range(101)]
col_cos = [f'cosDelta_{i}' for i in range(101)]

df = pd.DataFrame(resultados, columns=col_params + col_tan + col_cos)
df.to_csv('3osc/dataset_3osc.csv', index=False)

print(f"\nDataset generado: {df.shape}")
print(df['n_osc'].value_counts().sort_index())

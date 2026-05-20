# Elipsometría Espectroscópica con Redes Neuronales Informadas por la Física

Código del Trabajo Fin de Máster: *"Aplicación de Redes Neuronales Informadas para la Caracterización Multioscilador de Capas mediante Elipsometría Espectroscópica"*

## Descripción

Redes neuronales informadas por la física (PINNs) para la caracterización óptica de capas delgadas mediante elipsometría espectroscópica. Los modelos recuperan los parámetros del modelo Tauc-Lorentz a partir de espectros Ψ/Δ, soportando configuraciones de 1, 2 y 3 osciladores.

## Estructura del repositorio

- `modelo_tl.py` — Modelo Tauc-Lorentz (ε₁, ε₂) con integración de Kramers-Kronig
- `modelo_tl_tf.py` — Modelo directo TMM en TensorFlow nativo para la loss física de la PINN
- `utils.py` — Funciones auxiliares (escalado, métricas)
- `1osc/` — Un oscilador, espesor fijo
- `1osc_d/` — Un oscilador con espesor variable
- `3osc/` — Tres osciladores, espesor fijo
- `3osc_d/` — Tres osciladores con espesor variable

Cada carpeta contiene:
- `generar_dataset.py` — Generación de datos sintéticos (espectros Ψ, Δ mediante TMM)
- `entrenar_nn.py` — Entrenamiento de la red neuronal estándar 
- `entrenar_pinn.py` — Entrenamiento de la PINN 


## Requisitos

Python 3.11, TensorFlow, NumPy, SciPy

## Autor

Ismael Pardo Ortiz — Universidad de Cantabria, 2026

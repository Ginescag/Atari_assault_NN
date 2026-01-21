================================================================================
PROYECTO: AGENTE INTELEIGENTE PARA ATARI ASSAULT
AUTOR/ES: Ginés Caballero Guijarro 
================================================================================

1. DESCRIPCIÓN GENERAL
--------------------------------------------------------------------------------
Este proyecto implementa un agente autónomo capaz de aprender a jugar al videojuego
"Assault" de Atari 2600. El núcleo del sistema es una Red Neuronal Artificial (ANN)
de tipo Perceptrón Multicapa (MLP) desarrollada completamente en C++ estándar.

El agente interactúa con el entorno de emulación ALE (Arcade Learning Environment)
leyendo el estado de la memoria RAM de la consola y decidiendo la mejor acción
a realizar basándose en un modelo entrenado mediante aprendizaje supervisado.

2. TECNOLOGÍAS UTILIZADAS
--------------------------------------------------------------------------------
- Lenguaje: C++ (Estándar C++17 recomendado).
- Entorno de Emulación: Arcade Learning Environment (ALE 0.6.1).
- Gráficos/Input: SDL (Simple DirectMedia Layer) 1.2.
- Librería Matemática: Implementación propia de álgebra matricial (clase Matrix).
- Scripting Auxiliar: Python 3 (pandas, matplotlib, seaborn) para generación de
  datasets balanceados y visualización de heatmaps/métricas.

3. ARQUITECTURA DE LA RED NEURONAL (NeuralNetwork.h)
--------------------------------------------------------------------------------
El modelo ha sido construido con la siguiente Topología:

A. Topología Dinámica:
   - Entrada: 128 neuronas (correspondientes a los 128 bytes de la RAM de Atari).
   - Capas Ocultas: Configurable. (Configuración óptima probada: 128 -> 128 -> 64 -> 32).
   - Salida: 6 neuronas (Acciones: NOOP, FIRE_RIGHT, FIRE_LEFT, FIRE_UP, LEFT, RIGHT).

B. Parámetros usados:

    nn.train(trainInputs, trainTargets, valInputs, valTargets, 50, 0.02, "tanh", 20, true);
    - 50 (epochs)
    - 0.01 (learning rate)
    - "tahn" (funcion de activación)
    - 20  (paciencia)
    - true (modo VERBOSE activado)
    
    El dataset consta de 6500 entradas donde todas las posibles acciones tienen el mismo numero de entradas (1000) menos disparar hacia arriba que tiene más (1500)

    se ha realizado un split donde el 80% del dataset va al conjunto de entrenamiento (trainTargets y trainInputs) y el 20% va al conjunto de validación

C. Funciones de Activación:
   - Capas Ocultas: Tangente Hiperbólica (tanh) para mantener el rango [-1, 1].
   - Normalización: Los datos de entrada (RAM) se normalizan de [0, 255] a [-1, 1].

D. Regularización:
   - Early Stopping: El entrenamiento se detiene automáticamente si el error
     en el conjunto de validación no mejora tras un número definido de épocas ('patience'),
     evitando el sobreajuste (overfitting).

     se ha usado una paciencia de 20 epocas

4. FLUJO DE TRABAJO
--------------------------------------------------------------------------------
1. Recolección: Se juega manualmente para generar un dataset de pares {Estado RAM, Acción}.
2. Preprocesado: Se balancean los datos (Python) para evitar sesgos hacia la inacción.

1 y 2 estan hechos y el dataset se encuentra en dataset_balanced_boosted_fire.txt

3. Entrenamiento: La red carga el dataset, divide en Train/Validation y ajusta los pesos.
4. Resultado Final: El agente juega automáticamente calculando la acción con mayor activación.
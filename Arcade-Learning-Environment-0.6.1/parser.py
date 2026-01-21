import pandas as pd
import numpy as np

# Configuración
INPUT_FILE = 'dataset.txt'
OUTPUT_FILE = 'dataset_balanced_3000.txt'
SAMPLES_PER_CLASS = 3000
NOISE_SCALE = 0.05  # Ruido muy pequeño

def generate_balanced_dataset():
    print(f"Leyendo {INPUT_FILE}...")
    try:
        # Leemos el archivo asumiendo separación por espacios
        df = pd.read_csv(INPUT_FILE, sep='\s+', header=None, engine='python')
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        return

    # La última columna es la etiqueta (clase)
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    balanced_dfs = []
    
    # Obtenemos las clases únicas (0, 1, 2, 3, 4, 5)
    classes = sorted(df[label_col].unique())

    print(f"Procesando clases: {classes}")

    for c in classes:
        # Filtramos los datos de la clase actual
        df_class = df[df[label_col] == c]
        count = len(df_class)
        
        if count >= SAMPLES_PER_CLASS:
            # Si sobran datos (Clases 0, 4, 5), cogemos 3000 aleatorios
            print(f"Clase {c}: {count} muestras -> Recortando a {SAMPLES_PER_CLASS}")
            df_resampled = df_class.sample(n=SAMPLES_PER_CLASS, random_state=42)
            
        else:
            # Si faltan datos (Clases 1, 2, 3), hacemos Oversampling
            print(f"Clase {c}: {count} muestras -> Aumentando a {SAMPLES_PER_CLASS}")
            
            # 1. Mantenemos los originales
            originals = df_class.copy()
            
            # 2. Calculamos cuántos faltan
            n_needed = SAMPLES_PER_CLASS - count
            
            # 3. Muestreamos con reemplazo (duplicamos)
            generated = df_class.sample(n=n_needed, replace=True, random_state=42)
            
            # 4. Solo añadimos ruido a las clases 1 y 2 (según tus instrucciones)
            if c in [1, 2]:
                print(f"   -> Añadiendo ruido a clase {c}")
                noise = np.random.normal(loc=0.0, scale=NOISE_SCALE, size=generated[feature_cols].shape)
                generated[feature_cols] = generated[feature_cols] + noise
            
            # Concatenamos originales + generados
            df_resampled = pd.concat([originals, generated])

        balanced_dfs.append(df_resampled)

    # Unimos todo y hacemos Shuffle
    print("Mezclando datos (Shuffle)...")
    df_final = pd.concat(balanced_dfs)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Guardamos sin cabecera ni índice, separado por espacios (formato original)
    df_final.to_csv(OUTPUT_FILE, sep=' ', header=False, index=False)
    print(f"¡Hecho! Archivo guardado como: {OUTPUT_FILE}")
    print("\nRecuento final por clase:")
    print(df_final[label_col].value_counts().sort_index())

if __name__ == "__main__":
    generate_balanced_dataset()
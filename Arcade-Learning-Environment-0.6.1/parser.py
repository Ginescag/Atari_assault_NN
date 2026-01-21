import pandas as pd
import numpy as np

# ESTE SCRIPT DE PYTHON SIRVE PARA GENERAR UN DATASET BALANCEADO A PARTIR DE UNO QUE NO LO ESTA
# MODIFICADO: AHORA LA CLASE 3 TIENE MAS MUESTRAS PARA INCENTIVAR EL DISPARO

# Configuración
INPUT_FILE = 'dataset.txt'
OUTPUT_FILE = 'dataset_balanced_boosted_fire.txt'
SAMPLES_PER_CLASS = 1000
SAMPLES_CLASS_3 = int(SAMPLES_PER_CLASS * 1.5)
NOISE_INT_RANGE = 2      # El ruido variará aleatoriamente entre -2 y +2

def generate_balanced_dataset():
    print(f"Leyendo {INPUT_FILE}...")
    try:
        # GENERAMOS UN DATAFRAME DE PANDAS PARA PODER PARSEAR MEJOR LOS DATOS
        df = pd.read_csv(INPUT_FILE, sep='\s+', header=None, engine='python')
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        return

    # La última columna es la etiqueta (clase)
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    balanced_dfs = []
    
    # Obtenemos las clases únicas
    classes = sorted(df[label_col].unique())
    print(f"Clases detectadas: {classes}")

    for c in classes:
        # Filtramos los datos de la clase actual
        df_class = df[df[label_col] == c]
        count = len(df_class)
        
        # DEFINIMOS EL OBJETIVO DE MUESTRAS SEGÚN LA CLASE
        target_count = SAMPLES_CLASS_3 if c == 3 else SAMPLES_PER_CLASS

        if count >= target_count:
            # Si sobran datos, hacemos downsampling al objetivo correspondiente
            print(f"Clase {c}: {count} muestras -> Recortando a {target_count}")
            df_resampled = df_class.sample(n=target_count, random_state=42)
            
        else:
            # Si faltan datos, hacemos data augmentation
            print(f"Clase {c}: {count} muestras -> Aumentando a {target_count}")
            
            # 1. Copiamos los originales
            originals = df_class.copy()
            
            # 2. Calculamos cuántos faltan para llegar al objetivo específico
            n_needed = target_count - count
            
            # 3. Muestreamos con reemplazo para generar los nuevos
            generated = df_class.sample(n=n_needed, replace=True, random_state=42)
            
            # AHORA APLICAMOS RUIDO TAMBIÉN A LA CLASE 3 PARA EVITAR DUPLICADOS EXACTOS
            if c in [1, 2, 3]: 
                print(f"   -> Añadiendo ruido entero (±{NOISE_INT_RANGE}) a clase {c}")
                
                # Generamos ruido entero entre -NOISE_INT_RANGE y +NOISE_INT_RANGE
                noise = np.random.randint(
                    low=-NOISE_INT_RANGE, 
                    high=NOISE_INT_RANGE + 1, 
                    size=generated[feature_cols].shape
                )
                
                # Sumamos el ruido
                generated[feature_cols] = generated[feature_cols] + noise
                
                # CLIPPING: Forzamos a que los valores estén entre 0 y 255
                generated[feature_cols] = generated[feature_cols].clip(0, 255)
            
            # Concatenamos originales + generados
            df_resampled = pd.concat([originals, generated])

        balanced_dfs.append(df_resampled)

    # Unimos todo y hacemos Shuffle
    print("Mezclando datos (Shuffle)...")
    df_final = pd.concat(balanced_dfs)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Nos aseguramos de que todo sean enteros antes de guardar
    df_final = df_final.astype(int)

    # Guardamos sin cabecera ni índice, separado por espacios
    df_final.to_csv(OUTPUT_FILE, sep=' ', header=False, index=False)
    
    print(f"¡Hecho! Archivo guardado como: {OUTPUT_FILE}")
    print("\nRecuento final por clase:")
    print(df_final[label_col].value_counts().sort_index())

if __name__ == "__main__":
    generate_balanced_dataset()
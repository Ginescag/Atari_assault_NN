import pandas as pd
import numpy as np

#ESTE SCRIPT DE PYTHON SIRVE PARA GENERAR UN DATASET BALANCEADO A PARTIR DE UNO QUE NO LO ESTA
#AL HABER CIERTOS MOVIMIENTOS QUE SE REALIZAN MENOS QUE OTROS, REALIZO "DATA AUGMENTATION" GENERANDO COPIAS DE LAS ACCIONES QUE APARECEN MENOS CON UN RUIDO INTRODUCIDO

# Configuración
INPUT_FILE = 'dataset.txt'
OUTPUT_FILE = 'dataset_balanced_1000.txt'
SAMPLES_PER_CLASS = 1000
NOISE_INT_RANGE = 2  # El ruido variará aleatoriamente entre -2 y +2

def generate_balanced_dataset():
    print(f"Leyendo {INPUT_FILE}...")
    try:
        #GENERAMOS UN DATAFRAME DE PANDAS PARA PODER PARSEAR MEJOR LOS DATOS

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
        
        if count >= SAMPLES_PER_CLASS:
            # Si sobran datos (Clases 0, 4, 5), hacemos downsampling
            print(f"Clase {c}: {count} muestras -> Recortando a {SAMPLES_PER_CLASS}")
            df_resampled = df_class.sample(n=SAMPLES_PER_CLASS, random_state=42)
            
        else:
            # Si faltan datos (Clases 1, 2, 3), hacemos data augmentation
            print(f"Clase {c}: {count} muestras -> Aumentando a {SAMPLES_PER_CLASS}")
            
            # 1. Copiamos los originales
            originals = df_class.copy()
            
            # 2. Calculamos cuántos faltan (aqui le añado algo mas para hacer que se intente disparar mas)
            n_needed = SAMPLES_PER_CLASS - count
            
            # 3. Muestreamos con reemplazo para generar los nuevos
            generated = df_class.sample(n=n_needed, replace=True, random_state=42)
            
            if c in [1, 2]:
                print(f"   -> Añadiendo ruido entero (±{NOISE_INT_RANGE}) a clase {c}")
                
                # Generamos ruido entero entre -NOISE_INT_RANGE y +NOISE_INT_RANGE
                # high es exclusivo en randint, por eso +1
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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fileToHeatmap(filename: str, rows: int = 8, cols: int = 16):
    # Leer fichero a lista 1D
    RAMl = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            RAMl.append(int(line))

    if len(RAMl) != rows * cols:
        raise ValueError(
            f"El fichero tiene {len(RAMl)} valores, pero rows*cols={rows*cols}."
        )

    arr = np.array(RAMl)
    matrix = arr.reshape(rows, cols)


    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        matrix,
        cmap='coolwarm',
        annot=True,
        fmt='d',
        linewidths=0.1,
        linecolor='black',
        cbar_kws={'label': 'Frecuencia de cambios'}
    )


    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels([format(i, 'X') for i in range(cols)])
    ax.set_xlabel("Columna (hex)")

    plt.title("Heatmap RAM")
    plt.ylabel("Fila")
    plt.tight_layout()
    plt.savefig("heatmap_RAM.png", dpi=200)
    print("Heatmap guardado en heatmap_RAM.png")

fileToHeatmap("RamFILE.txt")
    

    


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List

def graficar_puzzle(secuencia: List[Tuple[Tuple[int]]], nombre_archivo: str = "ultima_solucion.png", pasos_por_fila: int = 4):
    pasos = len(secuencia)
    filas = (pasos + pasos_por_fila - 1) // pasos_por_fila
    cols = min(pasos, pasos_por_fila)
    fig, axes = plt.subplots(filas, cols, figsize=(3*cols, 3*filas))
    if filas == 1:
        axes = [axes]
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    for idx, estado in enumerate(secuencia):
        ax = axes[idx]
        ax.set_title(f"Paso {idx}")
        ax.axis('off')
        for i in range(3):
            for j in range(3):
                valor = estado[i][j]
                rect = patches.Rectangle((j, 2-i), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                if valor != 0:
                    ax.text(j+0.5, 2-i+0.5, str(valor), va='center', ha='center', fontsize=16)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
    # Ocultar ejes vacíos si hay
    for idx in range(len(secuencia), len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    plt.savefig(nombre_archivo)
    plt.close()

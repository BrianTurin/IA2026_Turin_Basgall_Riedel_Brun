"""
=============================================================================
TAREA DE CLASE — 19/03/2026
Visualización Gráfica del Árbol de Búsqueda + BFS + DFS
=============================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from collections import deque


# ---------------------------------------------------------------------------
# DATOS DEL ÁRBOL
# ---------------------------------------------------------------------------
NOMBRES = {
    "A": "TuCasa",
    "B": "Plaza",
    "C": "Parque",
    "D": "Supermercado",
    "E": "Escuela",
    "F": "Cine",
    "G": "Biblioteca",
    "H": "Kiosco",
    "I": "Restaurante",
    "J": "Hospital",
    "K": "Banco",
    "L": "Museo",
    "M": "Farmacia",
    "N": "Café",
    "O": "Correo",
    "P": "Estadio",
    "Q": "Zoológico",
    "R": "Terminal",
    "S": "Liceo",
}

ARBOL_EDGES = [
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "E"), ("B", "F"), ("B", "G"),
    ("C", "H"), ("C", "I"),
    ("D", "J"), ("D", "K"),
    ("E", "L"),
    ("F", "M"),
    ("H", "G"),
    ("I", "N"),
    ("J", "G"),
    ("K", "O"),
    ("L", "P"),
    ("M", "Q"),
    ("N", "R"),
    ("O", "S"),
]

INICIO   = "A"
OBJETIVO = "G"

# Posiciones fijas (x, y) para cada nodo en el dibujo
POSICIONES = {
    # Nivel 0
    "A": (9,   8),
    # Nivel 1
    "B": (3,   6),
    "C": (9,   6),
    "D": (15,  6),
    # Nivel 2
    "E": (1,   4),
    "F": (3,   4),
    "G": (5,   4),
    "H": (8,   4),
    "I": (10,  4),
    "J": (13,  4),
    "K": (16,  4),
    # Nivel 3
    "L": (1,   2),
    "M": (3,   2),
    "N": (10,  2),
    "O": (16,  2),
    # Nivel 4
    "P": (1,   0),
    "Q": (3,   0),
    "R": (10,  0),
    "S": (16,  0),
}


def construir_grafo():
    G = nx.DiGraph()
    G.add_nodes_from(NOMBRES.keys())
    G.add_edges_from(ARBOL_EDGES)
    return G


def color_nodo(nodo, bfs_orden=None, dfs_orden=None, camino=None, modo="arbol"):
    if nodo == INICIO:
        return "#4CAF50"   # verde — inicio
    if nodo == OBJETIVO:
        return "#F44336"   # rojo — objetivo
    if camino and nodo in camino:
        return "#FF9800"   # naranja — en el camino solución
    if modo == "bfs" and bfs_orden and nodo in bfs_orden:
        return "#2196F3"   # azul — explorado por BFS
    if modo == "dfs" and dfs_orden and nodo in dfs_orden:
        return "#9C27B0"   # violeta — explorado por DFS
    return "#90A4AE"       # gris — no explorado


def etiqueta_nodo(nodo):
    return f"{nodo}\n{NOMBRES[nodo]}"


# ---------------------------------------------------------------------------
# BFS & DFS (mismos que en busqueda_no_informada.py)
# ---------------------------------------------------------------------------
ARBOL_DICT = {}
for padre, hijo in ARBOL_EDGES:
    ARBOL_DICT.setdefault(padre, []).append(hijo)
for n in NOMBRES:
    ARBOL_DICT.setdefault(n, [])


def bfs():
    cola = deque([INICIO])
    padres = {INICIO: None}
    orden = []
    visitados = {INICIO}
    while cola:
        nodo = cola.popleft()
        orden.append(nodo)
        if nodo == OBJETIVO:
            camino = []
            n = OBJETIVO
            while n:
                camino.append(n)
                n = padres[n]
            return orden, list(reversed(camino))
        for hijo in ARBOL_DICT.get(nodo, []):
            if hijo not in visitados:
                visitados.add(hijo)
                padres[hijo] = nodo
                cola.append(hijo)
    return orden, []


def dfs():
    pila = [(INICIO, [INICIO])]
    visitados = set()
    orden = []
    while pila:
        nodo, camino_actual = pila.pop()
        if nodo in visitados:
            continue
        visitados.add(nodo)
        orden.append(nodo)
        if nodo == OBJETIVO:
            return orden, camino_actual
        for hijo in reversed(ARBOL_DICT.get(nodo, [])):
            if hijo not in visitados:
                pila.append((hijo, camino_actual + [hijo]))
    return orden, []


# ---------------------------------------------------------------------------
# DIBUJO
# ---------------------------------------------------------------------------

def dibujar_grafo(ax, titulo, modo, bfs_orden=None, dfs_orden=None, camino=None):
    G = construir_grafo()
    pos = POSICIONES

    # Colores de nodos
    node_colors = [color_nodo(n, bfs_orden, dfs_orden, camino, modo) for n in G.nodes()]

    # Resaltar aristas del camino
    edge_colors = []
    camino_edges = set()
    if camino:
        camino_edges = set(zip(camino[:-1], camino[1:]))
    for u, v in G.edges():
        if (u, v) in camino_edges:
            edge_colors.append("#FF9800")
        else:
            edge_colors.append("#B0BEC5")

    edge_widths = [3.5 if (u, v) in camino_edges else 1.0 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           width=edge_widths,
                           arrows=True,
                           arrowsize=15,
                           connectionstyle="arc3,rad=0.0")

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=1200,
                           edgecolors="#263238",
                           linewidths=1.5)

    labels = {n: etiqueta_nodo(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6.5, font_weight="bold", font_color="#1A237E")

    # Numeración del orden de exploración
    if modo in ("bfs", "dfs") and (bfs_orden or dfs_orden):
        orden_usado = bfs_orden if modo == "bfs" else dfs_orden
        offsets = {"x": 0.55, "y": 0.30}
        for i, n in enumerate(orden_usado, 1):
            x, y = pos[n]
            ax.text(x + 0.55, y + 0.30, str(i),
                    fontsize=7, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.15",
                              facecolor="#37474F", edgecolor="none"))

    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=10, color="#1A237E")
    ax.axis("off")


def leyenda(ax, modo):
    patches = [
        mpatches.Patch(color="#4CAF50", label="Inicio (TuCasa)"),
        mpatches.Patch(color="#F44336", label="Objetivo (Biblioteca)"),
        mpatches.Patch(color="#FF9800", label="Camino solución"),
    ]
    if modo == "bfs":
        patches.append(mpatches.Patch(color="#2196F3", label="Explorado (BFS)"))
    elif modo == "dfs":
        patches.append(mpatches.Patch(color="#9C27B0", label="Explorado (DFS)"))
    patches.append(mpatches.Patch(color="#90A4AE", label="No explorado"))
    ax.legend(handles=patches, loc="lower left", fontsize=8,
              framealpha=0.9, edgecolor="#B0BEC5")


def main():
    bfs_orden, bfs_camino = bfs()
    dfs_orden, dfs_camino = dfs()

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.patch.set_facecolor("#F5F5F5")
    fig.suptitle(
        "TAREA DE CLASE — 19/03/2026\n"
        "Búsqueda No Informada: BFS vs DFS  |  Ciudad X  |  TuCasa → Biblioteca",
        fontsize=14, fontweight="bold", color="#0D47A1", y=1.01
    )

    # ── Panel 1: Árbol puro ───────────────────────────────────────────────
    dibujar_grafo(axes[0], "Árbol de Búsqueda\n(estructura completa)", "arbol")

    # ── Panel 2: BFS ──────────────────────────────────────────────────────
    dibujar_grafo(axes[1],
                  f"BFS — Búsqueda en Anchura\nCamino: {' → '.join(bfs_camino)}  ({len(bfs_camino)-1} paso(s))\nNodos explorados: {len(bfs_orden)}",
                  "bfs",
                  bfs_orden=bfs_orden,
                  camino=bfs_camino)
    leyenda(axes[1], "bfs")

    # ── Panel 3: DFS ──────────────────────────────────────────────────────
    dibujar_grafo(axes[2],
                  f"DFS — Búsqueda en Profundidad\nCamino: {' → '.join(dfs_camino)}  ({len(dfs_camino)-1} paso(s))\nNodos explorados: {len(dfs_orden)}",
                  "dfs",
                  dfs_orden=dfs_orden,
                  camino=dfs_camino)
    leyenda(axes[2], "dfs")

    plt.tight_layout(pad=2.5)

    # Anotaciones de nivel
    for ax in axes:
        for nivel, y in [(0, 8), (1, 6), (2, 4), (3, 2), (4, 0)]:
            ax.text(-0.5, y, f"Niv.{nivel}", fontsize=7, color="#607D8B",
                    va="center", ha="right")

    salida = "resultado_tarea_19_03_2026.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Gráfico guardado: {salida}")
    plt.show()


if __name__ == "__main__":
    main()

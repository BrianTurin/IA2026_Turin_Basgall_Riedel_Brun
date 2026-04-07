import heapq, time
from collections import deque

OBJETIVO_8 = (1, 2, 3, 4, 5, 6, 7, 8, 0)   # 0 representa el hueco

MOVIMIENTOS = {
    0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
    3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
}

def sucesores_puzzle(estado):
    pos_hueco = estado.index(0)
    resultados = []
    for vecino in MOVIMIENTOS[pos_hueco]:
        lista = list(estado)
        lista[pos_hueco], lista[vecino] = lista[vecino], lista[pos_hueco]
        resultados.append(tuple(lista))
    return resultados

class NodoPuzzle:
    def __init__(self, estado, padre=None, costo=0):
        self.estado = estado
        self.padre  = padre
        self.costo  = costo

    def camino(self):
        nodo, pasos = self, []
        while nodo:
            pasos.append(nodo.estado)
            nodo = nodo.padre
        return list(reversed(pasos))

    def __lt__(self, otro):
        return self.costo < otro.costo

# BFS para 8-puzzle
def bfs_puzzle(inicio):
    if inicio == OBJETIVO_8:
        return NodoPuzzle(inicio)
    frontera   = deque([NodoPuzzle(inicio)])
    explorados = {inicio}
    while frontera:
        nodo = frontera.popleft()
        for suc in sucesores_puzzle(nodo.estado):
            if suc == OBJETIVO_8:
                return NodoPuzzle(suc, nodo, nodo.costo + 1)
            if suc not in explorados:
                explorados.add(suc)
                frontera.append(NodoPuzzle(suc, nodo, nodo.costo + 1))
    return None

# IDS para 8-puzzle
def dls_puzzle(nodo, objetivo, limite, explorados):
    if nodo.estado == objetivo:
        return nodo
    if limite == 0:
        return 'corte'
    hubo_corte = False
    for suc in sucesores_puzzle(nodo.estado):
        if suc in explorados:
            continue
        explorados.add(suc)
        resultado = dls_puzzle(NodoPuzzle(suc, nodo, nodo.costo + 1), objetivo, limite - 1, explorados)
        explorados.discard(suc)
        if resultado == 'corte':
            hubo_corte = True
        elif resultado is not None:
            return resultado
    return 'corte' if hubo_corte else None

def ids_puzzle(inicio):
    for limite in range(50):
        explorados = {inicio}
        resultado = dls_puzzle(NodoPuzzle(inicio), OBJETIVO_8, limite, explorados)
        if resultado not in (None, 'corte'):
            return resultado
    return None

# Prueba 
CASOS = [
    (1, 2, 3, 4, 5, 6, 0, 7, 8),   # 2 pasos
    (1, 2, 3, 4, 5, 0, 6, 7, 8),   # 3 pasos
    (1, 2, 3, 0, 4, 5, 7, 8, 6),   # 5 pasos
    (0, 1, 3, 4, 2, 5, 7, 8, 6),   # 6 pasos
    (1, 0, 3, 4, 2, 5, 7, 8, 6),   # 5 pasos
]

def mostrar_tablero(estado):
    for i in range(0, 9, 3):
        print(' '.join(str(x) if x != 0 else '_' for x in estado[i:i+3]))
    print()

for i, caso in enumerate(CASOS, 1):
    print(f"{'='*40}\nCaso {i} — Estado inicial:")
    mostrar_tablero(caso)

    for nombre, func in [("BFS", bfs_puzzle), ("IDS", ids_puzzle)]:
        t0 = time.perf_counter()
        sol = func(caso)
        elapsed = (time.perf_counter() - t0) * 1000
        pasos = len(sol.camino()) - 1
        print(f"  {nombre}: {pasos} pasos | {elapsed:.3f} ms")
    print()
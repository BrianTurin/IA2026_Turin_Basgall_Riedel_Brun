"""
Agente de búsqueda no informada para el 8-puzzle (Búsqueda en anchura).
"""
from collections import deque
from typing import List, Tuple, Optional

# Estado objetivo del 8-puzzle
GOAL_STATE = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)
)

# Movimientos posibles: (dx, dy)
MOVES = {
    'arriba': (-1, 0),
    'abajo': (1, 0),
    'izquierda': (0, -1),
    'derecha': (0, 1)
}

def encontrar_cero(estado: Tuple[Tuple[int]]) -> Tuple[int, int]:
    for i, fila in enumerate(estado):
        for j, val in enumerate(fila):
            if val == 0:
                return i, j
    raise ValueError("No hay cero en el estado")

def mover(estado: Tuple[Tuple[int]], direccion: str) -> Optional[Tuple[Tuple[int]]]:
    x, y = encontrar_cero(estado)
    dx, dy = MOVES[direccion]
    nx, ny = x + dx, y + dy
    if 0 <= nx < 3 and 0 <= ny < 3:
        estado_lista = [list(fila) for fila in estado]
        estado_lista[x][y], estado_lista[nx][ny] = estado_lista[nx][ny], estado_lista[x][y]
        return tuple(tuple(fila) for fila in estado_lista)
    return None

def sucesores(estado: Tuple[Tuple[int]]) -> List[Tuple[Tuple[int], str]]:
    sucesores = []
    for direccion in MOVES:
        nuevo_estado = mover(estado, direccion)
        if nuevo_estado:
            sucesores.append((nuevo_estado, direccion))
    return sucesores

def bfs(inicial: Tuple[Tuple[int]]) -> Optional[List[str]]:
    visitados = set()
    cola = deque()
    cola.append((inicial, [], [inicial]))  # (estado, camino, secuencia_estados)
    while cola:
        estado, camino, secuencia = cola.popleft()
        if estado == GOAL_STATE:
            return camino, secuencia
        visitados.add(estado)
        for sucesor, accion in sucesores(estado):
            if sucesor not in visitados:
                cola.append((sucesor, camino + [accion], secuencia + [sucesor]))
                visitados.add(sucesor)
    return None, None

def imprimir_estado(estado: Tuple[Tuple[int]]):
    for fila in estado:
        print(' '.join(str(x) if x != 0 else ' ' for x in fila))
    print()

import random
def es_solvable(estado_flat):
    inv = 0
    for i in range(len(estado_flat)):
        for j in range(i+1, len(estado_flat)):
            if estado_flat[i] != 0 and estado_flat[j] != 0 and estado_flat[i] > estado_flat[j]:
                inv += 1
    return inv % 2 == 0

def generar_estado_aleatorio():
    while True:
        nums = list(range(9))
        random.shuffle(nums)
        if es_solvable(nums):
            return tuple(tuple(nums[i*3:(i+1)*3]) for i in range(3))

def probar_estados():
    from graficos import graficar_puzzle
    estado = generar_estado_aleatorio()
    print("Estado inicial aleatorio:")
    imprimir_estado(estado)
    solucion, secuencia = bfs(estado)
    if solucion:
        print(f"Solución encontrada en {len(solucion)} pasos: {solucion}")
        print("Guardando gráfico de la solución en 'ultima_solucion.png'")
        graficar_puzzle(secuencia, "ultima_solucion.png", pasos_por_fila=4)
    else:
        print("No se encontró solución.")

if __name__ == "__main__":
    probar_estados()

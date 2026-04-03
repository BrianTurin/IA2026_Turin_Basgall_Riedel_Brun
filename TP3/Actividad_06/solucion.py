# -*- coding: utf-8 -*-
"""
Actividad 6 - Busqueda Informada: 8-Puzzle
==========================================
Compara dos heuristicas clasicas con dos algoritmos de busqueda informada:

  Heuristicas:
    - h1: Fichas mal colocadas (Misplaced Tiles)
    - h2: Distancia de Manhattan

  Algoritmos:
    - Busqueda Voraz (Greedy Best-First Search): expande segun h(n)
    - A* (A-star)                              : expande segun f(n) = g(n) + h(n)

  => 4 variantes en total, ejecutadas sobre los mismos estados iniciales.

Referencia: Russell & Norvig, AIMA - Cap. 3
"""
import sys
import io
# Forzar UTF-8 en la consola de Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import heapq
import time
import random

# ---------------------------------------------------------------------------
# Estado objetivo
# ---------------------------------------------------------------------------
GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)          # 0 representa la casilla vacia
GOAL_POS = {v: (i // 3, i % 3) for i, v in enumerate(GOAL)}

# ---------------------------------------------------------------------------
# Heuristicas
# ---------------------------------------------------------------------------
def h_misplaced(state):
    """h1: cantidad de fichas en posicion incorrecta (sin contar el hueco)."""
    return sum(1 for i, v in enumerate(state) if v != 0 and v != GOAL[i])


def h_manhattan(state):
    """h2: suma de distancias de Manhattan de cada ficha a su posicion meta."""
    total = 0
    for i, v in enumerate(state):
        if v != 0:
            r, c = divmod(i, 3)
            gr, gc = GOAL_POS[v]
            total += abs(r - gr) + abs(c - gc)
    return total


HEURISTICAS = {
    'Fichas mal colocadas (h1)': h_misplaced,
    'Distancia de Manhattan  (h2)': h_manhattan,
}

# ---------------------------------------------------------------------------
# Operadores: sucesores del 8-puzzle
# ---------------------------------------------------------------------------
MOVES = (-3, 3, -1, 1)   # arriba, abajo, izquierda, derecha


def sucesores(state):
    idx = state.index(0)
    row, col = divmod(idx, 3)
    result = []
    # arriba
    if row > 0:
        s = list(state); s[idx], s[idx-3] = s[idx-3], s[idx]; result.append(tuple(s))
    # abajo
    if row < 2:
        s = list(state); s[idx], s[idx+3] = s[idx+3], s[idx]; result.append(tuple(s))
    # izquierda
    if col > 0:
        s = list(state); s[idx], s[idx-1] = s[idx-1], s[idx]; result.append(tuple(s))
    # derecha
    if col < 2:
        s = list(state); s[idx], s[idx+1] = s[idx+1], s[idx]; result.append(tuple(s))
    return result

# ---------------------------------------------------------------------------
# Busqueda Voraz (Greedy Best-First)
# ---------------------------------------------------------------------------
def busqueda_voraz(estado_inicial, heuristica):
    t0 = time.perf_counter()
    h = heuristica

    frontera = []
    nodo_raiz = (h(estado_inicial), 0, estado_inicial, None)  # (f, id, estado, padre)
    heapq.heappush(frontera, nodo_raiz)
    explorados = {}   # estado -> padre
    id_counter = [1]
    nodos_expandidos = 0
    nodos_generados = 1

    padre_map = {estado_inicial: None}

    while frontera:
        _, _, estado, _ = heapq.heappop(frontera)

        if estado == GOAL:
            camino = reconstruir_camino(padre_map, estado)
            return {
                'camino':           camino,
                'pasos':            len(camino) - 1,
                'nodos_expandidos': nodos_expandidos,
                'nodos_generados':  nodos_generados,
                'tiempo_ms':        (time.perf_counter() - t0) * 1000,
            }

        if estado in explorados:
            continue
        explorados[estado] = True
        nodos_expandidos += 1

        for sucesor in sucesores(estado):
            if sucesor not in explorados:
                if sucesor not in padre_map:
                    padre_map[sucesor] = estado
                nodos_generados += 1
                uid = id_counter[0]; id_counter[0] += 1
                heapq.heappush(frontera, (h(sucesor), uid, sucesor, estado))

    return {'camino': None, 'pasos': None,
            'nodos_expandidos': nodos_expandidos,
            'nodos_generados':  nodos_generados,
            'tiempo_ms':        (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------------------
# A* (A-star)
# ---------------------------------------------------------------------------
def a_estrella(estado_inicial, heuristica):
    t0 = time.perf_counter()
    h = heuristica

    frontera = []
    heapq.heappush(frontera, (h(estado_inicial), 0, 0, estado_inicial))
    mejor_g = {estado_inicial: 0}
    padre_map = {estado_inicial: None}
    explorados = set()
    id_counter = [1]
    nodos_expandidos = 0
    nodos_generados = 1

    while frontera:
        f, _, g, estado = heapq.heappop(frontera)

        if estado == GOAL:
            camino = reconstruir_camino(padre_map, estado)
            return {
                'camino':           camino,
                'pasos':            len(camino) - 1,
                'nodos_expandidos': nodos_expandidos,
                'nodos_generados':  nodos_generados,
                'tiempo_ms':        (time.perf_counter() - t0) * 1000,
            }

        if estado in explorados:
            continue
        explorados.add(estado)
        nodos_expandidos += 1

        for sucesor in sucesores(estado):
            nuevo_g = g + 1
            if sucesor not in explorados and nuevo_g < mejor_g.get(sucesor, float('inf')):
                mejor_g[sucesor] = nuevo_g
                padre_map[sucesor] = estado
                nodos_generados += 1
                uid = id_counter[0]; id_counter[0] += 1
                heapq.heappush(frontera, (nuevo_g + h(sucesor), uid, nuevo_g, sucesor))

    return {'camino': None, 'pasos': None,
            'nodos_expandidos': nodos_expandidos,
            'nodos_generados':  nodos_generados,
            'tiempo_ms':        (time.perf_counter() - t0) * 1000}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def reconstruir_camino(padre_map, estado_final):
    camino = []
    e = estado_final
    while e is not None:
        camino.append(e)
        e = padre_map[e]
    return list(reversed(camino))


def imprimir_tablero(state):
    for i in range(0, 9, 3):
        fila = state[i:i+3]
        print('  ' + ' '.join(str(x) if x != 0 else '_' for x in fila))


def es_resoluble(state):
    """Un 8-puzzle es resoluble si el numero de inversiones es par."""
    lst = [x for x in state if x != 0]
    inv = sum(1 for i in range(len(lst))
              for j in range(i+1, len(lst)) if lst[i] > lst[j])
    return inv % 2 == 0


def generar_estado(pasos=20):
    """Genera un estado aleatorio aplicando 'pasos' movimientos desde el GOAL."""
    estado = GOAL
    for _ in range(pasos):
        vecinos = sucesores(estado)
        estado = random.choice(vecinos)
    return estado


SEP  = '=' * 68
LINE = '-' * 68

# ---------------------------------------------------------------------------
# Ejecucion principal
# ---------------------------------------------------------------------------
def comparar(estados_prueba):
    algoritmos = [
        ('Busqueda Voraz (Greedy)', busqueda_voraz),
        ('A* (A-star)',             a_estrella),
    ]

    print()
    print(SEP)
    print('  BUSQUEDA INFORMADA - 8-Puzzle')
    print('  Comparacion de heuristicas: Fichas mal colocadas vs Manhattan')
    print(SEP)

    for idx, estado_ini in enumerate(estados_prueba):
        print()
        print(f'  === Estado inicial #{idx + 1} ===')
        imprimir_tablero(estado_ini)

        # Tabla de resultados: filas=algoritmos, columnas=heuristicas
        resultados = {}   # (alg_nombre, h_nombre) -> res

        for h_nombre, h_fn in HEURISTICAS.items():
            for alg_nombre, alg_fn in algoritmos:
                res = alg_fn(estado_ini, h_fn)
                resultados[(alg_nombre, h_nombre)] = res

        # Encabezado de tabla
        col_w = 24
        print()
        h_nombres = list(HEURISTICAS.keys())
        print(f"  {'Algoritmo / Metrica':<28}", end='')
        for hn in h_nombres:
            print(f"  {hn[:col_w]:<{col_w}}", end='')
        print()
        print('  ' + LINE)

        metricas = [
            ('Pasos hasta meta',       'pasos',            'd'),
            ('Nodos expandidos',        'nodos_expandidos', 'd'),
            ('Nodos generados',         'nodos_generados',  'd'),
            ('Tiempo (ms)',             'tiempo_ms',        '.4f'),
        ]

        for alg_nombre, _ in algoritmos:
            print(f"\n  [{alg_nombre}]")
            for met_label, met_key, fmt in metricas:
                print(f"    {met_label:<26}", end='')
                for h_nombre in h_nombres:
                    v = resultados[(alg_nombre, h_nombre)][met_key]
                    if v is None:
                        s = 'N/A'
                    else:
                        s = format(v, fmt)
                    print(f"  {s:>{col_w}}", end='')
                print()

        print()
        print('  ' + LINE)


def main():
    random.seed(42)

    # Estados de prueba: dificultad creciente
    estados = [
        # Facil: 4 pasos desde goal
        (1, 2, 3,
         4, 5, 6,
         0, 7, 8),

        # Medio: generado con 15 movimientos
        generar_estado(pasos=15),

        # Dificil: generado con 25 movimientos
        generar_estado(pasos=25),
    ]

    comparar(estados)

    # Conclusion
    print()
    print(SEP)
    print('  CONCLUSIONES')
    print(SEP)
    print("""
  h1 - Fichas mal colocadas:
    Admisible (nunca sobreestima), pero menos informada que h2.
    A* con h1 expande mas nodos que con h2.
    La busqueda Voraz con h1 puede encontrar caminos no optimos.

  h2 - Distancia de Manhattan:
    Tambien admisible y ademas mas informada (h2 >= h1 siempre).
    A* con h2 expande menos nodos: mejor eficiencia, misma optimalidad.
    La busqueda Voraz con h2 suele encontrar antes el objetivo.

  Relacion entre heuristicas: h2 domina a h1 (h2(n) >= h1(n) para todo n).
  => A* con h2 es la variante mas eficiente de las cuatro probadas.
    """)
    print(SEP)
    print()


if __name__ == '__main__':
    main()

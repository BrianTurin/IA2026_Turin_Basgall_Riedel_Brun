# -*- coding: utf-8 -*-
"""
Busqueda Informada - Problema de Romania: Arad -> Bucarest
Algoritmos:
  - Busqueda Voraz (Greedy Best-First Search): h(n)
  - A* (A-star)                              : f(n) = g(n) + h(n)
Heuristica: distancia en linea recta a Bucarest (SLD)
"""
import sys
import io
# Forzar UTF-8 en la consola de Windows para evitar UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import heapq
import time

# ---------------------------------------------------------------------------
# Grafo de Romania (distancias de carretera en km, segun figura)
# ---------------------------------------------------------------------------
GRAFO = {
    'Arad':           [('Zerind', 75),    ('Sibiu', 140),          ('Timisoara', 118)],
    'Zerind':         [('Arad', 75),      ('Oradea', 71)],
    'Oradea':         [('Zerind', 71),    ('Sibiu', 151)],
    'Sibiu':          [('Arad', 140),     ('Oradea', 151),         ('Fagaras', 99),    ('Rimnicu Vilcea', 80)],
    'Timisoara':      [('Arad', 118),     ('Lugoj', 111)],
    'Lugoj':          [('Timisoara', 111),('Mehadia', 70)],
    'Mehadia':        [('Lugoj', 70),     ('Dobreta', 75)],
    'Dobreta':        [('Mehadia', 75),   ('Craiova', 120)],
    'Craiova':        [('Dobreta', 120),  ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
    'Rimnicu Vilcea': [('Sibiu', 80),     ('Craiova', 146),        ('Pitesti', 97)],
    'Fagaras':        [('Sibiu', 99),     ('Bucarest', 211)],
    'Pitesti':        [('Rimnicu Vilcea', 97), ('Craiova', 138),   ('Bucarest', 101)],
    'Bucarest':       [('Fagaras', 211),  ('Pitesti', 101),        ('Giurgiu', 90),    ('Urziceni', 85)],
    'Giurgiu':        [('Bucarest', 90)],
    'Urziceni':       [('Bucarest', 85),  ('Hirsova', 98),         ('Vaslui', 142)],
    'Hirsova':        [('Urziceni', 98),  ('Eforie', 86)],
    'Eforie':         [('Hirsova', 86)],
    'Vaslui':         [('Urziceni', 142), ('Iasi', 92)],
    'Iasi':           [('Vaslui', 92),    ('Neamt', 87)],
    'Neamt':          [('Iasi', 87)],
}

# ---------------------------------------------------------------------------
# Heuristica SLD (Straight-Line Distance) a Bucarest
# Fuente: AIMA (Russell & Norvig), Figura 3.22
# ---------------------------------------------------------------------------
SLD_BUCAREST = {
    'Arad':           366,
    'Bucarest':         0,
    'Craiova':        160,
    'Dobreta':        242,
    'Eforie':         161,
    'Fagaras':        176,
    'Giurgiu':         77,
    'Hirsova':        151,
    'Iasi':           226,
    'Lugoj':          244,
    'Mehadia':        241,
    'Neamt':          234,
    'Oradea':         380,
    'Pitesti':        100,
    'Rimnicu Vilcea': 193,
    'Sibiu':          253,
    'Timisoara':      329,
    'Urziceni':        80,
    'Vaslui':         199,
    'Zerind':         374,
}

def h(nodo):
    """Heuristica: linea recta a Bucarest."""
    return SLD_BUCAREST[nodo]


# ---------------------------------------------------------------------------
# Clase Nodo
# ---------------------------------------------------------------------------
class Nodo:
    def __init__(self, estado, padre=None, costo_g=0):
        self.estado  = estado
        self.padre   = padre
        self.costo_g = costo_g   # costo real acumulado desde el inicio

    def reconstruir_camino(self):
        """Devuelve la lista de ciudades desde la raiz hasta este nodo."""
        nodos, n = [], self
        while n:
            nodos.append(n.estado)
            n = n.padre
        return list(reversed(nodos))

    # heapq requiere comparacion cuando las prioridades empatan
    def __lt__(self, otro):
        return self.estado < otro.estado


# ---------------------------------------------------------------------------
# Busqueda Voraz (Greedy Best-First Search)
# Criterio de expansion: menor h(n)
# ---------------------------------------------------------------------------
def busqueda_voraz(inicio, objetivo):
    t_ini = time.perf_counter()

    frontera         = []
    heapq.heappush(frontera, (h(inicio), Nodo(inicio)))
    explorados       = set()
    nodos_expandidos = 0

    while frontera:
        _, nodo = heapq.heappop(frontera)

        if nodo.estado == objetivo:
            ruta   = nodo.reconstruir_camino()
            costo  = sum(
                next(c for v, c in GRAFO[ruta[i]] if v == ruta[i+1])
                for i in range(len(ruta) - 1)
            )
            return {
                'camino':           ruta,
                'costo':            costo,
                'nodos_expandidos': nodos_expandidos,
                'tiempo_ms':        (time.perf_counter() - t_ini) * 1000,
            }

        if nodo.estado in explorados:
            continue
        explorados.add(nodo.estado)
        nodos_expandidos += 1

        for vecino, peso in GRAFO.get(nodo.estado, []):
            if vecino not in explorados:
                hijo = Nodo(vecino, padre=nodo, costo_g=nodo.costo_g + peso)
                heapq.heappush(frontera, (h(vecino), hijo))

    return {'camino': None, 'costo': None,
            'nodos_expandidos': nodos_expandidos,
            'tiempo_ms': (time.perf_counter() - t_ini) * 1000}


# ---------------------------------------------------------------------------
# A* (A-star)
# Criterio de expansion: menor f(n) = g(n) + h(n)
# ---------------------------------------------------------------------------
def a_estrella(inicio, objetivo):
    t_ini = time.perf_counter()

    frontera         = []
    nodo_raiz        = Nodo(inicio, costo_g=0)
    heapq.heappush(frontera, (h(inicio), nodo_raiz))
    mejor_g          = {inicio: 0}   # mejor costo g conocido por ciudad
    explorados       = set()
    nodos_expandidos = 0

    while frontera:
        _, nodo = heapq.heappop(frontera)

        if nodo.estado == objetivo:
            return {
                'camino':           nodo.reconstruir_camino(),
                'costo':            nodo.costo_g,
                'nodos_expandidos': nodos_expandidos,
                'tiempo_ms':        (time.perf_counter() - t_ini) * 1000,
            }

        if nodo.estado in explorados:
            continue
        explorados.add(nodo.estado)
        nodos_expandidos += 1

        for vecino, peso in GRAFO.get(nodo.estado, []):
            nuevo_g = nodo.costo_g + peso
            if vecino not in explorados and nuevo_g < mejor_g.get(vecino, float('inf')):
                mejor_g[vecino] = nuevo_g
                hijo = Nodo(vecino, padre=nodo, costo_g=nuevo_g)
                heapq.heappush(frontera, (nuevo_g + h(vecino), hijo))

    return {'camino': None, 'costo': None,
            'nodos_expandidos': nodos_expandidos,
            'tiempo_ms': (time.perf_counter() - t_ini) * 1000}


# ---------------------------------------------------------------------------
# Presentacion de resultados y comparacion
# ---------------------------------------------------------------------------
SEP  = "=" * 62
LINE = "-" * 62

def mostrar_resultado(nombre, res):
    print()
    print(LINE)
    print(f"  Algoritmo  : {nombre}")
    print(LINE)
    if res['camino']:
        print(f"  Camino     : {' -> '.join(res['camino'])}")
        print(f"  Costo      : {res['costo']} km")
        print(f"  Nodos exp. : {res['nodos_expandidos']}")
        print(f"  Tiempo     : {res['tiempo_ms']:.5f} ms")
    else:
        print("  Sin solucion.")


def comparar(inicio, objetivo, repeticiones=5000):
    print()
    print(SEP)
    print(f"  BUSQUEDA INFORMADA - Romania")
    print(f"  Problema   : {inicio} -> {objetivo}")
    print(f"  Heuristica : SLD (linea recta a Bucarest)")
    print(f"  Iteraciones para tiempo promedio: {repeticiones}")
    print(SEP)

    # --- Busqueda Voraz ---
    t_acc = 0.0
    for _ in range(repeticiones):
        r_v = busqueda_voraz(inicio, objetivo)
        t_acc += r_v['tiempo_ms']
    r_v['tiempo_ms'] = t_acc / repeticiones
    mostrar_resultado("Busqueda Voraz (Greedy Best-First)", r_v)

    # --- A* ---
    t_acc = 0.0
    for _ in range(repeticiones):
        r_a = a_estrella(inicio, objetivo)
        t_acc += r_a['tiempo_ms']
    r_a['tiempo_ms'] = t_acc / repeticiones
    mostrar_resultado("A* (A-star)", r_a)

    # --- Tabla comparativa ---
    print()
    print(LINE)
    print(f"  {'Metrica':<30} {'Voraz':>12} {'A*':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Costo del camino (km)':<30} {r_v['costo']:>12} {r_a['costo']:>12}")
    print(f"  {'Nodos expandidos':<30} {r_v['nodos_expandidos']:>12} {r_a['nodos_expandidos']:>12}")
    print(f"  {'Tiempo promedio (ms)':<30} {r_v['tiempo_ms']:>12.5f} {r_a['tiempo_ms']:>12.5f}")
    print(f"  {'Camino optimo garantizado':<30} {'No*':>12} {'Si':>12}")
    print(LINE)
    print()
    print("  (*) La Busqueda Voraz no garantiza optimalidad.")
    print("      A* es optima con heuristica admisible (SLD lo es).")
    print()

    # Diferencia de tiempos
    diff = r_a['tiempo_ms'] - r_v['tiempo_ms']
    if diff > 0:
        print(f"  >> A* tardo {diff:.5f} ms mas que Voraz en promedio.")
    else:
        print(f"  >> Voraz tardo {-diff:.5f} ms mas que A* en promedio.")
    print()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    comparar('Arad', 'Bucarest', repeticiones=5000)

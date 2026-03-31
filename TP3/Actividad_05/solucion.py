import heapq
import time

# Grafo del mapa de Rumania
romania_map = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Fagaras': {'Sibiu': 99, 'Bucarest': 211},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Pitesti': 97, 'Craiova': 146},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucarest': 101},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Dobreta': 75},
    'Dobreta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Dobreta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Bucarest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucarest': 90},
    'Urziceni': {'Bucarest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

# Heurística: distancia en línea recta a Bucarest
straight_line_heuristic = {
    'Arad': 366,
    'Bucarest': 0,
    'Craiova': 160,
    'Dobreta': 242,
    'Eforie': 161,
    'Fagaras': 176,
    'Giurgiu': 77,
    'Hirsova': 151,
    'Iasi': 226,
    'Lugoj': 244,
    'Mehadia': 241,
    'Neamt': 234,
    'Oradea': 380,
    'Pitesti': 100,
    'Rimnicu Vilcea': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Urziceni': 80,
    'Vaslui': 199,
    'Zerind': 374
}

def reconstruir_camino(padres, destino):
    camino = []
    while destino:
        camino.append(destino)
        destino = padres.get(destino)
    return list(reversed(camino))

def busqueda_voraz(grafo, heuristica, inicio, objetivo):
    frontera = []
    heapq.heappush(frontera, (heuristica[inicio], inicio))
    visitados = set()
    padres = {inicio: None}
    while frontera:
        _, actual = heapq.heappop(frontera)
        if actual == objetivo:
            return reconstruir_camino(padres, objetivo)
        visitados.add(actual)
        for vecino in grafo[actual]:
            if vecino not in visitados and vecino not in [n for _, n in frontera]:
                padres[vecino] = actual
                heapq.heappush(frontera, (heuristica[vecino], vecino))
    return None

def busqueda_a_estrella(grafo, heuristica, inicio, objetivo):
    frontera = []
    heapq.heappush(frontera, (heuristica[inicio], 0, inicio))
    costos = {inicio: 0}
    padres = {inicio: None}
    while frontera:
        _, costo_actual, actual = heapq.heappop(frontera)
        if actual == objetivo:
            return reconstruir_camino(padres, objetivo)
        for vecino in grafo[actual]:
            nuevo_costo = costo_actual + grafo[actual][vecino]
            if vecino not in costos or nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                prioridad = nuevo_costo + heuristica[vecino]
                padres[vecino] = actual
                heapq.heappush(frontera, (prioridad, nuevo_costo, vecino))
    return None

def main():
    inicio = 'Arad'
    objetivo = 'Bucarest'

    print('Búsqueda Voraz:')
    t0 = time.time()
    camino_voraz = busqueda_voraz(romania_map, straight_line_heuristic, inicio, objetivo)
    t1 = time.time()
    print('Camino:', ' -> '.join(camino_voraz))
    print('Tiempo de ejecución: %.6f segundos' % (t1 - t0))

    print('\nBúsqueda A* (A estrella):')
    t0 = time.time()
    camino_aestrella = busqueda_a_estrella(romania_map, straight_line_heuristic, inicio, objetivo)
    t1 = time.time()
    print('Camino:', ' -> '.join(camino_aestrella))
    print('Tiempo de ejecución: %.6f segundos' % (t1 - t0))

if __name__ == '__main__':
    main()

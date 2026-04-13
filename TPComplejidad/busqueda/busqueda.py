import heapq
import time
from collections import deque

class Nodo:
    def __init__(self, estado, padre=None, accion=None, costo_paso=0, heuristica=0):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo_paso = costo_paso
        self.h = heuristica
        self.g = costo_paso
        self.f = self.g + self.h

    def __lt__(self, otro):
        return self.f < otro.f

def reconstruir_camino(nodo):
    camino = []
    actual = nodo
    while actual:
        camino.append(actual.estado)
        actual = actual.padre
    return camino[::-1]

def busqueda_generica(problema, algoritmo, heuristica=None):
    inicio_tiempo = time.perf_counter()
    nodos_expandidos = 0
    
    start_h = heuristica(problema.estado_inicial) if heuristica else 0
    frontera = [] 
    
    nodo_inicial = Nodo(problema.estado_inicial, costo_paso=0, heuristica=start_h)
    
    if algoritmo == 'BFS':
        frontera = deque([nodo_inicial])
    elif algoritmo == 'DFS':
        frontera = [nodo_inicial]
    else: # UCS, Greedy, A*
        heapq.heappush(frontera, nodo_inicial)
        
    explorados = set()
    
    while frontera:
        if algoritmo == 'BFS': nodo = frontera.popleft()
        elif algoritmo == 'DFS': nodo = frontera.pop()
        else: nodo = heapq.heappop(frontera)
            
        if problema.es_meta(nodo.estado):
            fin_tiempo = time.perf_counter()
            return reconstruir_camino(nodo), nodo.costo_paso, nodos_expandidos, fin_tiempo - inicio_tiempo

        if nodo.estado not in explorados:
            explorados.add(nodo.estado)
            nodos_expandidos += 1
            
            for accion, resultado, costo in problema.sucesores(nodo.estado):
                h_cost = heuristica(resultado) if heuristica else 0
                g_cost = nodo.costo_paso + costo
                
                nuevo_nodo = Nodo(resultado, nodo, accion, g_cost, h_cost)
                if algoritmo == 'Greedy':
                    nuevo_nodo.f = h_cost
                elif algoritmo == 'UCS':
                    nuevo_nodo.f = g_cost
                
                if resultado not in explorados:
                    if algoritmo in ['BFS', 'DFS']:
                        frontera.append(nuevo_nodo)
                    else:
                        heapq.heappush(frontera, nuevo_nodo)
                        
    return None, float('inf'), nodos_expandidos, time.perf_counter() - inicio_tiempo
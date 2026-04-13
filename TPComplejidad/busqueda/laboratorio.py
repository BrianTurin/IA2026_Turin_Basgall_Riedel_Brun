from busqueda import busqueda_generica
from problemas import ProblemaRumania, Problema8Puzzle, h_manhattan

def ejecutar_laboratorio():
    # 1. Mapa de Rumania
    print("--- MEDICIONES: MAPA DE RUMANIA (Meta: Bucharest) ---")
    ciudades_inicio = ['Arad', 'Timisoara', 'Oradea']
    algoritmos = ['BFS', 'DFS', 'UCS', 'Greedy', 'A*']
    
    for ciudad in ciudades_inicio:
        prob = ProblemaRumania(ciudad)
        print(f"\nEstado Inicial: {ciudad}")
        print(f"{'Algoritmo':<10} | {'Costo':<6} | {'Nodos Exp':<10} | {'Tiempo (ms)':<10}")
        for alg in algoritmos:
            h = (lambda s: ProblemaRumania.H_SLD_BUCHAREST[s]) if alg in ['Greedy', 'A*'] else None
            _, costo, nodos, t = busqueda_generica(prob, alg, h)
            print(f"{alg:<10} | {costo:<6} | {nodos:<10} | {t*1000:.4f}")

    # 2. 8-Puzzle
    print("\n\n--- MEDICIONES: 8-PUZZLE (Meta: Estándar) ---")

    estados_puzzle = [
        (1, 2, 3, 4, 5, 6, 0, 7, 8), # Fácil
        (1, 2, 3, 4, 0, 6, 7, 5, 8), # Medio
        (4, 1, 2, 7, 0, 3, 8, 5, 6)  # Difícil
    ]
    
    for i, est in enumerate(estados_puzzle):
        prob = Problema8Puzzle(est)
        print(f"\nEstado Inicial {i+1}: {est}")
        print(f"{'Algoritmo':<10} | {'Pasos':<6} | {'Nodos Exp':<10} | {'Tiempo (ms)':<10}")

        for alg in ['BFS', 'UCS', 'Greedy', 'A*']:
            h = h_manhattan if alg in ['Greedy', 'A*'] else None
            _, costo, nodos, t = busqueda_generica(prob, alg, h)
            print(f"{alg:<10} | {costo:<6} | {nodos:<10} | {t*1000:.4f}")

if __name__ == "__main__":
    ejecutar_laboratorio()
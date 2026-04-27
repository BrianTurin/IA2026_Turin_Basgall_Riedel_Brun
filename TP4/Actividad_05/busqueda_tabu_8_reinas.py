import random

# =============================================================================
# PROBLEMA DE LAS 8 REINAS — Búsqueda Tabú
#
# Segunda resolución, manteniendo el mismo problema que en ags_8_reinas.py.
# La Búsqueda Tabú es una metaheurística de búsqueda local que evita ciclos
# mediante una "lista tabú" que prohíbe temporalmente movimientos recientes.
# =============================================================================

# --- Parámetros ---
N_REINAS        = 8
MAX_ITERACIONES = 500
TAMANO_TABU     = 20   # Cuántos movimientos recientes quedan prohibidos


# =============================================================================
# 1. CODIFICACIÓN DE LA SOLUCIÓN
# =============================================================================
# Igual que en el AGS: una PERMUTACIÓN de [1..8].
# El índice i representa la columna i+1; el valor representa la fila.
# Ejemplo: [4, 2, 7, 5, 1, 8, 6, 3]  →  col1=fila4, col2=fila2, ...
#
# Al usar permutaciones, dos reinas nunca comparten fila ni columna.
# Solo quedan conflictos diagonales, que son los que la búsqueda elimina.
# =============================================================================

def solucion_inicial():
    """Genera una solución inicial aleatoria (permutación de [1..N])."""
    sol = list(range(1, N_REINAS + 1))
    random.shuffle(sol)
    return sol


# =============================================================================
# 2. FUNCIÓN OBJETIVO
# =============================================================================
# Cuenta la cantidad de PARES de reinas que se amenazan en diagonal.
# f(sol) = 0  →  solución perfecta (ningún conflicto).
# Queremos MINIMIZAR esta función.
#
# Dos reinas en columnas i y j se amenazan en diagonal si:
#   |sol[i] - sol[j]| == |i - j|
# =============================================================================

def funcion_objetivo(sol):
    """
    Retorna el número de pares de reinas en conflicto diagonal.
    Objetivo: minimizar hasta llegar a 0.
    """
    conflictos = 0
    n = len(sol)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(sol[i] - sol[j]) == abs(i - j):
                conflictos += 1
    return conflictos


# =============================================================================
# 3. FUNCIÓN DE VECINDARIO
# =============================================================================
# El vecindario de una solución son todos los estados alcanzables mediante
# un INTERCAMBIO (swap) de dos columnas cualesquiera.
#
# Para N=8 hay C(8,2) = 28 vecinos posibles por estado.
# Cada vecino se identifica por el par de índices intercambiados: (i, j).
#
# Ejemplo: si sol = [4, 2, 7, 5, 1, 8, 6, 3], y hacemos swap(0,4)
#          → vecino = [1, 2, 7, 5, 4, 8, 6, 3]
# =============================================================================

def generar_vecindario(sol):
    """
    Genera todos los vecinos posibles por intercambio de dos posiciones.
    Retorna una lista de (vecino, movimiento), donde movimiento = (i, j).
    """
    vecinos = []
    n = len(sol)
    for i in range(n):
        for j in range(i + 1, n):
            vecino = list(sol)
            vecino[i], vecino[j] = vecino[j], vecino[i]   # swap
            movimiento = (i, j)
            vecinos.append((vecino, movimiento))
    return vecinos


# =============================================================================
# 4. LISTA Y MOVIMIENTOS TABÚ
# =============================================================================
# La lista tabú es una cola de tamaño fijo (TAMANO_TABU) que almacena los
# ÚLTIMOS movimientos realizados, representados como pares (i, j).
#
# Un movimiento (i, j) está PROHIBIDO si se encuentra en la lista tabú,
# salvo que el vecino que produce cumpla el CRITERIO DE ASPIRACIÓN:
#   si el vecino es mejor que la mejor solución global conocida,
#   se permite ignoar la prohibición tabú.
#
# Al agregar un nuevo movimiento, si la lista supera TAMANO_TABU,
# se elimina el movimiento más antiguo (FIFO).
# =============================================================================

def es_tabu(movimiento, lista_tabu):
    """Devuelve True si el movimiento está en la lista tabú."""
    return movimiento in lista_tabu

def agregar_tabu(movimiento, lista_tabu, tamano_max):
    """Agrega el movimiento a la lista tabú; elimina el más antiguo si supera el límite."""
    lista_tabu.append(movimiento)
    if len(lista_tabu) > tamano_max:
        lista_tabu.pop(0)   # eliminar el movimiento más antiguo (FIFO)


# =============================================================================
# VISUALIZACIÓN DEL TABLERO
# =============================================================================

def imprimir_tablero(sol):
    """Imprime el tablero de ajedrez con las reinas posicionadas."""
    n = len(sol)
    print("  +" + "---+" * n)
    for col in range(n):
        fila_reina = sol[col]
        linea = f"{col+1} |"
        for fila in range(1, n + 1):
            linea += " Q |" if fila == fila_reina else "   |"
        print(linea)
        print("  +" + "---+" * n)
    print("    " + "  ".join(str(f) for f in range(1, n + 1)))
    print("    (filas)")


# =============================================================================
# BÚSQUEDA TABÚ PRINCIPAL
# =============================================================================

def busqueda_tabu(max_iteraciones=MAX_ITERACIONES, tamano_tabu=TAMANO_TABU):

    # --- Estado inicial ---
    solucion_actual  = solucion_inicial()
    mejor_solucion   = list(solucion_actual)
    mejor_costo      = funcion_objetivo(mejor_solucion)

    lista_tabu = []   # Cola FIFO de movimientos prohibidos

    print("=" * 55)
    print("       BUSQUEDA TABU — 8 REINAS")
    print("=" * 55)
    print(f"Solucion inicial : {solucion_actual}")
    print(f"Costo inicial    : {mejor_costo} conflictos\n")

    for iteracion in range(1, max_iteraciones + 1):

        # --- Generar vecindario completo ---
        vecinos = generar_vecindario(solucion_actual)

        # --- Seleccionar el mejor vecino no tabú (o que satisfaga aspiración) ---
        mejor_vecino     = None
        mejor_mov        = None
        mejor_costo_vec  = float('inf')

        for vecino, movimiento in vecinos:
            costo_vec = funcion_objetivo(vecino)

            # Criterio de aspiración: si mejora el mejor global, se acepta aunque sea tabú
            criterio_aspiracion = (costo_vec < mejor_costo)

            if (not es_tabu(movimiento, lista_tabu)) or criterio_aspiracion:
                if costo_vec < mejor_costo_vec:
                    mejor_vecino    = vecino
                    mejor_mov       = movimiento
                    mejor_costo_vec = costo_vec

        # Si todos los vecinos son tabú (caso raro), forzamos el menos costoso
        if mejor_vecino is None:
            mejor_vecino, mejor_mov = min(
                ((v, m) for v, m in vecinos),
                key=lambda x: funcion_objetivo(x[0])
            )
            mejor_costo_vec = funcion_objetivo(mejor_vecino)

        # --- Mover a la mejor solución vecina ---
        solucion_actual = mejor_vecino
        agregar_tabu(mejor_mov, lista_tabu, tamano_tabu)

        # --- Actualizar mejor global ---
        if mejor_costo_vec < mejor_costo:
            mejor_solucion = list(solucion_actual)
            mejor_costo    = mejor_costo_vec

        # Reporte de progreso
        if iteracion <= 5 or iteracion % 50 == 0:
            print(f"Iteracion {iteracion:4d} | Costo actual: {mejor_costo_vec} "
                  f"| Mejor global: {mejor_costo} | Lista tabu: {len(lista_tabu)} mov.")

        # --- Condición de parada: solución encontrada ---
        if mejor_costo == 0:
            print(f"\n[OK] Solucion perfecta encontrada en la iteracion {iteracion}!")
            break

    # --- Resultado final ---
    print("\n" + "=" * 55)
    print("             RESULTADO FINAL")
    print("=" * 55)
    print(f"Mejor solucion   : {mejor_solucion}")
    print(f"Conflictos       : {mejor_costo}")

    if mejor_costo == 0:
        print("[OK] Ninguna reina se amenaza.\n")
    else:
        print("[--] No se alcanzo solucion perfecta.\n")

    print("Tablero (Q = reina, col = columna, fila = fila):\n")
    imprimir_tablero(mejor_solucion)

    return mejor_solucion, mejor_costo


# =============================================================================
# ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    random.seed(7)   # Semilla para reproducibilidad (quitar para resultados variados)
    busqueda_tabu()

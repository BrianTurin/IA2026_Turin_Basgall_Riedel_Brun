import random

# =============================================================================
# PROBLEMA DE LAS 8 REINAS — Algoritmo Genético Simple (AGS)
# Referencia: Russell & Norvig, Sección 4.3
#
# Max Bezzel (1848): ubicar 8 reinas en un tablero de ajedrez 8×8
# de forma que ninguna se amenace entre sí (misma fila, columna o diagonal).
# =============================================================================

# --- Parámetros del AGS ---
TAMANO_POBLACION = 100
GENERACIONES     = 500
PROB_CRUCE       = 0.9
PROB_MUTACION    = 0.05
N_REINAS         = 8

# =============================================================================
# a) CODIFICACIÓN DE LOS INDIVIDUOS
# =============================================================================
# Cada individuo es una PERMUTACIÓN de [1..8], donde el índice representa la
# columna y el valor representa la fila en que está ubicada la reina.
# Ejemplo: [3, 1, 6, 2, 8, 6, 4, 7]  → col 1 → fila 3, col 2 → fila 1, ...
#
# Ventaja: al usar permutaciones, garantizamos que no hay dos reinas en la
# misma fila ni en la misma columna por construcción. Solo resta penalizar
# las amenazas diagonales.
# =============================================================================

def generar_individuo():
    """Crea un individuo: permutación aleatoria de [1..N_REINAS]."""
    individuo = list(range(1, N_REINAS + 1))
    random.shuffle(individuo)
    return individuo

def generar_poblacion(tamano):
    """Genera la población inicial."""
    return [generar_individuo() for _ in range(tamano)]


# =============================================================================
# FUNCIÓN DE APTITUD (fitness)
# =============================================================================
# Contamos los PARES de reinas que NO se amenazan mutuamente.
# El máximo posible (sin conflictos) es C(8,2) = 28 pares.
# Una aptitud de 28 = solución perfecta.
# =============================================================================

def contar_conflictos(individuo):
    """Cuenta cuántos pares de reinas se amenazan en diagonal."""
    conflictos = 0
    n = len(individuo)
    for i in range(n):
        for j in range(i + 1, n):
            # Dos reinas se amenazan en diagonal si |fila_i - fila_j| == |col_i - col_j|
            if abs(individuo[i] - individuo[j]) == abs(i - j):
                conflictos += 1
    return conflictos

def aptitud(individuo):
    """
    Aptitud = pares de reinas que NO se amenazan.
    Máximo = C(8,2) = 28  →  solución encontrada.
    """
    max_pares = (N_REINAS * (N_REINAS - 1)) // 2  # 28
    conflictos = contar_conflictos(individuo)
    return max_pares - conflictos


# =============================================================================
# b) FUNCIÓN DE SELECCIÓN NATURAL — Selección por Ruleta
# =============================================================================
# La probabilidad de ser seleccionado es proporcional a la aptitud del individuo.
# Los más aptos tienen más chances de "girar la ruleta" a su favor.
# =============================================================================

def seleccion_ruleta(poblacion, aptitudes):
    """
    Selecciona un individuo mediante ruleta (selección proporcional).
    Retorna el individuo (no su índice).
    """
    suma_aptitudes = sum(aptitudes)
    r = random.uniform(0, suma_aptitudes)
    acumulado = 0
    for individuo, apt in zip(poblacion, aptitudes):
        acumulado += apt
        if acumulado >= r:
            return individuo
    return poblacion[-1]  # fallback


# =============================================================================
# c) FUNCIÓN DE COMBINACIÓN (Cruce) — Order Crossover (OX1)
# =============================================================================
# Se elige un segmento del padre1 y se completa con los genes del padre2
# en el orden en que aparecen, preservando la propiedad de permutación.
# Así los hijos siguen siendo permutaciones válidas de [1..8].
# =============================================================================

def cruce_ox1(padre1, padre2):
    """
    Order Crossover (OX1): genera un hijo manteniendo el orden relativo
    de los genes del padre2, respetando el segmento del padre1.
    """
    n = len(padre1)
    # 1. Elegir punto de corte
    inicio = random.randint(0, n - 1)
    fin    = random.randint(inicio + 1, n)

    # 2. Copiar segmento del padre1 al hijo
    hijo = [None] * n
    hijo[inicio:fin] = padre1[inicio:fin]

    # 3. Completar con los genes del padre2 en orden, omitiendo los ya presentes
    genes_padre2 = [g for g in padre2 if g not in hijo]
    pos = 0
    for i in range(n):
        if hijo[i] is None:
            hijo[i] = genes_padre2[pos]
            pos += 1

    return hijo

def cruce(padre1, padre2, prob_cruce):
    """Aplica OX1 con probabilidad prob_cruce; si no, los padres pasan sin cambio."""
    if random.random() < prob_cruce:
        hijo1 = cruce_ox1(padre1, padre2)
        hijo2 = cruce_ox1(padre2, padre1)
        return hijo1, hijo2
    return list(padre1), list(padre2)


# =============================================================================
# d) FUNCIÓN DE MUTACIÓN — Intercambio de posiciones (Swap Mutation)
# =============================================================================
# Se eligen dos posiciones al azar y se intercambian sus valores.
# Esto garantiza que el individuo resultado siga siendo una permutación válida.
# =============================================================================

def mutacion(individuo, prob_mutacion):
    """
    Swap Mutation: intercambia dos genes al azar con probabilidad prob_mutacion.
    """
    individuo = list(individuo)
    if random.random() < prob_mutacion:
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo


# =============================================================================
# VISUALIZACIÓN DEL TABLERO
# =============================================================================

def imprimir_tablero(individuo):
    """Imprime el tablero de ajedrez con las reinas posicionadas."""
    n = len(individuo)
    print("  +" + "---+" * n)
    for col in range(n):
        fila_reina = individuo[col]
        linea = f"{col+1} |"
        for fila in range(1, n + 1):
            if fila == fila_reina:
                linea += " Q |"
            else:
                linea += "   |"
        print(linea)
        print("  +" + "---+" * n)
    print("    " + "  ".join(str(f) for f in range(1, n + 1)))
    print("    (filas)")


# =============================================================================
# ALGORITMO GENÉTICO PRINCIPAL
# =============================================================================

def ags_8_reinas(tamano_poblacion=TAMANO_POBLACION,
                 generaciones=GENERACIONES,
                 prob_cruce=PROB_CRUCE,
                 prob_mutacion=PROB_MUTACION):

    MAX_APTITUD = (N_REINAS * (N_REINAS - 1)) // 2  # 28 = solución perfecta

    # --- Generación de población inicial ---
    poblacion = generar_poblacion(tamano_poblacion)

    mejor_historico      = None
    mejor_aptitud_hist   = -1

    for generacion in range(generaciones):

        # --- Evaluación ---
        aptitudes = [aptitud(ind) for ind in poblacion]

        # Mejor individuo de esta generación
        mejor_apt_gen = max(aptitudes)
        mejor_ind_gen = poblacion[aptitudes.index(mejor_apt_gen)]

        # Actualizar mejor histórico
        if mejor_apt_gen > mejor_aptitud_hist:
            mejor_aptitud_hist = mejor_apt_gen
            mejor_historico    = list(mejor_ind_gen)

        # Reportar progreso cada 50 generaciones
        if generacion == 0 or (generacion + 1) % 50 == 0:
            print(f"Generación {generacion+1:4d} | Mejor aptitud: {mejor_apt_gen}/{MAX_APTITUD} "
                  f"| Conflictos: {MAX_APTITUD - mejor_apt_gen}")

        # Solución encontrada
        if mejor_aptitud_hist == MAX_APTITUD:
            print(f"\n¡Solución encontrada en la generación {generacion + 1}!")
            break

        # --- Selección + Cruce + Mutación ---
        nueva_poblacion = []

        # Elitismo: conservar el mejor individuo de la generación anterior
        nueva_poblacion.append(list(mejor_ind_gen))

        while len(nueva_poblacion) < tamano_poblacion:
            padre1 = seleccion_ruleta(poblacion, aptitudes)
            padre2 = seleccion_ruleta(poblacion, aptitudes)

            hijo1, hijo2 = cruce(padre1, padre2, prob_cruce)

            hijo1 = mutacion(hijo1, prob_mutacion)
            hijo2 = mutacion(hijo2, prob_mutacion)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tamano_poblacion:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    # --- Resultado final ---
    print("\n" + "=" * 50)
    print("         RESULTADO FINAL — AGS 8 REINAS")
    print("=" * 50)
    print(f"Mejor individuo  : {mejor_historico}")
    print(f"Aptitud          : {mejor_aptitud_hist}/{MAX_APTITUD}")
    print(f"Conflictos restantes: {MAX_APTITUD - mejor_aptitud_hist}")

    if mejor_aptitud_hist == MAX_APTITUD:
        print("\n[OK] Solucion perfecta: ninguna reina se amenaza.\n")
    else:
        print("\n[--] No se encontro solucion perfecta en las generaciones dadas.\n")

    print("Tablero (Q = reina, cols = columna, filas = fila):\n")
    imprimir_tablero(mejor_historico)

    return mejor_historico, mejor_aptitud_hist


# =============================================================================
# ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    random.seed(42)  # Semilla para reproducibilidad (quitar para resultados variados)
    ags_8_reinas()

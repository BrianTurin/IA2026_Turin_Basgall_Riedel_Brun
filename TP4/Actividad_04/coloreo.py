import random

# MAPA DE ALEMANIA (Grafo de Adyacencias)
# Cada número es un estado (0 a 15) y su lista contiene a sus vecinos.
adyacencias = {
    0: [1, 6, 10],                     # Baden-Württemberg
    1: [0, 6, 15, 12],                 # Bayern
    2: [3],                            # Berlin
    3: [2, 7, 8, 13, 12],              # Brandenburg
    4: [8],                            # Bremen
    5: [14, 8],                        # Hamburg
    6: [0, 1, 15, 8, 9, 10],           # Hessen
    7: [14, 8, 3],                     # Mecklenburg-Vorpommern
    8: [14, 5, 4, 7, 3, 13, 15, 6, 9], # Niedersachsen
    9: [8, 6, 10],                     # Nordrhein-Westfalen
    10: [9, 6, 0, 11],                 # Rheinland-Pfalz
    11: [10],                          # Saarland
    12: [1, 15, 13, 3],                # Sachsen
    13: [3, 12, 15, 8],                # Sachsen-Anhalt
    14: [5, 8, 7],                     # Schleswig-Holstein
    15: [1, 6, 8, 13, 12]              # Thüringen
}

NUM_ESTADOS = 16
NUM_COLORES = 4  # Colores representados por 0, 1, 2 y 3

# DEFINICIÓN DE LAS FUNCIONES DEL AGS

# a. Codificación de los individuos
def generar_individuo():
    """Genera un arreglo de 16 enteros, cada uno representando un color al azar."""
    return [random.randint(0, NUM_COLORES - 1) for _ in range(NUM_ESTADOS)]

# b. Función de selección natural
def calcular_fitness(individuo):
    """Cuenta la cantidad de conflictos (vecinos con igual color). Se busca minimizar."""
    conflictos = 0
    for nodo, vecinos in adyacencias.items():
        for vecino in vecinos:
            # Solo miramos nodo < vecino para no contar el mismo límite de frontera dos veces
            if nodo < vecino:
                if individuo[nodo] == individuo[vecino]:
                    conflictos += 1
    return conflictos

def seleccion_torneo(poblacion, fitness_poblacion, k=3):
    seleccionados = random.sample(range(len(poblacion)), k)
    mejor_idx = min(seleccionados, key=lambda idx: fitness_poblacion[idx])
    return poblacion[mejor_idx]

# c. Función de combinación
def cruce_uniforme(padre1, padre2):
    hijo = []
    for i in range(NUM_ESTADOS):
        if random.random() < 0.5:
            hijo.append(padre1[i])
        else:
            hijo.append(padre2[i])
    return hijo

# d. Función de mutación
def mutacion(individuo, tasa_mutacion=0.1):
    if random.random() < tasa_mutacion:
        # Elegimos un estado al azar para mutar
        estado_mutar = random.randint(0, NUM_ESTADOS - 1)
        color_actual = individuo[estado_mutar]
        
        # Le asignamos un color al azar diferente al que ya tiene
        colores_disponibles = [c for c in range(NUM_COLORES) if c != color_actual]
        individuo[estado_mutar] = random.choice(colores_disponibles)
        
    return individuo

# BUCLE PRINCIPAL

def ags_coloreo(tamano_poblacion=150, max_generaciones=1000, tasa_mutacion=0.15):
    # 1. Inicializar población
    poblacion = [generar_individuo() for _ in range(tamano_poblacion)]
    mejor_solucion = None
    mejor_fitness = float('inf')

    for generacion in range(max_generaciones):
        # Evaluar fitness de toda la población
        fitness_poblacion = [calcular_fitness(ind) for ind in poblacion]
        
        # Guardar el mejor de la generación actual
        min_fitness_actual = min(fitness_poblacion)
        if min_fitness_actual < mejor_fitness:
            mejor_fitness = min_fitness_actual
            mejor_solucion = poblacion[fitness_poblacion.index(min_fitness_actual)]
            
        # Condición de corte prematuro: si hay 0 conflictos, encontramos la solución perfecta
        if mejor_fitness == 0:
            print(f"¡Solución perfecta encontrada en la generación {generacion}!")
            break

        # 2. Crear nueva generación
        nueva_poblacion = []
        
        # Elitismo: pasamos a la mejor solución directamente a la siguiente generación
        nueva_poblacion.append(mejor_solucion.copy())

        while len(nueva_poblacion) < tamano_poblacion:
            # Selección
            padre1 = seleccion_torneo(poblacion, fitness_poblacion)
            padre2 = seleccion_torneo(poblacion, fitness_poblacion)
            
            # Cruce
            hijo = cruce_uniforme(padre1, padre2)
            
            # Mutación
            hijo = mutacion(hijo, tasa_mutacion)
            
            nueva_poblacion.append(hijo)
            
        poblacion = nueva_poblacion

    return mejor_solucion, mejor_fitness

# EJECUCIÓN
if __name__ == "__main__":
    print("Iniciando AGS para el coloreo del mapa de Alemania (4 colores)...")
    solucion, conflictos = ags_coloreo()
    
    print("\n--- RESULTADO FINAL ---")
    print(f"Configuración de colores: {solucion}")
    print(f"Conflictos restantes: {conflictos}")
    
    if conflictos == 0:
        print("ÉXITO: Ningún estado comparte color con su vecino. Mapa coloreado correctamente.")
    else:
        print("El algoritmo se estancó en un mínimo local. Volvé a ejecutar el script para intentar de nuevo.")
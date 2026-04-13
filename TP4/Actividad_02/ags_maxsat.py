import random

def fitness_a(cromosoma):
    """
    Evalúa la instancia a:
    (x0 ∨ x1) ∧ (x0 ∨ ¬x1) ∧ (¬x0 ∨ x1) ∧ (¬x0 ∨ ¬x1)
    """
    x0 = cromosoma[0] == '1'
    x1 = cromosoma[1] == '1'
    
    clausulas = [
        x0 or x1,
        x0 or not x1,
        not x0 or x1,
        not x0 or not x1
    ]
    return sum(clausulas)

def fitness_b(cromosoma):
    """
    Evalúa la instancia b:
    (x4 ∨ x2 ∨ x3) ∧ (x5 ∨ x1 ∨ x2) ∧ (x4 ∨ x1 ∨ ¬x3) ∧ 
    (x3 ∨ x1 ∨ x2) ∧ (x4 ∨ x1 ∨ ¬x2) ∧ (¬x5 ∨ ¬x1 ∨ x4)
    """
    # x1 corresponde a cromosoma[0], etc.
    x1 = cromosoma[0] == '1'
    x2 = cromosoma[1] == '1'
    x3 = cromosoma[2] == '1'
    x4 = cromosoma[3] == '1'
    x5 = cromosoma[4] == '1'
    
    clausulas = [
        x4 or x2 or x3,
        x5 or x1 or x2,
        x4 or x1 or not x3,
        x3 or x1 or x2,
        x4 or x1 or not x2,
        not x5 or not x1 or x4
    ]
    return sum(clausulas)

def generar_poblacion_inicial(tamano_poblacion, bits):
    """Genera una población inicial aleatoria de cromosomas (cadenas binarias)."""
    poblacion = []
    for _ in range(tamano_poblacion):
        cromosoma = "".join(random.choice(['0', '1']) for _ in range(bits))
        poblacion.append(cromosoma)
    return poblacion

def seleccion_ruleta(poblacion, aptitudes):
    """Selecciona individuos usando el método de la ruleta."""
    min_aptitud = min(aptitudes)
    if min_aptitud < 0:
        aptitudes_corregidas = [a - min_aptitud + 1.0 for a in aptitudes]
    else:
        aptitudes_corregidas = [a + 0.1 for a in aptitudes] 

    suma_aptitudes = sum(aptitudes_corregidas)
    
    seleccionados = []
    for _ in range(len(poblacion)):
        r = random.uniform(0, suma_aptitudes)
        suma_acumulada = 0
        for i, aptitud in enumerate(aptitudes_corregidas):
            suma_acumulada += aptitud
            if suma_acumulada >= r:
                seleccionados.append(poblacion[i])
                break
    return seleccionados

def cruce(padre1, padre2, prob_cruce):
    """Aplica cruce de un punto entre dos padres."""
    if random.random() < prob_cruce:
        punto_cruce = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
        hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
        return hijo1, hijo2
    return padre1, padre2

def mutacion(cromosoma, prob_mutacion):
    """Aplica mutación bit a bit en un cromosoma."""
    cromosoma_mutado = ""
    for bit in cromosoma:
        if random.random() < prob_mutacion:
            cromosoma_mutado += '1' if bit == '0' else '0'
        else:
            cromosoma_mutado += bit
    return cromosoma_mutado

def ags_maxsat(nombre_instancia, funcion_aptitud, bits, tamano_poblacion=20, generaciones=40, prob_cruce=0.85, prob_mutacion=0.05):
    """Algoritmo Genético Simple principal adaptado para MAX-SAT."""
    print(f"\nGenerando AGS para la Instancia {nombre_instancia}...")
    poblacion = generar_poblacion_inicial(tamano_poblacion, bits)
    
    mejor_historico = None
    mejor_aptitud_historica = float('-inf')

    for generacion in range(generaciones):
        aptitudes = [funcion_aptitud(ind) for ind in poblacion]

        mejor_aptitud_gen = max(aptitudes)
        indice_mejor = aptitudes.index(mejor_aptitud_gen)
        mejor_cromosoma_gen = poblacion[indice_mejor]

        if mejor_aptitud_gen > mejor_aptitud_historica:
            mejor_aptitud_historica = mejor_aptitud_gen
            mejor_historico = mejor_cromosoma_gen

        if generacion == 0 or (generacion + 1) % 5 == 0 or generacion == generaciones - 1:
            print(f"Generación {generacion + 1:2d} | Mejor cromosoma: {mejor_cromosoma_gen} | Cláusulas satisfechas: {mejor_aptitud_gen}")

        poblacion_seleccionada = seleccion_ruleta(poblacion, aptitudes)

        nueva_poblacion = []
        random.shuffle(poblacion_seleccionada)
        
        for i in range(0, len(poblacion_seleccionada), 2):
            padre1 = poblacion_seleccionada[i]
            if i + 1 < len(poblacion_seleccionada):
                padre2 = poblacion_seleccionada[i+1]
            else:
                padre2 = poblacion_seleccionada[0]

            hijo1, hijo2 = cruce(padre1, padre2, prob_cruce)
            
            hijo1 = mutacion(hijo1, prob_mutacion)
            hijo2 = mutacion(hijo2, prob_mutacion)
            
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion[:tamano_poblacion]

    print("="*40)
    print(f"--- Resultado Final AGS ({nombre_instancia}) ---")
    if nombre_instancia == 'A':
        vars_str = f"x0={mejor_historico[0]}, x1={mejor_historico[1]}"
    else:
        vars_str = f"x1={mejor_historico[0]}, x2={mejor_historico[1]}, x3={mejor_historico[2]}, x4={mejor_historico[3]}, x5={mejor_historico[4]}"
    
    print(f"Mejor asignación encontrada: {mejor_historico} ({vars_str})")
    print(f"Número máximo de cláusulas satisfechas: {mejor_aptitud_historica}")
    print("="*40)

if __name__ == "__main__":
    # Parámetros para la Instancia A (4 cláusulas, 2 variables)
    ags_maxsat("A", fitness_a, bits=2, tamano_poblacion=10, generaciones=10)
    
    # Parámetros para la Instancia B (6 cláusulas, 5 variables)
    ags_maxsat("B", fitness_b, bits=5, tamano_poblacion=20, generaciones=20)

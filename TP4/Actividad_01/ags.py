import random

def funcion_aptitud(x):
    """
    Función a maximizar: f(x) = x^5 - x^3 - 2x^2
    """
    return (x**5) - (x**3) - 2 * (x**2)

def dec_a_bin(x, bits=6):
    """Convierte un decimal a una cadena binaria de longitud fija."""
    return bin(x)[2:].zfill(bits)

def bin_a_dec(b):
    """Convierte una cadena binaria a un número decimal."""
    return int(b, 2)

def generar_poblacion_inicial(tamano_poblacion, bits=6):
    """Genera una población inicial aleatoria de cromosomas (cadenas binarias)."""
    poblacion = []
    for _ in range(tamano_poblacion):
        cromosoma = "".join(random.choice(['0', '1']) for _ in range(bits))
        poblacion.append(cromosoma)
    return poblacion

def seleccion_ruleta(poblacion, aptitudes):
    """Selecciona individuos usando el método de la ruleta."""
    # Para la ruleta las aptitudes deben ser positivas.
    # Desplazamos las aptitudes si hay valores negativos.
    min_aptitud = min(aptitudes)
    if min_aptitud < 0:
        aptitudes_corregidas = [a - min_aptitud + 1.0 for a in aptitudes]
    else:
        # Sumamos un pequeño valor para asegurar que todos tengan alguna probabilidad
        # (incluso si la aptitud es 0)
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

def ags(tamano_poblacion=20, generaciones=40, prob_cruce=0.85, prob_mutacion=0.05, bits=6):
    """
    Algoritmo Genético Simple principal.
    """
    poblacion = generar_poblacion_inicial(tamano_poblacion, bits)
    
    mejor_historico = None
    mejor_aptitud_historica = float('-inf')

    for generacion in range(generaciones):
        # Evaluación
        valores_x = [bin_a_dec(ind) for ind in poblacion]
        aptitudes = [funcion_aptitud(x) for x in valores_x]

        # Encuentra el mejor individuo de la generación
        mejor_aptitud_gen = max(aptitudes)
        indice_mejor = aptitudes.index(mejor_aptitud_gen)
        mejor_x_gen = valores_x[indice_mejor]

        # Actualiza el mejor histórico
        if mejor_aptitud_gen > mejor_aptitud_historica:
            mejor_aptitud_historica = mejor_aptitud_gen
            mejor_historico = mejor_x_gen

        if generacion == 0 or (generacion + 1) % 5 == 0 or generacion == generaciones - 1:
            print(f"Generación {generacion + 1:2d} | Mejor x: {mejor_x_gen:2d} | f(x): {mejor_aptitud_gen}")

        # Selección
        poblacion_seleccionada = seleccion_ruleta(poblacion, aptitudes)

        # Cruce y Mutación
        nueva_poblacion = []
        random.shuffle(poblacion_seleccionada)
        
        for i in range(0, len(poblacion_seleccionada), 2):
            padre1 = poblacion_seleccionada[i]
            # En caso de tamaño de población impar
            if i + 1 < len(poblacion_seleccionada):
                padre2 = poblacion_seleccionada[i+1]
            else:
                padre2 = poblacion_seleccionada[0]

            hijo1, hijo2 = cruce(padre1, padre2, prob_cruce)
            
            hijo1 = mutacion(hijo1, prob_mutacion)
            hijo2 = mutacion(hijo2, prob_mutacion)
            
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion[:tamano_poblacion]

    print("\n" + "="*30)
    print("--- Resultado Final AGS ---")
    print(f"El mejor valor de x encontrado: {mejor_historico}")
    print(f"El valor máximo de f(x): {mejor_aptitud_historica}")
    print("="*30)

if __name__ == "__main__":
    ags()

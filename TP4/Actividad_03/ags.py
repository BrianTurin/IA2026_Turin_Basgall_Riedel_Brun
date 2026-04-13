import random
import math

def ags_continuo(funcion, num_dimensiones, limite_inf, limite_sup, 
                 tamano_poblacion=40, generaciones=100, prob_cruce=0.8, prob_mutacion=0.1):
    """
    Algoritmo Genético Simple adaptado para codificación real y minimización.
    """
    # 1. Generar población inicial
    poblacion = []
    for _ in range(tamano_poblacion):
        cromosoma = [random.uniform(limite_inf, limite_sup) for _ in range(num_dimensiones)]
        poblacion.append(cromosoma)
        
    mejor_historico = None
    mejor_aptitud_historica = float('inf')

    # Almacenar historial para gráficas o análisis si se desea
    historial_mejores = []

    for generacion in range(generaciones):
        # 2. Evaluación
        aptitudes = [funcion(ind) for ind in poblacion]

        # Encontramos el mejor individuo (el mínimo, ya que buscamos minimizar)
        mejor_aptitud_gen = min(aptitudes)
        indice_mejor = aptitudes.index(mejor_aptitud_gen)
        mejor_x_gen = poblacion[indice_mejor]

        # Actualiza el mejor histórico
        if mejor_aptitud_gen < mejor_aptitud_historica:
            mejor_aptitud_historica = mejor_aptitud_gen
            mejor_historico = list(mejor_x_gen)

        historial_mejores.append(mejor_aptitud_historica)

        # 3. Selección (Torneo)
        def seleccion_torneo(k=3):
            seleccionados_torneo = random.sample(list(zip(poblacion, aptitudes)), k)
            seleccionados_torneo.sort(key=lambda x: x[1]) # Ordenar por aptitud (minimizamos)
            return seleccionados_torneo[0][0] # Devolver el cromosoma del ganador

        poblacion_seleccionada = [seleccion_torneo() for _ in range(tamano_poblacion)]

        # 4. Cruce (Cruce aritmético plano o cruce de 1 punto adaptado)
        nueva_poblacion = []
        # El elitismo ayuda mucho en la minimización continua
        nueva_poblacion.append(list(mejor_historico)) 
        
        while len(nueva_poblacion) < tamano_poblacion:
            padre1 = random.choice(poblacion_seleccionada)
            padre2 = random.choice(poblacion_seleccionada)
            
            if random.random() < prob_cruce:
                # Cruce aritmético
                alpha = random.random()
                hijo1 = [(alpha * p1 + (1 - alpha) * p2) for p1, p2 in zip(padre1, padre2)]
                hijo2 = [((1 - alpha) * p1 + alpha * p2) for p1, p2 in zip(padre1, padre2)]
            else:
                hijo1, hijo2 = list(padre1), list(padre2)
                
            # 5. Mutación
            for h in [hijo1, hijo2]:
                if len(nueva_poblacion) < tamano_poblacion:
                    # Mutación Gaussiana
                    h_mutado = []
                    for gen in h:
                        if random.random() < prob_mutacion:
                            # Perturbación gaussiana respecto al rango
                            delta = random.gauss(0, (limite_sup - limite_inf) * 0.1)
                            nuevo_gen = gen + delta
                            # Asegurarnos de que no exceda los límites
                            nuevo_gen = max(limite_inf, min(nuevo_gen, limite_sup))
                            h_mutado.append(nuevo_gen)
                        else:
                            h_mutado.append(gen)
                    nueva_poblacion.append(h_mutado)

    return mejor_historico, mejor_aptitud_historica, historial_mejores

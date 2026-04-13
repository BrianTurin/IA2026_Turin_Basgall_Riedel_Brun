import random

def pso(funcion, num_dimensiones, limite_inf, limite_sup, 
        num_particulas=40, iteraciones=100, 
        w=0.729, c1=1.494, c2=1.494):
    """
    Algoritmo de Optimización por Enjambre de Partículas (PSO).
    """
    # 1. Inicialización
    particulas = []
    velocidades = []
    pbest_pos = []
    pbest_val = []
    
    gbest_pos = None
    gbest_val = float('inf')

    # Almacenar historial
    historial_mejores = []

    for _ in range(num_particulas):
        # Posiciones iniciales aleatorias dentro de los límites
        pos = [random.uniform(limite_inf, limite_sup) for _ in range(num_dimensiones)]
        # Velocidades iniciales aleatorias (opcionalmente 0, aquí usamos rango de límite)
        vel = [random.uniform(-abs(limite_sup-limite_inf), abs(limite_sup-limite_inf)) * 0.1 for _ in range(num_dimensiones)]
        
        particulas.append(pos)
        velocidades.append(vel)
        pbest_pos.append(list(pos))
        
        val = funcion(pos)
        pbest_val.append(val)
        
        # Actualizar Global Best
        if val < gbest_val:
            gbest_val = val
            gbest_pos = list(pos)

    # 2. Ciclo principal
    for iteracion in range(iteraciones):
        for i in range(num_particulas):
            for d in range(num_dimensiones):
                r1 = random.random()
                r2 = random.random()
                
                # Actualización de velocidad
                vel_cognitiva = c1 * r1 * (pbest_pos[i][d] - particulas[i][d])
                vel_social = c2 * r2 * (gbest_pos[d] - particulas[i][d])
                velocidades[i][d] = w * velocidades[i][d] + vel_cognitiva + vel_social
                
                # Actualización de posición
                particulas[i][d] = particulas[i][d] + velocidades[i][d]
                
                # Restricción de límites
                if particulas[i][d] < limite_inf:
                    particulas[i][d] = limite_inf
                    velocidades[i][d] *= -0.5 # Rebotar amortiguado
                elif particulas[i][d] > limite_sup:
                    particulas[i][d] = limite_sup
                    velocidades[i][d] *= -0.5 # Rebotar amortiguado
            
            # Evaluación
            nuevo_val = funcion(particulas[i])
            
            # Actualización del Personal Best
            if nuevo_val < pbest_val[i]:
                pbest_val[i] = nuevo_val
                pbest_pos[i] = list(particulas[i])
                
                # Actualización del Global Best
                if nuevo_val < gbest_val:
                    gbest_val = nuevo_val
                    gbest_pos = list(particulas[i])
                    
        historial_mejores.append(gbest_val)
        
    return gbest_pos, gbest_val, historial_mejores

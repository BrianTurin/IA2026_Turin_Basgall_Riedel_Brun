import numpy as np

# ==========================================
# 1. DATASET Y RED NEURONAL (PERCEPTRÓN)
# ==========================================

# Generamos 200 puntos aleatorios entre -5 y 5
np.random.seed(42)
X_data = np.random.uniform(-5, 5, (200, 2))
Y_data = np.zeros((200, 2))

# Asignamos las salidas deseadas según el cuadrante
for i, (x, y) in enumerate(X_data):
    if x > 0 and y > 0:   Y_data[i] = [0, 0] # Q1 -> C1
    elif x < 0 and y > 0: Y_data[i] = [0, 1] # Q2 -> C2
    elif x < 0 and y < 0: Y_data[i] = [1, 0] # Q3 -> C3
    else:                 Y_data[i] = [1, 1] # Q4 -> C4

def sigmoide(x):
    # Previene overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def forward_pass(pesos, x_input):
    """Evalúa un punto en la red neuronal usando el vector de 12 pesos"""
    # Desempaquetamos los 12 genes en matrices y sesgos
    W1 = pesos[0:4].reshape(2, 2)
    b1 = pesos[4:6]
    W2 = pesos[6:10].reshape(2, 2)
    b2 = pesos[10:12]
    
    # Capa Oculta
    oculta = sigmoide(np.dot(x_input, W1) + b1)
    # Capa de Salida
    salida = sigmoide(np.dot(oculta, W2) + b2)
    return salida

def calcular_error_red(pesos):
    """Función de Fitness (MSE) - Buscamos minimizar este valor"""
    predicciones = np.array([forward_pass(pesos, x) for x in X_data])
    mse = np.mean((predicciones - Y_data) ** 2)
    return mse

# ==========================================
# 2. ALGORITMO GENÉTICO SIMPLE (AGS)
# ==========================================

def ags_red(dim=12, pop_size=50, generaciones=300, mutation_rate=0.1):
    # Población inicial entre -5 y 5 (pesos aleatorios)
    poblacion = np.random.uniform(-5, 5, (pop_size, dim))
    mejor_fitness = float('inf')
    
    for gen in range(generaciones):
        fitness = np.array([calcular_error_red(ind) for ind in poblacion])
        
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < mejor_fitness:
            mejor_fitness = fitness[min_idx]
            mejor_solucion = poblacion[min_idx].copy()
            
        nueva_poblacion = [mejor_solucion.copy()] # Elitismo
        
        while len(nueva_poblacion) < pop_size:
            # Selección Torneo (k=3)
            t1, t2 = np.random.choice(pop_size, 3), np.random.choice(pop_size, 3)
            p1, p2 = poblacion[t1[np.argmin(fitness[t1])]], poblacion[t2[np.argmin(fitness[t2])]]
            
            # Cruce Uniforme
            mask = np.random.rand(dim) < 0.5
            hijo = np.where(mask, p1, p2)
            
            # Mutación Gaussiana
            if np.random.rand() < mutation_rate:
                hijo += np.random.normal(0, 1.0, dim)
                
            nueva_poblacion.append(hijo)
            
        poblacion = np.array(nueva_poblacion)
        
    return mejor_fitness

# ==========================================
# 3. PARTICLE SWARM OPTIMIZATION (PSO)
# ==========================================

def pso_red(dim=12, num_particulas=50, iteraciones=300):
    w, c1, c2 = 0.729, 1.494, 1.494 # Hiperparámetros clásicos
    
    # Inicialización
    posiciones = np.random.uniform(-5, 5, (num_particulas, dim))
    velocidades = np.random.uniform(-1, 1, (num_particulas, dim))
    
    pbest_pos = posiciones.copy()
    pbest_fit = np.array([calcular_error_red(p) for p in posiciones])
    
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    for _ in range(iteraciones):
        r1, r2 = np.random.rand(num_particulas, dim), np.random.rand(num_particulas, dim)
        
        # Actualización de velocidad y posición
        velocidades = (w * velocidades + 
                       c1 * r1 * (pbest_pos - posiciones) + 
                       c2 * r2 * (gbest_pos - posiciones))
        posiciones += velocidades
        
        # Evaluación
        fitness_actual = np.array([calcular_error_red(p) for p in posiciones])
        
        # Actualizar pbest
        mejoraron = fitness_actual < pbest_fit
        pbest_pos[mejoraron] = posiciones[mejoraron]
        pbest_fit[mejoraron] = fitness_actual[mejoraron]
        
        # Actualizar gbest
        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()
            
    return gbest_fit

# --- EJECUCIÓN Y COMPARACIÓN ---
if __name__ == "__main__":
    print("Entrenando Red Neuronal (12 parámetros)...")
    print("-" * 40)
    
    error_ags = ags_red()
    print(f"AGS -> Mejor Error (MSE): {error_ags:.6f}")
    
    error_pso = pso_red()
    print(f"PSO -> Mejor Error (MSE): {error_pso:.6f}")
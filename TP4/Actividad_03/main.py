from funciones import FuncionesPrueba, CONFIG_FUNCIONES
from ags import ags_continuo
from pso import pso
import time

def ejecutar_comparacion():
    print("="*60)
    print("   Comparación AGS vs PSO (Minimización de Funciones)")
    print("="*60)

    for nombre_func, config in CONFIG_FUNCIONES.items():
        print(f"\n--- Función: {nombre_func} ---")
        print(f"Límites: [{config['limite_inf']}, {config['limite_sup']}] | Dimensiones: {config['dimensiones']}")
        
        funcion = config['funcion']
        dim = config['dimensiones']
        lim_inf = config['limite_inf']
        lim_sup = config['limite_sup']
        
        # --- Configuración común ---
        # Aumentamos poblacion/partículas y generaciones para problemas más complejos (10D)
        pop_size = 50 
        iters = 500
        
        # 1. Ejecutar AGS
        t0 = time.time()
        mejor_pos_ags, mejor_val_ags, hist_ags = ags_continuo(
            funcion=funcion, 
            num_dimensiones=dim, 
            limite_inf=lim_inf, 
            limite_sup=lim_sup,
            tamano_poblacion=pop_size,
            generaciones=iters,
            prob_cruce=0.8,
            prob_mutacion=0.15 # Un poco más alto para evitar óptimos locales en 10D
        )
        t_ags = time.time() - t0
        
        # 2. Ejecutar PSO
        t0 = time.time()
        mejor_pos_pso, mejor_val_pso, hist_pso = pso(
            funcion=funcion,
            num_dimensiones=dim,
            limite_inf=lim_inf,
            limite_sup=lim_sup,
            num_particulas=pop_size,
            iteraciones=iters,
            w=0.729,
            c1=1.494,
            c2=1.494
        )
        t_pso = time.time() - t0
        
        # 3. Resultados
        print("\nResultados AGS:")
        print(f"  Mínimo encontrado : {mejor_val_ags:.6f}")
        # print(f"  Posición          : {[round(x, 4) for x in mejor_pos_ags]}")
        print(f"  Tiempo ejecución  : {t_ags:.4f} seg")
        
        print("\nResultados PSO:")
        print(f"  Mínimo encontrado : {mejor_val_pso:.6f}")
        # print(f"  Posición          : {[round(x, 4) for x in mejor_pos_pso]}")
        print(f"  Tiempo ejecución  : {t_pso:.4f} seg")
        
        print("-" * 40)
        
        # Determinamos el ganador
        if mejor_val_pso < mejor_val_ags:
            print(f"GANADOR en {nombre_func}: PSO (Menor valor alcanzado: {mejor_val_pso:.6f})")
        else:
            print(f"GANADOR en {nombre_func}: AGS (Menor valor alcanzado: {mejor_val_ags:.6f})")
            
    print("\n" + "="*60)
    print("Fin de la ejecución.")
    print("="*60)

if __name__ == "__main__":
    ejecutar_comparacion()

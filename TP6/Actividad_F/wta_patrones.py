# -*- coding: utf-8 -*-
"""
TP 6 - Actividad F
==================
Aprendizaje No Supervisado: Winner-Takes-All (WTA)
Separacion de 3 patrones en 2 categorias.

  PROBLEMA:
    Dado el conjunto de patrones de entrada:
      P1 = (1, 0, 1)
      P2 = (1, 1, 0)
      P3 = (1, 1, 1)

    Clasificarlos en 2 categorias usando el algoritmo WTA.
    Los vectores de pesos iniciales y el factor de aprendizaje
    se generan de forma aleatoria.

  ALGORITMO WINNER-TAKES-ALL (WTA):
    1. Inicializar pesos W_j (aleatorios) para cada neurona j = C0, C1.
    2. Para cada patron de entrada x:
       a. Calcular la distancia euclidiana entre x y cada W_j.
       b. La neurona con menor distancia "gana" (Winner).
       c. Actualizar SOLO los pesos del ganador:
            W_ganador(t+1) = W_ganador(t) + eta * (x - W_ganador(t))
    3. Repetir hasta convergencia (asignaciones estables) o max epocas.
"""

import math
import random
import sys

# Forzar UTF-8 en stdout para evitar errores de encoding en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# UTILIDADES
# =============================================================================

def euclidea(x, w):
    """Distancia euclidiana entre vector x y vector de pesos w."""
    return math.sqrt(sum((xi - wi) ** 2 for xi, wi in zip(x, w)))

def formato_vec(v, decimales=4):
    """Formato compacto de un vector."""
    return "[" + ", ".join(f"{vi:.{decimales}f}" for vi in v) + "]"

SEP  = "=" * 68
sep  = "-" * 68
SSEP = "~" * 68

# =============================================================================
# DATOS DEL PROBLEMA
# =============================================================================
PATRONES = {
    "P1": [1, 0, 1],
    "P2": [1, 1, 0],
    "P3": [1, 1, 1],
}
NOMBRES  = list(PATRONES.keys())
VECTORES = list(PATRONES.values())
N_ENTRADA   = 3   # Dimension de cada patron
N_NEURONAS  = 2   # Numero de categorias (C0, C1)

# =============================================================================
# ENCABEZADO
# =============================================================================
print(SEP)
print("  TP6 - ACTIVIDAD F")
print("  Aprendizaje No Supervisado: Winner-Takes-All (WTA)")
print(SEP)
print()
print("  PROBLEMA:")
print("    Separar los siguientes patrones en 2 categorias:")
for nombre, vec in PATRONES.items():
    print(f"      {nombre} = {vec}")
print()
print("  ALGORITMO: Winner-Takes-All (WTA)")
print("    - Red competitiva con 2 neuronas (C0, C1).")
print("    - Sin supervision: no se proporcionan etiquetas.")
print("    - En cada paso, solo la neurona MAS CERCANA al patron")
print("      actualiza sus pesos (acercandose al patron presentado).")
print("    - La otra neurona permanece intacta.")
print()

# =============================================================================
# PARAMETROS ALEATORIOS
# =============================================================================
# Semilla NO fija -> resultados distintos en cada ejecucion (pesos y eta aleatorios)
random.seed()   # Usa la hora del sistema como semilla

ETA = round(random.uniform(0.1, 0.9), 4)   # Factor de aprendizaje aleatorio
MAX_EPOCAS = 100

print(SEP)
print("  PARAMETROS INICIALES (ALEATORIOS)")
print(SEP)
print()
print(f"  Factor de aprendizaje (eta) = {ETA}  (generado aleatoriamente en [0.1, 0.9])")
print(f"  Numero de epocas maximas    = {MAX_EPOCAS}")
print()

# Pesos iniciales: se asigna aleatoriamente un patron distinto a cada neurona
# y se le añade ruido gaussiano pequeno para diferenciarlas.
# Esto evita el problema de "neurona muerta" (dead neuron) que ocurre cuando
# los pesos aleatorios en [0,1] dejan a una neurona siempre perdedora.
# Esta estrategia es equivalente a la inicializacion de Kohonen con muestras.
indices_base = random.sample(range(len(VECTORES)), N_NEURONAS)
W_iniciales = []
for c, idx in enumerate(indices_base):
    base = VECTORES[idx][:]
    ruido = [random.uniform(-0.15, 0.15) for _ in range(N_ENTRADA)]
    pesos = [max(0.0, min(1.0, base[j] + ruido[j])) for j in range(N_ENTRADA)]
    W_iniciales.append(pesos)

W = [row[:] for row in W_iniciales]   # copia de trabajo

print("  Pesos iniciales (patron base + ruido aleatorio en [-0.15, 0.15]):")
print("  (cada neurona se inicializa con un patron del conjunto)")
for c, idx in enumerate(indices_base):
    print(f"    C{c}: {formato_vec(W[c])}  (base: {NOMBRES[idx]} = {VECTORES[idx]})")
print()

# =============================================================================
# ENTRENAMIENTO WTA
# =============================================================================
print(SEP)
print("  PROCESO DE ENTRENAMIENTO WTA")
print(SEP)
print()

# Cabecera de la tabla de iteraciones
print(f"  {'Ep':>3}  {'Patron':>6}  {'Ganadora':>9}  "
      f"{'dist_C0':>9}  {'dist_C1':>9}  |  W_C0                W_C1")
print(f"  {sep}")

asignaciones_previas = [None] * len(VECTORES)
convergio = False
epoca_convergencia = -1

for epoca in range(1, MAX_EPOCAS + 1):
    asignaciones_actuales = []

    for idx, (nombre, x) in enumerate(zip(NOMBRES, VECTORES)):
        # Paso 1: Calcular distancias a cada neurona
        d = [euclidea(x, W[c]) for c in range(N_NEURONAS)]

        # Paso 2: Determinar ganadora (neurona con menor distancia)
        ganadora = d.index(min(d))
        asignaciones_actuales.append(ganadora)

        # Paso 3: Actualizar pesos de la ganadora (WTA)
        for j in range(N_ENTRADA):
            W[ganadora][j] += ETA * (x[j] - W[ganadora][j])

        # Mostrar solo epocas 1, 2, 3 y la primera de convergencia
        if epoca <= 3 or (convergio is False):
            w0_str = formato_vec(W[0], 3)
            w1_str = formato_vec(W[1], 3)
            print(f"  {epoca:>3}  {nombre:>6}  {'C'+str(ganadora):>9}  "
                  f"{d[0]:>9.5f}  {d[1]:>9.5f}  |  {w0_str}  {w1_str}")

    # Verificar convergencia: asignaciones estables entre epocas consecutivas
    if asignaciones_actuales == asignaciones_previas:
        if not convergio:
            convergio = True
            epoca_convergencia = epoca
            print(f"  {sep}")
    asignaciones_previas = asignaciones_actuales[:]

    if convergio and epoca > 3:
        break   # Mostrar tabla hasta convergencia y salir

print()
if convergio:
    print(f"  >> Convergencia alcanzada en epoca {epoca_convergencia}")
    print(f"     (Las asignaciones no cambiaron entre las epocas "
          f"{epoca_convergencia - 1} y {epoca_convergencia})")
else:
    print(f"  >> No convergio en {MAX_EPOCAS} epocas. Usando asignacion final.")
print()

# =============================================================================
# PESOS FINALES
# =============================================================================
print(SEP)
print("  PESOS FINALES DE LAS NEURONAS")
print(SEP)
print()
for c in range(N_NEURONAS):
    print(f"  C{c}: {formato_vec(W[c])}")
print()

# =============================================================================
# CLASIFICACION FINAL
# =============================================================================
print(SEP)
print("  CLASIFICACION FINAL")
print(SEP)
print()

grupos = {0: [], 1: []}   # Nombre de patrones asignados a cada neurona

print(f"  {'Patron':>6}  {'Vector':>14}  {'dist_C0':>9}  {'dist_C1':>9}  {'Categoria':>10}")
print(f"  {sep}")

for nombre, x in zip(NOMBRES, VECTORES):
    d = [euclidea(x, W[c]) for c in range(N_NEURONAS)]
    ganadora = d.index(min(d))
    grupos[ganadora].append(nombre)
    print(f"  {nombre:>6}  {str(x):>14}  {d[0]:>9.5f}  {d[1]:>9.5f}  {'-> C'+str(ganadora):>10}")

print()
print("  Agrupacion resultante:")
for c in range(N_NEURONAS):
    miembros = ", ".join(grupos[c]) if grupos[c] else "(vacio)"
    print(f"    Categoria C{c}: {miembros}")
print()

# =============================================================================
# ANALISIS Y RAZONAMIENTO DE LA AGRUPACION
# =============================================================================
print(SEP)
print("  RAZONAMIENTO: ¿Por que la red agrupa asi los patrones?")
print(SEP)
print()

# Calcular distancias entre patrones para dar el razonamiento
def dist_patrones(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

P1, P2, P3 = VECTORES
d12 = dist_patrones(P1, P2)
d13 = dist_patrones(P1, P3)
d23 = dist_patrones(P2, P3)

print("  Distancias euclidianas entre los patrones:")
print(f"    dist(P1, P2) = dist({P1}, {P2}) = {d12:.4f}")
print(f"    dist(P1, P3) = dist({P1}, {P3}) = {d13:.4f}")
print(f"    dist(P2, P3) = dist({P2}, {P3}) = {d23:.4f}")
print()

# Identificar cual par es mas cercano
pares = [("P1-P2", d12), ("P1-P3", d13), ("P2-P3", d23)]
par_cercano  = min(pares, key=lambda t: t[1])
par_lejano   = max(pares, key=lambda t: t[1])

print(f"  Par mas CERCANO  : {par_cercano[0]}  (d = {par_cercano[1]:.4f})")
print(f"  Par mas LEJANO   : {par_lejano[0]}  (d = {par_lejano[1]:.4f})")
print()
print("  Analisis geometrico de los patrones (espacio R^3):")
print(f"    P1 = (1, 0, 1) -> posicion 'esquina' baja en x2")
print(f"    P2 = (1, 1, 0) -> posicion 'esquina' baja en x3")
print(f"    P3 = (1, 1, 1) -> posicion 'vertice' maximo del cubo")
print()
print(f"  Observacion clave:")
print(f"    - P2 y P3 comparten x1=1 y x2=1  (difieren solo en x3)")
print(f"      => son los patrones mas CERCANOS entre si  (d={d23:.4f})")
print(f"    - P1 difiere de P2 en x2 y x3 (2 bits)")
print(f"      => P1 es el mas ALEJADO del par P2-P3")
print()
print("  Comportamiento del algoritmo WTA:")
print("    - Los pesos de cada neurona se acercan al centroide")
print("      geometrico de los patrones que le corresponden.")
print("    - Dado que P2 y P3 son mas similares entre si que")
print("      cualquiera de ellos respecto a P1, el algoritmo")
print("      tiende a agruparlos en la MISMA categoria.")
print("    - P1, siendo el mas distinto, queda en la categoria opuesta.")
print()

# Mostrar el agrupamiento obtenido y validar frente al razonamiento geometrico
grupo_c0 = grupos[0]
grupo_c1 = grupos[1]

print("  Agrupacion obtenida por la red WTA:")
print(f"    C0 = {{{', '.join(grupo_c0) if grupo_c0 else 'vacio'}}}")
print(f"    C1 = {{{', '.join(grupo_c1) if grupo_c1 else 'vacio'}}}")
print()

# Interpretacion dinamica segun resultado real
def set_nombres(lista):
    return set(lista)

grp_p1 = next(c for c in range(N_NEURONAS) if "P1" in grupos[c])
grp_p2 = next(c for c in range(N_NEURONAS) if "P2" in grupos[c])
grp_p3 = next(c for c in range(N_NEURONAS) if "P3" in grupos[c])

if grp_p2 == grp_p3 and grp_p1 != grp_p2:
    print("  [CONSISTENTE CON LA GEOMETRIA]")
    print("    La red separo correctamente P1 del par P2-P3.")
    print("    Justificacion:")
    print("      P2=(1,1,0) y P3=(1,1,1) comparten dos coordenadas")
    print("      identicas (x1=1, x2=1) y solo difieren en x3.")
    print(f"     Su distancia euclidiana es {d23:.4f}, la menor del conjunto.")
    print("      El algoritmo WTA, al minimizar distancias, converge")
    print("      hacia representantes (prototipos) que reflejan los")
    print("      centroides naturales: uno para {P2, P3} y otro para {P1}.")
elif grp_p1 == grp_p3 and grp_p2 != grp_p1:
    print("  [RESULTADO INFLUENCIADO POR LOS PESOS INICIALES]")
    print("    La red agrupo P1 con P3 y dejo P2 separado.")
    print("    Justificacion:")
    print("      P1=(1,0,1) y P3=(1,1,1) comparten x1=1 y x3=1.")
    print(f"     dist(P1,P3) = {d13:.4f}  vs  dist(P2,P3) = {d23:.4f}")
    print("      Aunque P2 y P3 son geometricamente los mas cercanos,")
    print("      los pesos iniciales aleatorios pueden llevar a la red")
    print("      a un minimo local diferente en el espacio de busqueda.")
    print("      Con eta o inicializacion distintas puede obtenerse")
    print("      la agrupacion {P2,P3} | {P1} en su lugar.")
elif grp_p1 == grp_p2 and grp_p3 != grp_p1:
    print("  [RESULTADO INFLUENCIADO POR LOS PESOS INICIALES]")
    print("    La red agrupo P1 con P2 y dejo P3 separado.")
    print("    Justificacion:")
    print("      P1=(1,0,1) y P2=(1,1,0) comparten solo x1=1.")
    print(f"     dist(P1,P2) = {d12:.4f} (la mayor del conjunto).")
    print("      Esta agrupacion es la MENOS esperable geometricamente,")
    print("      pero puede ocurrir si los pesos iniciales aleatorios")
    print("      posicionan las neuronas de modo que el recorrido de")
    print("      patrones (P1->P2->P3) los bifurca en este orden.")
    print("      Ejecutar el script varias veces mostrara que la")
    print("      agrupacion {P2,P3} | {P1} es la mas frecuente.")
else:
    print("  [RESULTADO CON NEURONA VACIA O MIXTO]")
    print("    Una de las neuronas capturo todos los patrones.")
    print("    Esto puede ocurrir si ETA es muy alto o los pesos")
    print("    iniciales favorecen fuertemente a una neurona.")
    print("    Prueba ejecutar el script nuevamente.")

print()
print("  Nota sobre aleatoriedad:")
print(f"    eta usado = {ETA}  |  W_C0 inicial: {formato_vec(W_iniciales[0])}  |  W_C1 inicial: {formato_vec(W_iniciales[1])}")
print("    Cada ejecucion puede producir una agrupacion diferente")
print("    segun la semilla aleatoria, pero la mas frecuente")
print("    (geometricamente optima) es:")
print("      Categoria A = {P2, P3}  /  Categoria B = {P1}")
print("    porque P2 y P3 son los patrones mas similares entre si")
print("    (dist=1.0), mientras P1 dista 1.4142 de P2 y 1.0 de P3.")

print()
print(SEP)
print("  FIN - TP6 Actividad F")
print(SEP)

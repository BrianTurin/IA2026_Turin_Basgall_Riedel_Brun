# -*- coding: utf-8 -*-
"""
TP 6 - Actividad E
==================
Modelado del aprendizaje infantil por analogias (razonamiento por analogias).

  ESCENARIO:
    Un nino ha visto un LEON en el zoo y sabe que es peligroso.
    Representacion interna del LEON : P_leon = (1, 1, 0, 1, 0)
    Un dia ve un GATO en la calle   : P_gato  = (1, 1, 1, 0, 1)

    Pregunta: ¿Debe salir corriendo el nino al ver el gato,
              porque se parece demasiado al leon?

  PARTE 1 - Perceptron Multicapa (MLP) con Backpropagation
  ---------------------------------------------------------
    Arquitectura:
      Capa de entrada : x1..x5   (5 entradas binarias)
      Capa oculta     : H        (1 neurona)
      Capa de salida  : Y        (1 neurona)
        Salida 1 = PELIGROSO, Salida 0 = NO peligroso

    Entrenamiento:
      Solo se dispone de UN patron etiquetado (el leon).
      Se entrena hasta convergencia o max_epocas.

    Clasificacion del gato:
      Tras el entrenamiento, se presenta el patron del gato
      y se observa si la red lo clasifica como peligroso.

  PARTE 2 - Aprendizaje No Supervisado: Red de Kohonen (SOM 1-D)
  ---------------------------------------------------------------
    Arquitectura:
      Capa de entrada  : x1..x5   (5 entradas binarias)
      Capa competitiva : 2 neuronas (C0, C1)
        C0 se asociara al prototipo "PELIGROSO"
        C1 se asociara al prototipo "NO PELIGROSO"

    La red aprende sin etiquetas: solo agrupa por similitud.
    Tras el entrenamiento:
      - La neurona ganadora al presentar el LEON define la clase "peligroso".
      - Al presentar el GATO, si activa la misma neurona => parecido al leon.

  Parametros compartidos:
    ETA_MLP       = 0.6   (factor de aprendizaje MLP)
    ETA_KOHONEN   = 0.5   (factor de aprendizaje Kohonen inicial)
    THETA         = 2.5   (umbral MLP, aprox. mitad de neuronas ocultas * max_net)
    MAX_EPOCAS    = 1000
"""

import math
import random
import sys

# Forzar UTF-8 en stdout para evitar errores de encoding en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# DATOS DEL PROBLEMA
# =============================================================================
LEON = [1, 1, 0, 1, 0]   # Patron del leon  -> PELIGROSO (d = 1)
GATO = [1, 1, 1, 0, 1]   # Patron del gato  -> a clasificar

# Distancia de Hamming entre patron A y patron B
def hamming(a, b):
    return sum(ai != bi for ai, bi in zip(a, b))

# Distancia Euclidea entre patron A y pesos W
def euclidea(a, w):
    return math.sqrt(sum((ai - wi)**2 for ai, wi in zip(a, w)))

SEP  = "=" * 68
sep  = "-" * 68
SSEP = "~" * 68

# =============================================================================
# =============================================================================
#   ENCABEZADO GENERAL
# =============================================================================
# =============================================================================
print(SEP)
print("  TP6 - ACTIVIDAD E")
print("  Aprendizaje por Analogias: Leon y Gato")
print(SEP)
print()
print("  ESCENARIO:")
print("    El nino conoce al LEON (zoo) y sabe que es PELIGROSO.")
print(f"    Patron LEON : {LEON}   -> d = 1 (PELIGROSO)")
print(f"    Patron GATO : {GATO}   -> ?")
print()
print(f"  Distancia de Hamming(LEON, GATO) = {hamming(LEON, GATO)} bits diferentes de 5")
print(f"  Similitud = {(5 - hamming(LEON, GATO)) / 5 * 100:.0f}%  (comparten "
      f"{5 - hamming(LEON, GATO)} de 5 caracteristicas)")
print()

# =============================================================================
# =============================================================================
#   PARTE 1: PERCEPTRON MULTICAPA (BACKPROPAGATION)
# =============================================================================
# =============================================================================
print(SEP)
print("  PARTE 1: PERCEPTRON MULTICAPA (MLP) - BACKPROPAGATION")
print(SEP)
print()
print("  Arquitectura:")
print("    Entradas : x1, x2, x3, x4, x5  (5 entradas binarias)")
print("    Oculta   : H  (1 neurona, umbral theta=2.5)")
print("    Salida   : Y  (1 neurona, umbral theta=2.5)")
print("               1 = PELIGROSO,  0 = NO peligroso")
print()
print("  Entrenamiento: solo patron del LEON  (1 patron etiquetado)")
print()

# -----------------------------------------------------------------------------
# Hiperparametros MLP
# -----------------------------------------------------------------------------
ETA_MLP   = 0.6
THETA_MLP = 2.5   # Umbral adecuado para 5 entradas (valor medio del rango neto)
MAX_EPOCAS_MLP = 5000
TOLERANCIA_MLP = 0.01

# -----------------------------------------------------------------------------
# Funcion sigmoide y derivada
# -----------------------------------------------------------------------------
def sigmoid(net):
    return 1.0 / (1.0 + math.exp(-net))

def sigmoid_deriv(o):
    return o * (1.0 - o)

# -----------------------------------------------------------------------------
# Inicializacion de pesos (aleatorios en [-0.5, 0.5])
# -----------------------------------------------------------------------------
random.seed(7)   # Semilla fija para reproducibilidad

v = [random.uniform(-0.5, 0.5) for _ in range(5)]   # Capa oculta: v1..v5
w1 = random.uniform(-0.5, 0.5)                        # Capa salida:  w1

print("  Pesos iniciales (aleatorios en [-0.5, 0.5]):")
print("    Capa oculta (H):")
for i, vi in enumerate(v, 1):
    print(f"      v{i} (x{i}) = {vi:+.6f}")
print(f"    Capa salida (Y):")
print(f"      w1 (H)  = {w1:+.6f}")
print(f"    Umbral (theta) = {THETA_MLP}")
print()

# -----------------------------------------------------------------------------
# Entrenamiento del MLP - Solo patron LEON
# -----------------------------------------------------------------------------
print(f"  Entrenando con el patron LEON durante max {MAX_EPOCAS_MLP} epocas...")
print(f"  (convergencia cuando |error| < {TOLERANCIA_MLP})")
print()

patron_mlp = LEON
d_mlp      = 1        # Deseado: PELIGROSO

historia_mlp = []     # Guardamos (epoca, net_H, o_H, net_Y, o_Y, error)

for epoca in range(1, MAX_EPOCAS_MLP + 1):
    # FORWARD
    net_H = sum(vi * xi for vi, xi in zip(v, patron_mlp)) - THETA_MLP
    o_H   = sigmoid(net_H)
    net_Y = w1 * o_H - THETA_MLP
    o_Y   = sigmoid(net_Y)

    error = d_mlp - o_Y

    # Registrar para mostrar epocas clave
    historia_mlp.append((epoca, net_H, o_H, net_Y, o_Y, error))

    # Convergencia
    if abs(error) < TOLERANCIA_MLP:
        break

    # BACKWARD
    delta_Y = error * sigmoid_deriv(o_Y)
    dw1     = ETA_MLP * delta_Y * o_H

    delta_H = sigmoid_deriv(o_H) * (w1 * delta_Y)
    dv      = [ETA_MLP * delta_H * xi for xi in patron_mlp]

    # Actualizacion
    w1 += dw1
    for i in range(len(v)):
        v[i] += dv[i]

epocas_totales_mlp = epoca

# Mostrar epocas seleccionadas
print(f"  {'Epoca':<8} {'net_H':>10} {'o_H':>8} {'net_Y':>10} "
      f"{'o_Y':>8} {'error':>10}")
print(f"  {sep}")

epocas_mostrar = [0, 1, 4, 9, 24, 49, 99, 199, 499]
for idx in epocas_mostrar:
    if idx < len(historia_mlp):
        ep, nh, oh, ny, oy, err = historia_mlp[idx]
        print(f"  {ep:<8} {nh:>10.6f} {oh:>8.6f} {ny:>10.6f} "
              f"{oy:>8.6f} {err:>+10.6f}")

# Mostrar ultima epoca si no esta ya
last_ep = historia_mlp[-1]
if last_ep[0] not in [ep for ep, *_ in [historia_mlp[i] for i in epocas_mostrar if i < len(historia_mlp)]]:
    ep, nh, oh, ny, oy, err = last_ep
    print(f"  {ep:<8} {nh:>10.6f} {oh:>8.6f} {ny:>10.6f} "
          f"{oy:>8.6f} {err:>+10.6f}")
print()

convergio = abs(historia_mlp[-1][5]) < TOLERANCIA_MLP
if convergio:
    print(f"  Convergencia alcanzada en epoca {epocas_totales_mlp}  "
          f"(|error| = {abs(historia_mlp[-1][5]):.6f} < {TOLERANCIA_MLP})")
else:
    print(f"  No convergio en {MAX_EPOCAS_MLP} epocas  "
          f"(|error| final = {abs(historia_mlp[-1][5]):.6f})")
print()
print("  Pesos finales tras entrenamiento:")
print("    Capa oculta (H):")
for i, vi in enumerate(v, 1):
    print(f"      v{i} (x{i}) = {vi:+.6f}")
print(f"    Capa salida (Y): w1 = {w1:+.6f}")
print()

# -----------------------------------------------------------------------------
# Clasificacion del LEON (verificacion) y el GATO (prediccion)
# -----------------------------------------------------------------------------
def clasificar_mlp(patron, v_pesos, w_peso, theta, nombre):
    net_H = sum(vi * xi for vi, xi in zip(v_pesos, patron)) - theta
    o_H   = sigmoid(net_H)
    net_Y = w_peso * o_H - theta
    o_Y   = sigmoid(net_Y)
    clase = "PELIGROSO" if o_Y >= 0.5 else "NO peligroso"
    print(f"  Patron {nombre} {patron}:")
    print(f"    net_H = {net_H:+.6f},  o_H = {o_H:.6f}")
    print(f"    net_Y = {net_Y:+.6f},  o_Y = {o_Y:.6f}  ->  {clase}  "
          f"({'HUIR' if o_Y >= 0.5 else 'seguro'})")
    return o_Y

print(SSEP)
print("  CLASIFICACION FINAL (MLP)")
print(SSEP)
print()
o_leon = clasificar_mlp(LEON, v, w1, THETA_MLP, "LEON")
print()
o_gato = clasificar_mlp(GATO, v, w1, THETA_MLP, "GATO")
print()

umbral_decision = 0.5
decision_mlp = "HUIR (peligroso)" if o_gato >= umbral_decision else "NO huir (no peligroso)"
print(f"  >> Salida MLP para el GATO: o_Y = {o_gato:.6f}")
print(f"  >> Umbral de decision      : 0.5")
print(f"  >> DECISION DEL NINO       : {decision_mlp}")
print()

# =============================================================================
# =============================================================================
#   PARTE 2: APRENDIZAJE NO SUPERVISADO - RED DE KOHONEN (SOM 1-D)
# =============================================================================
# =============================================================================
print()
print(SEP)
print("  PARTE 2: APRENDIZAJE NO SUPERVISADO - RED DE KOHONEN (SOM 1-D)")
print(SEP)
print()
print("  Arquitectura:")
print("    Entradas    : x1..x5  (5 caracteristicas binarias)")
print("    Neuronas    : C0, C1  (2 neuronas competitivas)")
print("    Sin etiquetas: la red agrupa por similitud geometrica")
print()
print("  Protocolo:")
print("    1. La red aprende con el patron LEON (unico patron)")
print("    2. La neurona que gana con el LEON define el cluster 'peligroso'")
print("    3. Se presenta el GATO: ¿activa la misma neurona que el leon?")
print()

# -----------------------------------------------------------------------------
# Hiperparametros Kohonen
# -----------------------------------------------------------------------------
ETA_KOH_0     = 0.5    # Factor de aprendizaje inicial
MAX_EPOCAS_KOH = 50    # Epocas de entrenamiento
TAU_ETA       = 20.0   # Constante de decaimiento de eta

def eta_kohonen(t, eta0=ETA_KOH_0, tau=TAU_ETA):
    """Decaimiento exponencial del factor de aprendizaje."""
    return eta0 * math.exp(-t / tau)

# -----------------------------------------------------------------------------
# Inicializacion de pesos Kohonen (aleatorios en [0, 1])
# -----------------------------------------------------------------------------
random.seed(3)
W_K = [
    [random.uniform(0, 1) for _ in range(5)],  # C0
    [random.uniform(0, 1) for _ in range(5)],  # C1
]

print("  Pesos iniciales de Kohonen (aleatorios en [0,1]):")
for c, wc in enumerate(W_K):
    print(f"    C{c}: [{', '.join(f'{wi:.4f}' for wi in wc)}]")
print()

# -----------------------------------------------------------------------------
# Entrenamiento Kohonen - Solo patron LEON (1 patron de referencia)
#
#   Nota academica: con un solo patron, la neurona ganadora siempre sera la
#   que tenga pesos mas cercanos al leon.  Actualizamos solo esa neurona
#   (sin vecindad, radio = 0).  Tras suficientes epocas su prototipo converge
#   al vector del leon.  La otra neurona queda con sus pesos iniciales.
# -----------------------------------------------------------------------------
print(f"  Entrenando {MAX_EPOCAS_KOH} epocas con el patron LEON...")
print()
print(f"  {'Epoca':<8} {'eta':>8} {'ganadora':>10} "
      f"{'dist_C0':>10} {'dist_C1':>10}  |  "
      f"W_C0[:3]                W_C1[:3]")
print(f"  {sep}")

for t in range(1, MAX_EPOCAS_KOH + 1):
    eta_t = eta_kohonen(t)
    x = LEON

    # Calculo de distancias euclidianas
    d0 = euclidea(x, W_K[0])
    d1 = euclidea(x, W_K[1])

    # Neurona ganadora (Winner Takes All)
    ganadora = 0 if d0 <= d1 else 1

    # Actualizacion solo de la neurona ganadora
    for j in range(5):
        W_K[ganadora][j] += eta_t * (x[j] - W_K[ganadora][j])

    # Mostrar epocas seleccionadas
    if t in [1, 2, 5, 10, 20, 30, 50] or t == MAX_EPOCAS_KOH:
        w0_str = f"[{W_K[0][0]:.3f},{W_K[0][1]:.3f},{W_K[0][2]:.3f}]"
        w1_str = f"[{W_K[1][0]:.3f},{W_K[1][1]:.3f},{W_K[1][2]:.3f}]"
        print(f"  {t:<8} {eta_t:>8.4f} {'C'+str(ganadora):>10} "
              f"{d0:>10.6f} {d1:>10.6f}  |  "
              f"{w0_str:<24} {w1_str}")

print()
print("  Pesos finales de Kohonen:")
for c, wc in enumerate(W_K):
    print(f"    C{c}: [{', '.join(f'{wi:.6f}' for wi in wc)}]")
print()

# -----------------------------------------------------------------------------
# Identificar cual neurona corresponde al prototipo PELIGROSO
# -----------------------------------------------------------------------------
d_leon_C0 = euclidea(LEON, W_K[0])
d_leon_C1 = euclidea(LEON, W_K[1])
neurona_peligrosa = 0 if d_leon_C0 <= d_leon_C1 else 1

print(f"  Distancias al vector LEON {LEON}:")
print(f"    C0: {d_leon_C0:.6f}")
print(f"    C1: {d_leon_C1:.6f}")
print(f"  => Neurona 'PELIGROSO' = C{neurona_peligrosa}")
print()

# -----------------------------------------------------------------------------
# Clasificacion del GATO con Kohonen
# -----------------------------------------------------------------------------
d_gato_C0 = euclidea(GATO, W_K[0])
d_gato_C1 = euclidea(GATO, W_K[1])
neurona_gato = 0 if d_gato_C0 <= d_gato_C1 else 1

print(SSEP)
print("  CLASIFICACION FINAL (KOHONEN)")
print(SSEP)
print()
print(f"  Patron LEON {LEON}:")
print(f"    dist(C0)={d_leon_C0:.6f},  dist(C1)={d_leon_C1:.6f}")
print(f"    Neurona ganadora = C{neurona_peligrosa}  => cluster PELIGROSO")
print()
print(f"  Patron GATO {GATO}:")
print(f"    dist(C0)={d_gato_C0:.6f},  dist(C1)={d_gato_C1:.6f}")
print(f"    Neurona ganadora = C{neurona_gato}")
if neurona_gato == neurona_peligrosa:
    decision_koh = "HUIR (mismo cluster que el LEON -> peligroso)"
else:
    decision_koh = "NO huir (cluster distinto al LEON -> no peligroso)"
print(f"  >> DECISION DEL NINO (Kohonen): {decision_koh}")
print()

# =============================================================================
# RESUMEN COMPARATIVO FINAL
# =============================================================================
print(SEP)
print("  RESUMEN COMPARATIVO")
print(SEP)
print()
print(f"  Patron LEON : {LEON}   (conocido: PELIGROSO)")
print(f"  Patron GATO : {GATO}   (desconocido)")
print()
print(f"  Similitud de Hamming : {(5-hamming(LEON,GATO))/5*100:.0f}%  "
      f"({5 - hamming(LEON, GATO)}/5 rasgos comunes)")
print()
print(f"  MLP (supervisado):")
print(f"    Salida o_Y (LEON) = {o_leon:.6f}  ->  PELIGROSO")
print(f"    Salida o_Y (GATO) = {o_gato:.6f}  ->  "
      f"{'PELIGROSO' if o_gato >= 0.5 else 'NO peligroso'}")
print(f"    Decision           : {decision_mlp}")
print()
print(f"  KOHONEN (no supervisado):")
print(f"    LEON -> C{neurona_peligrosa} (cluster peligroso)")
print(f"    GATO -> C{neurona_gato} ({decision_koh})")
print()
# Determinar acuerdo entre modelos
mlp_peligroso = o_gato >= 0.5
koh_peligroso = neurona_gato == neurona_peligrosa

print("  CONCLUSION:")
if mlp_peligroso and koh_peligroso:
    print("    AMBOS modelos clasifican al GATO como PELIGROSO.")
    print("    El nino, por analogia, saldria CORRIENDO al ver el gato.")
elif mlp_peligroso and not koh_peligroso:
    print("    El MLP (supervisado) clasifica al GATO como PELIGROSO,")
    print("    pero Kohonen (no supervisado) lo ubica en otro cluster.")
    print("    El MLP generaliza la etiqueta 'peligroso'; Kohonen ve")
    print("    que geometricamente el gato es mas distinto que similar.")
elif not mlp_peligroso and koh_peligroso:
    print("    Kohonen agrupa al GATO con el LEON, pero el MLP")
    print("    no lo clasifica como peligroso (umbral no superado).")
else:
    print("    AMBOS modelos coinciden: el GATO NO es peligroso.")
    print("    El nino NO saldria corriendo.")
print()
print(f"    Similitud Hamming: {(5-hamming(LEON,GATO))/5*100:.0f}%  "
      f"({5-hamming(LEON,GATO)}/5 rasgos comunes de {len(LEON)}).")
print("    Con solo 2/5 rasgos compartidos, la diferencia es suficiente")
print("    para que un modelo no supervisado distinga las dos categorias.")
print("    (En la realidad, el error se corrige con experiencia adicional.)")
print()
print(SEP)
print("  FIN - TP6 Actividad E")
print(SEP)

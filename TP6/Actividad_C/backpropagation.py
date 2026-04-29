# -*- coding: utf-8 -*-
"""
TP 6 - Actividad C
==================
Red neuronal con UNA neurona oculta:

  Arquitectura (segun diagrama):
    Capa de entrada  : X0, X1, X2  (3 entradas + bias implicito en cada neurona)
    Capa oculta      : X3  (1 neurona)
    Capa de salida   : X4  (1 neurona)

  Funcion discriminante : XOR
    XOR(x0, x1, x2) = x0 XOR x1 XOR x2
    P1 = (1, 0, 1)  ->  1 XOR 0 XOR 1 = 0  ->  d = 0
    P2 = (1, 1, 0)  ->  1 XOR 1 XOR 0 = 0  ->  d = 0

  Funcion de activacion : sigmoide  s(net) = 1 / (1 + e^(-net))

  Algoritmo de aprendizaje : Backpropagation (Rumelhart & McClelland, 1986)
    - Una iteracion por patron (P1 primero, luego P2)
    - Factor de aprendizaje n = 0.5
    - Pesos iniciales aleatorios en [-0.5, 0.5]

  Notacion de pesos:
    Capa oculta  (entrada -> X3):
      v0 = bias de X3
      v1 = w(X0 -> X3)
      v2 = w(X1 -> X3)
      v3 = w(X2 -> X3)

    Capa de salida (X3 -> X4):
      w0 = bias de X4
      w1 = w(X3 -> X4)

  Regla Backpropagation:
    Capa de salida:
      net4 = w0 + w1 * o3
      o4   = s(net4)
      d4   = e * s'(net4) = (d - o4) * o4 * (1 - o4)
      Dw_j = n * d4 * entrada_j

    Capa oculta:
      net3 = v0 + v1*x0 + v2*x1 + v3*x2
      o3   = s(net3)
      d3   = s'(net3) * (w1 * d4) = o3 * (1 - o3) * (w1 * d4)
      Dv_j = n * d3 * entrada_j
"""

import math
import random
import sys

# Forzar UTF-8 en stdout para evitar errores de encoding en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -------------------------------------------------------------
# Semilla fija para reproducibilidad
# -------------------------------------------------------------
random.seed(42)

# -------------------------------------------------------------
# Hiperparametros
# -------------------------------------------------------------
ETA = 0.5   # Factor de aprendizaje

# -------------------------------------------------------------
# Funcion sigmoide y su derivada
# -------------------------------------------------------------
def sigmoid(net):
    """s(net) = 1 / (1 + e^(-net))"""
    return 1.0 / (1.0 + math.exp(-net))

def sigmoid_deriv(o):
    """s'(net) = o * (1 - o)  (expresada en terminos de la salida)"""
    return o * (1.0 - o)

# -------------------------------------------------------------
# Funcion discriminante: XOR de 3 bits
# -------------------------------------------------------------
def xor3(x0, x1, x2):
    return x0 ^ x1 ^ x2

# -------------------------------------------------------------
# Patrones de entrenamiento
# -------------------------------------------------------------
#   Cada patron: ((x0, x1, x2), d)
patterns = [
    ((1, 0, 1), xor3(1, 0, 1)),   # P1: 1 XOR 0 XOR 1 = 0
    ((1, 1, 0), xor3(1, 1, 0)),   # P2: 1 XOR 1 XOR 0 = 0
]

# -------------------------------------------------------------
# Pesos iniciales aleatorios en [-0.5, 0.5]
# Capa oculta (bias v0, y pesos v1..v3 para X0,X1,X2 -> X3)
# Capa salida (bias w0, y peso w1 para X3 -> X4)
# -------------------------------------------------------------
v0 = random.uniform(-0.5, 0.5)   # bias de la neurona oculta X3
v1 = random.uniform(-0.5, 0.5)   # X0 -> X3
v2 = random.uniform(-0.5, 0.5)   # X1 -> X3
v3 = random.uniform(-0.5, 0.5)   # X2 -> X3

w0 = random.uniform(-0.5, 0.5)   # bias de la neurona de salida X4
w1 = random.uniform(-0.5, 0.5)   # X3 -> X4

# -------------------------------------------------------------
# Separadores visuales
# -------------------------------------------------------------
SEP  = "=" * 66
sep  = "-" * 66
SSEP = "~" * 66

# =============================================================
# Encabezado general
# =============================================================
print(SEP)
print("  TP6 - ACTIVIDAD C")
print("  Backpropagation (Rumelhart & McClelland) - 1 neurona oculta")
print(SEP)
print()
print("  Arquitectura:")
print("    Entradas  : X0, X1, X2")
print("    Oculta    : X3  (1 neurona)")
print("    Salida    : X4  (1 neurona)")
print()
print("  Funcion discriminante: XOR(x0, x1, x2) = x0 XOR x1 XOR x2")
print()
print("  Patrones de entrenamiento:")
for i, ((x0, x1, x2), d) in enumerate(patterns, 1):
    print(f"    P{i} = ({x0}, {x1}, {x2})  ->  "
          f"XOR = {x0} XOR {x1} XOR {x2} = {d}  ->  d = {d}")
print()
print(f"  Factor de aprendizaje  : n = {ETA}")
print("  Funcion de activacion  : s(net) = 1 / (1 + e^(-net))")
print()
print("  Pesos iniciales (aleatorios en [-0.5, 0.5]):")
print("    Capa oculta (X3):")
print(f"      v0 (bias) = {v0:+.6f}")
print(f"      v1 (X0)   = {v1:+.6f}")
print(f"      v2 (X1)   = {v2:+.6f}")
print(f"      v3 (X2)   = {v3:+.6f}")
print("    Capa salida (X4):")
print(f"      w0 (bias) = {w0:+.6f}")
print(f"      w1 (X3)   = {w1:+.6f}")
print()

# =============================================================
# Backpropagation - Una iteracion por patron
# =============================================================
for idx, ((x0_p, x1_p, x2_p), d) in enumerate(patterns, start=1):

    print(SEP)
    print(f"  PATRON P{idx} = ({x0_p}, {x1_p}, {x2_p})   |   "
          f"Salida deseada d = {d}")
    print(SEP)
    print()

    # ----------------------------------------------------------
    # FASE FORWARD
    # ----------------------------------------------------------
    print("  ── FASE FORWARD ──────────────────────────────────────")
    print()

    # [1] Neurona oculta X3
    net3 = v0*1 + v1*x0_p + v2*x1_p + v3*x2_p
    o3   = sigmoid(net3)

    print("  [1] Neurona oculta X3:")
    print("      net3 = v0·1 + v1·x0 + v2·x1 + v3·x2")
    print(f"           = ({v0:+.6f})(1) + ({v1:+.6f})({x0_p})"
          f" + ({v2:+.6f})({x1_p}) + ({v3:+.6f})({x2_p})")
    print(f"           = {net3:+.6f}")
    print()
    print(f"      o3 = s(net3) = 1 / (1 + e^(-({net3:+.6f}))) = {o3:.6f}")
    print()

    # [2] Neurona de salida X4
    net4 = w0*1 + w1*o3
    o4   = sigmoid(net4)

    print("  [2] Neurona de salida X4:")
    print("      net4 = w0·1 + w1·o3")
    print(f"           = ({w0:+.6f})(1) + ({w1:+.6f})({o3:.6f})")
    print(f"           = {net4:+.6f}")
    print()
    print(f"      o4 = s(net4) = 1 / (1 + e^(-({net4:+.6f}))) = {o4:.6f}")
    print()

    # ----------------------------------------------------------
    # FASE BACKWARD
    # ----------------------------------------------------------
    print("  ── FASE BACKWARD (Backpropagation) ───────────────────")
    print()

    # [3] Error global
    error = d - o4
    print("  [3] Error en la salida:")
    print(f"      e = d - o4 = {d} - {o4:.6f} = {error:+.6f}")
    print()

    # [4] Delta capa de salida (d4)
    sp4  = sigmoid_deriv(o4)
    d4   = error * sp4
    print("  [4] Delta de la neurona de salida (X4):")
    print("      d4 = (d - o4) * s'(net4)")
    print("         = (d - o4) * o4 * (1 - o4)")
    print(f"         = ({error:+.6f}) * {o4:.6f} * {1-o4:.6f}")
    print(f"         = ({error:+.6f}) * {sp4:.6f}")
    print(f"         = {d4:+.6f}")
    print()

    # [5] Incrementos capa de salida
    dw0 = ETA * d4 * 1
    dw1 = ETA * d4 * o3
    print("  [5] Incrementos capa de salida  (Dw = n * d4 * entrada):")
    print(f"      Dw0 = {ETA} * ({d4:+.6f}) * 1        = {dw0:+.6f}")
    print(f"      Dw1 = {ETA} * ({d4:+.6f}) * o3")
    print(f"          = {ETA} * ({d4:+.6f}) * {o3:.6f} = {dw1:+.6f}")
    print()

    # [6] Delta capa oculta (d3)
    sp3  = sigmoid_deriv(o3)
    # Propagacion del error hacia atras: delta_oculta = s'(net3) * sum(w_k * d_k)
    # Solo hay una neurona de salida, entonces: d3 = sp3 * (w1 * d4)
    d3   = sp3 * (w1 * d4)
    print("  [6] Delta de la neurona oculta (X3):")
    print("      d3 = s'(net3) * (w1 * d4)")
    print("         = o3 * (1 - o3) * (w1 * d4)")
    print(f"         = {o3:.6f} * {1-o3:.6f} * ({w1:+.6f} * {d4:+.6f})")
    print(f"         = {sp3:.6f} * ({w1 * d4:+.6f})")
    print(f"         = {d3:+.6f}")
    print()

    # [7] Incrementos capa oculta
    dv0 = ETA * d3 * 1
    dv1 = ETA * d3 * x0_p
    dv2 = ETA * d3 * x1_p
    dv3 = ETA * d3 * x2_p
    print("  [7] Incrementos capa oculta  (Dv = n * d3 * entrada):")
    print(f"      Dv0 = {ETA} * ({d3:+.6f}) * 1        = {dv0:+.6f}")
    print(f"      Dv1 = {ETA} * ({d3:+.6f}) * {x0_p}        = {dv1:+.6f}")
    print(f"      Dv2 = {ETA} * ({d3:+.6f}) * {x1_p}        = {dv2:+.6f}")
    print(f"      Dv3 = {ETA} * ({d3:+.6f}) * {x2_p}        = {dv3:+.6f}")
    print()

    # [8] Actualizacion de pesos
    w0_old, w1_old = w0, w1
    v0_old, v1_old, v2_old, v3_old = v0, v1, v2, v3

    w0 += dw0;  w1 += dw1
    v0 += dv0;  v1 += dv1;  v2 += dv2;  v3 += dv3

    print("  [8] Actualizacion de pesos (w_nuevo = w_viejo + Dw):")
    print()
    print("      Capa de salida (X4):")
    print(f"        w0: {w0_old:+.6f} + ({dw0:+.6f}) = {w0:+.6f}")
    print(f"        w1: {w1_old:+.6f} + ({dw1:+.6f}) = {w1:+.6f}")
    print()
    print("      Capa oculta (X3):")
    print(f"        v0: {v0_old:+.6f} + ({dv0:+.6f}) = {v0:+.6f}")
    print(f"        v1: {v1_old:+.6f} + ({dv1:+.6f}) = {v1:+.6f}")
    print(f"        v2: {v2_old:+.6f} + ({dv2:+.6f}) = {v2:+.6f}")
    print(f"        v3: {v3_old:+.6f} + ({dv3:+.6f}) = {v3:+.6f}")
    print()

# =============================================================
# Resumen final
# =============================================================
print(SEP)
print("  PESOS FINALES  (tras una iteracion sobre P1 y P2)")
print(SEP)
print()
print("  Capa oculta (X0, X1, X2 -> X3):")
print(f"    v0 (bias) = {v0:+.6f}")
print(f"    v1 (X0)   = {v1:+.6f}")
print(f"    v2 (X1)   = {v2:+.6f}")
print(f"    v3 (X2)   = {v3:+.6f}")
print()
print("  Capa salida (X3 -> X4):")
print(f"    w0 (bias) = {w0:+.6f}")
print(f"    w1 (X3)   = {w1:+.6f}")
print()

# Verificacion
print("  Verificacion con pesos finales:")
print(f"  {'Patron':<12} {'net3':>10} {'o3':>8} {'net4':>10} "
      f"{'o4':>8} {'d':>4} {'error':>10}")
print(f"  {'-'*64}")
for i, ((x0_p, x1_p, x2_p), d) in enumerate(patterns, 1):
    net3 = v0 + v1*x0_p + v2*x1_p + v3*x2_p
    o3   = sigmoid(net3)
    net4 = w0 + w1*o3
    o4   = sigmoid(net4)
    print(f"  P{i} ({x0_p},{x1_p},{x2_p})   {net3:>10.6f} {o3:>8.6f} "
          f"{net4:>10.6f} {o4:>8.6f} {d:>4}  {d-o4:>+10.6f}")
print()

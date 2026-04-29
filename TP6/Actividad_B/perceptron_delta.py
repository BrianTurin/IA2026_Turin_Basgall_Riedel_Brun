# -*- coding: utf-8 -*-
"""
TP 6 - Actividad B
==================
Separacion de puntos con un perceptron simple usando:
  - Funcion de activacion  : sigmoide  s(net) = 1 / (1 + e^(-net))
  - Regla de aprendizaje   : Regla Delta Generalizada
  - Factor de aprendizaje  : n = 0.5
  - Una sola iteracion por patron de entrada

Funcion objetivo (salida deseada):
  f(x, y) = 3x + 2y > 2  ->  1 si se cumple, 0 si no

Patrones:
  P1 = (1, 1) -> f = 3(1)+2(1) = 5 > 2  -> d = 1
  P2 = (1, 0) -> f = 3(1)+2(0) = 3 > 2  -> d = 1
  P3 = (0, 1) -> f = 3(0)+2(1) = 2 > 2  -> d = 0  (no estrictamente mayor)
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
ETA = 0.5          # Factor de aprendizaje (n)

# -------------------------------------------------------------
# Funcion sigmoide y su derivada
# -------------------------------------------------------------
def sigmoid(net):
    """Funcion de activacion sigmoide: s(net) = 1 / (1 + e^(-net))"""
    return 1.0 / (1.0 + math.exp(-net))

def sigmoid_deriv(s):
    """Derivada de la sigmoide: s'(net) = s(1 - s)"""
    return s * (1.0 - s)

# -------------------------------------------------------------
# Patrones de entrenamiento y salidas deseadas
# -------------------------------------------------------------
patterns = [
    ((1, 1), 1),   # P1 -> 3(1)+2(1)=5 > 2 -> d=1
    ((1, 0), 1),   # P2 -> 3(1)+2(0)=3 > 2 -> d=1
    ((0, 1), 0),   # P3 -> 3(0)+2(1)=2 > 2 -> d=0
]

# -------------------------------------------------------------
# Inicializacion: pesos aleatorios pequenos en [-0.5, 0.5]
# -------------------------------------------------------------
w0 = random.uniform(-0.5, 0.5)   # peso del bias
w1 = random.uniform(-0.5, 0.5)   # peso de x1
w2 = random.uniform(-0.5, 0.5)   # peso de x2

SEP = "=" * 60
sep = "-" * 60

print(SEP)
print("  PERCEPTRON SIMPLE - Regla Delta Generalizada")
print(SEP)
print()
print("Pesos iniciales (aleatorios en [-0.5, 0.5]):")
print(f"  w0 (bias) = {w0:+.6f}")
print(f"  w1        = {w1:+.6f}")
print(f"  w2        = {w2:+.6f}")
print()
print(f"Factor de aprendizaje n = {ETA}")
print("Funcion de activacion  : s(net) = 1 / (1 + e^(-net))")
print("Regla de actualizacion : Dw_i = n * delta * x_i")
print("  donde  delta = (d - o) * s'(net) = (d - o) * o * (1 - o)")
print()

# -------------------------------------------------------------
# Una iteracion por patron  (P1 -> P2 -> P3)
# -------------------------------------------------------------
for idx, ((x1, x2), d) in enumerate(patterns, start=1):
    print(sep)
    print(f"  PATRON P{idx} = ({x1}, {x2})   |   salida deseada d = {d}")
    print(sep)

    # [1] Entrada neta
    net = w0 * 1 + w1 * x1 + w2 * x2
    print()
    print("  [1] Entrada neta:")
    print("      net = w0*1 + w1*x1 + w2*x2")
    print(f"          = ({w0:+.6f})(1) + ({w1:+.6f})({x1}) + ({w2:+.6f})({x2})")
    print(f"          = {net:+.6f}")

    # [2] Salida de la neurona (sigmoide)
    o = sigmoid(net)
    print()
    print("  [2] Salida de la neurona (sigmoide):")
    print(f"      o = s({net:.6f}) = 1 / (1 + e^(-({net:.6f}))) = {o:.6f}")

    # [3] Error
    error = d - o
    print()
    print("  [3] Error:")
    print(f"      e = d - o = {d} - {o:.6f} = {error:+.6f}")

    # [4] Termino delta (Regla Delta Generalizada)
    sp = sigmoid_deriv(o)
    delta = error * sp
    print()
    print("  [4] Termino delta (Regla Delta Generalizada):")
    print(f"      s'(net) = o*(1-o) = {o:.6f} * {1-o:.6f} = {sp:.6f}")
    print(f"      delta   = e * s'(net) = ({error:+.6f}) * ({sp:.6f}) = {delta:+.6f}")

    # [5] Incrementos Dw
    dw0 = ETA * delta * 1
    dw1 = ETA * delta * x1
    dw2 = ETA * delta * x2
    print()
    print("  [5] Incrementos Dw = n * delta * x_i:")
    print(f"      Dw0 = {ETA} * ({delta:+.6f}) * 1  = {dw0:+.6f}")
    print(f"      Dw1 = {ETA} * ({delta:+.6f}) * {x1}  = {dw1:+.6f}")
    print(f"      Dw2 = {ETA} * ({delta:+.6f}) * {x2}  = {dw2:+.6f}")

    # [6] Actualizacion de pesos
    w0_old, w1_old, w2_old = w0, w1, w2
    w0 += dw0
    w1 += dw1
    w2 += dw2
    print()
    print("  [6] Actualizacion de pesos (w_nuevo = w_viejo + Dw):")
    print(f"      w0: {w0_old:+.6f} + ({dw0:+.6f}) = {w0:+.6f}")
    print(f"      w1: {w1_old:+.6f} + ({dw1:+.6f}) = {w1:+.6f}")
    print(f"      w2: {w2_old:+.6f} + ({dw2:+.6f}) = {w2:+.6f}")
    print()

# -------------------------------------------------------------
# Resumen final
# -------------------------------------------------------------
print(SEP)
print("  PESOS FINALES (tras 1 iteracion sobre los 3 patrones)")
print(SEP)
print(f"  w0 (bias) = {w0:+.6f}")
print(f"  w1        = {w1:+.6f}")
print(f"  w2        = {w2:+.6f}")
print()
print("  Verificacion con pesos finales:")
print(f"  {'Patron':<10} {'net':>12} {'o':>10} {'d':>5} {'error':>12}")
print(f"  {'-'*52}")
for idx, ((x1, x2), d) in enumerate(patterns, start=1):
    net = w0 * 1 + w1 * x1 + w2 * x2
    o   = sigmoid(net)
    print(f"  P{idx} ({x1},{x2})   {net:>12.6f} {o:>10.6f} {d:>5}  {d-o:>+12.6f}")
print()

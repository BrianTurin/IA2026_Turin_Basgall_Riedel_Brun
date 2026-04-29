# -*- coding: utf-8 -*-
"""
TP 6 - Actividad D
==================
Perceptron multicapa con UNA neurona oculta:

  Arquitectura:
    Capa de entrada  : X1, X2, X3, X4  (4 entradas)
    Capa oculta      : H   (1 neurona)
    Capa de salida   : Y   (1 neurona)

  Patrones de entrenamiento (tabla dada):
    P1 = (1, 1, 0, 1)  ->  Clase + -> d = 1
    P2 = (0, 1, 1, 0)  ->  Clase - -> d = 0
    P3 = (0, 0, 1, 1)  ->  Clase - -> d = 0
    P4 = (0, 0, 0, 1)  ->  Clase + -> d = 1

  Parametros:
    Valor umbral (theta) = 4  (se resta al net: net = sum(wi*xi) - theta)
    Pesos iniciales      = 1  (todos los pesos wij = 1, incluidos los de conexion con umbral)
    Factor de aprendizaje (eta) = 0.6
    Funcion de activacion: sigmoide  s(net) = 1 / (1 + e^(-net))
    Regla de aprendizaje : Regla Delta Generalizada (Backpropagation)

  Notacion de pesos:
    Capa oculta (entradas -> H):
      v1 = w(X1 -> H),  v2 = w(X2 -> H),  v3 = w(X3 -> H),  v4 = w(X4 -> H)
      (el umbral theta_H = 4 se resta del net, equivalente a un bias fijo)

    Capa de salida (H -> Y):
      w1 = w(H -> Y)
      (el umbral theta_Y = 4 se resta del net)

  Formula de activacion:
    net_H = v1*x1 + v2*x2 + v3*x3 + v4*x4 - theta
    o_H   = s(net_H)
    net_Y = w1 * o_H - theta
    o_Y   = s(net_Y)

  Regla Delta Generalizada:
    Capa salida:
      delta_Y = (d - o_Y) * o_Y * (1 - o_Y)
      Dw1     = eta * delta_Y * o_H
      Dtheta_Y= eta * delta_Y * (-1)   [el umbral actua como entrada con valor -1]

    Capa oculta:
      delta_H = o_H * (1 - o_H) * (w1 * delta_Y)
      Dvi     = eta * delta_H * xi
      Dtheta_H= eta * delta_H * (-1)

  Nota: el umbral se trata como un peso conectado a una entrada constante de -1,
  por lo que su actualizacion sigue la misma regla que el resto de los pesos.
"""

import math
import sys

# Forzar UTF-8 en stdout para evitar errores de encoding en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -------------------------------------------------------------
# Hiperparametros
# -------------------------------------------------------------
ETA   = 0.6   # Factor de aprendizaje
THETA = 4.0   # Valor umbral (fijo, se resta del net)

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
# Patrones de entrenamiento: ((x1, x2, x3, x4), d)
# Clase + -> d = 1,  Clase - -> d = 0
# -------------------------------------------------------------
patterns = [
    ((1, 1, 0, 1), 1),   # P1: Clase +
    ((0, 1, 1, 0), 0),   # P2: Clase -
    ((0, 0, 1, 1), 0),   # P3: Clase -
    ((0, 0, 0, 1), 1),   # P4: Clase +
]

# -------------------------------------------------------------
# Pesos iniciales: todos wij = 1
# Capa oculta: v1, v2, v3, v4  (para X1, X2, X3, X4 -> H)
# Capa salida: w1              (para H -> Y)
# -------------------------------------------------------------
v1 = 1.0   # X1 -> H
v2 = 1.0   # X2 -> H
v3 = 1.0   # X3 -> H
v4 = 1.0   # X4 -> H

w1 = 1.0   # H  -> Y

# -------------------------------------------------------------
# Separadores visuales
# -------------------------------------------------------------
SEP = "=" * 66
sep = "-" * 66

# =============================================================
# Encabezado general
# =============================================================
print(SEP)
print("  TP6 - ACTIVIDAD D")
print("  Backpropagation - Perceptron Multicapa - 1 neurona oculta")
print(SEP)
print()
print("  Arquitectura:")
print("    Entradas : X1, X2, X3, X4")
print("    Oculta   : H  (1 neurona)")
print("    Salida   : Y  (1 neurona)")
print()
print("  Patrones de entrenamiento:")
clase_str = {1: "+", 0: "-"}
for i, ((x1, x2, x3, x4), d) in enumerate(patterns, 1):
    print(f"    P{i} = ({x1}, {x2}, {x3}, {x4})  ->  Clase {clase_str[d]}  ->  d = {d}")
print()
print(f"  Umbral (theta)         : theta = {THETA:.1f}")
print(f"  Factor de aprendizaje  : eta = {ETA}")
print("  Funcion de activacion  : s(net) = 1 / (1 + e^(-net))")
print("  Regla de aprendizaje   : Regla Delta Generalizada (Backpropagation)")
print()
print("  Pesos iniciales (wij = 1):")
print("    Capa oculta (H):")
print(f"      v1 (X1) = {v1:+.6f}")
print(f"      v2 (X2) = {v2:+.6f}")
print(f"      v3 (X3) = {v3:+.6f}")
print(f"      v4 (X4) = {v4:+.6f}")
print("    Capa salida (Y):")
print(f"      w1 (H)  = {w1:+.6f}")
print()

# =============================================================
# Backpropagation - Una iteracion sobre todos los patrones
# =============================================================
for idx, ((x1_p, x2_p, x3_p, x4_p), d) in enumerate(patterns, start=1):

    print(SEP)
    print(f"  PATRON P{idx} = ({x1_p}, {x2_p}, {x3_p}, {x4_p})   |   "
          f"Salida deseada d = {d}  (Clase {clase_str[d]})")
    print(SEP)
    print()

    # ----------------------------------------------------------
    # FASE FORWARD
    # ----------------------------------------------------------
    print("  -- FASE FORWARD -----------------------------------------------")
    print()

    # [1] Neurona oculta H
    net_H = v1*x1_p + v2*x2_p + v3*x3_p + v4*x4_p - THETA
    o_H   = sigmoid(net_H)

    print("  [1] Neurona oculta H:")
    print("      net_H = v1*x1 + v2*x2 + v3*x3 + v4*x4 - theta")
    print(f"            = ({v1:+.6f})({x1_p}) + ({v2:+.6f})({x2_p})"
          f" + ({v3:+.6f})({x3_p}) + ({v4:+.6f})({x4_p}) - {THETA:.1f}")
    print(f"            = {v1*x1_p:+.6f} + {v2*x2_p:+.6f}"
          f" + {v3*x3_p:+.6f} + {v4*x4_p:+.6f} - {THETA:.6f}")
    print(f"            = {net_H:+.6f}")
    print()
    print(f"      o_H = s(net_H) = 1 / (1 + e^(-({net_H:+.6f}))) = {o_H:.6f}")
    print()

    # [2] Neurona de salida Y
    net_Y = w1*o_H - THETA
    o_Y   = sigmoid(net_Y)

    print("  [2] Neurona de salida Y:")
    print("      net_Y = w1 * o_H - theta")
    print(f"            = ({w1:+.6f})({o_H:.6f}) - {THETA:.1f}")
    print(f"            = {w1*o_H:+.6f} - {THETA:.6f}")
    print(f"            = {net_Y:+.6f}")
    print()
    print(f"      o_Y = s(net_Y) = 1 / (1 + e^(-({net_Y:+.6f}))) = {o_Y:.6f}")
    print()

    # ----------------------------------------------------------
    # FASE BACKWARD (Regla Delta Generalizada)
    # ----------------------------------------------------------
    print("  -- FASE BACKWARD (Regla Delta Generalizada) -------------------")
    print()

    # [3] Error global
    error = d - o_Y
    print("  [3] Error en la salida:")
    print(f"      e = d - o_Y = {d} - {o_Y:.6f} = {error:+.6f}")
    print()

    # [4] Delta de la neurona de salida
    sp_Y    = sigmoid_deriv(o_Y)
    delta_Y = error * sp_Y
    print("  [4] Delta de la neurona de salida (Y):")
    print("      delta_Y = (d - o_Y) * s'(net_Y)")
    print("              = (d - o_Y) * o_Y * (1 - o_Y)")
    print(f"              = ({error:+.6f}) * {o_Y:.6f} * {1-o_Y:.6f}")
    print(f"              = ({error:+.6f}) * {sp_Y:.6f}")
    print(f"              = {delta_Y:+.6f}")
    print()

    # [5] Incrementos capa de salida
    dw1 = ETA * delta_Y * o_H
    print("  [5] Incrementos capa de salida  (Dw = eta * delta_Y * entrada):")
    print(f"      Dw1 = {ETA} * ({delta_Y:+.6f}) * o_H")
    print(f"          = {ETA} * ({delta_Y:+.6f}) * {o_H:.6f}")
    print(f"          = {dw1:+.6f}")
    print()

    # [6] Delta de la neurona oculta
    sp_H    = sigmoid_deriv(o_H)
    delta_H = sp_H * (w1 * delta_Y)
    print("  [6] Delta de la neurona oculta (H):")
    print("      delta_H = s'(net_H) * (w1 * delta_Y)")
    print("              = o_H * (1 - o_H) * (w1 * delta_Y)")
    print(f"              = {o_H:.6f} * {1-o_H:.6f} * ({w1:+.6f} * {delta_Y:+.6f})")
    print(f"              = {sp_H:.6f} * ({w1 * delta_Y:+.6f})")
    print(f"              = {delta_H:+.6f}")
    print()

    # [7] Incrementos capa oculta
    dv1 = ETA * delta_H * x1_p
    dv2 = ETA * delta_H * x2_p
    dv3 = ETA * delta_H * x3_p
    dv4 = ETA * delta_H * x4_p
    print("  [7] Incrementos capa oculta  (Dv = eta * delta_H * entrada):")
    print(f"      Dv1 = {ETA} * ({delta_H:+.6f}) * {x1_p}  = {dv1:+.6f}")
    print(f"      Dv2 = {ETA} * ({delta_H:+.6f}) * {x2_p}  = {dv2:+.6f}")
    print(f"      Dv3 = {ETA} * ({delta_H:+.6f}) * {x3_p}  = {dv3:+.6f}")
    print(f"      Dv4 = {ETA} * ({delta_H:+.6f}) * {x4_p}  = {dv4:+.6f}")
    print()

    # [8] Actualizacion de pesos
    w1_old             = w1
    v1_old, v2_old     = v1, v2
    v3_old, v4_old     = v3, v4

    w1 += dw1
    v1 += dv1;  v2 += dv2;  v3 += dv3;  v4 += dv4

    print("  [8] Actualizacion de pesos (w_nuevo = w_viejo + Dw):")
    print()
    print("      Capa de salida (Y):")
    print(f"        w1: {w1_old:+.6f} + ({dw1:+.6f}) = {w1:+.6f}")
    print()
    print("      Capa oculta (H):")
    print(f"        v1: {v1_old:+.6f} + ({dv1:+.6f}) = {v1:+.6f}")
    print(f"        v2: {v2_old:+.6f} + ({dv2:+.6f}) = {v2:+.6f}")
    print(f"        v3: {v3_old:+.6f} + ({dv3:+.6f}) = {v3:+.6f}")
    print(f"        v4: {v4_old:+.6f} + ({dv4:+.6f}) = {v4:+.6f}")
    print()

# =============================================================
# Resumen final
# =============================================================
print(SEP)
print("  PESOS FINALES  (tras una iteracion sobre P1, P2, P3 y P4)")
print(SEP)
print()
print("  Capa oculta (X1, X2, X3, X4 -> H):")
print(f"    v1 (X1) = {v1:+.6f}")
print(f"    v2 (X2) = {v2:+.6f}")
print(f"    v3 (X3) = {v3:+.6f}")
print(f"    v4 (X4) = {v4:+.6f}")
print(f"    theta_H = {THETA:.6f}  (umbral fijo, no actualizado)")
print()
print("  Capa salida (H -> Y):")
print(f"    w1 (H)  = {w1:+.6f}")
print(f"    theta_Y = {THETA:.6f}  (umbral fijo, no actualizado)")
print()

# Verificacion con pesos finales
print("  Verificacion con pesos finales:")
print(f"  {'Patron':<14} {'net_H':>10} {'o_H':>8} {'net_Y':>10} "
      f"{'o_Y':>8} {'d':>4} {'error':>10}")
print(f"  {'-'*66}")
for i, ((x1_p, x2_p, x3_p, x4_p), d) in enumerate(patterns, 1):
    net_H = v1*x1_p + v2*x2_p + v3*x3_p + v4*x4_p - THETA
    o_H   = sigmoid(net_H)
    net_Y = w1*o_H - THETA
    o_Y   = sigmoid(net_Y)
    print(f"  P{i} ({x1_p},{x2_p},{x3_p},{x4_p})  "
          f"{net_H:>10.6f} {o_H:>8.6f} {net_Y:>10.6f} "
          f"{o_Y:>8.6f} {d:>4}  {d-o_Y:>+10.6f}")
print()

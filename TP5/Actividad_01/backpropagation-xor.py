import numpy as np

# Función sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# Datos de entrada: P1 y P2
patrones = [
    {"x": [1, 1, 0, 1], "target": 1},  # P1 = (1,0,1) => XOR(1,0)=1
    {"x": [1, 1, 1, 0], "target": 0}   # P2 = (1,1,0) => XOR(1,1)=0
]

# Factor de aprendizaje
eta = 0.1

# Pesos iniciales
# Neurona oculta (con entradas x0, x1, x2)
W3 = np.array([1.5, 1.0, 1.0])  # w30, w31, w32

# Neurona salida (con entradas x0 y salida de oculta)
W4 = np.array([1.0, -2.0])      # w40, w41

# Iteración sobre cada patrón
for i, patron in enumerate(patrones):
    print(f"\n Patrón P{i+1}: Entrada = {patron['x'][1:]}, Target = {patron['target']}")

    x = np.array(patron["x"])  # Incluye x0 = 1
    t = patron["target"]

    # Forward 
    net3 = np.dot(W3, x[:3])  # x0, x1, x2
    y3 = sigmoid(net3)

    net4 = W4[0] * 1 + W4[1] * y3
    y4 = sigmoid(net4)

    print(f" Salida oculta (y3): {y3:.4f}")
    print(f" Salida final (y4): {y4:.4f}")

    # Backward 
    delta4 = (t - y4) * sigmoid_derivative(y4)
    delta3 = delta4 * W4[1] * sigmoid_derivative(y3)

    # Actualizar pesos salida
    W4[0] += eta * delta4 * 1     # Sesgo
    W4[1] += eta * delta4 * y3   # Desde y3

    # Actualizar pesos oculta
    W3[0] += eta * delta3 * x[0]  # Sesgo
    W3[1] += eta * delta3 * x[1]
    W3[2] += eta * delta3 * x[2]

    print(f" Nuevos pesos W3: {np.round(W3, 4)}")
    print(f" Nuevos pesos W4: {np.round(W4, 4)}")

import math

class FuncionesPrueba:
    @staticmethod
    def sphere(x):
        """
        Función Sphere
        Mínimo global: 0.0 en x = [0, 0, ...]
        Límites: -5 <= xi <= 5
        """
        return sum(xi**2 for xi in x)
    
    @staticmethod
    def schwefel(x):
        """
        Función Schwefel
        Mínimo global: 0.0 en x = [420.9687, 420.9687, ...]
        Límites: -500 <= xi <= 500
        
        Nota: La imagen menciona 10V + sum(..), con V=4189.829101. 
        Matemáticamente, lo estándar es 418.9829 * d - sum(xi * sin(sqrt(|xi|))).
        Si V = 418.9829101, entonces n*V se acerca a ese valor. 
        Usaremos la formulación equivalente generalizada a n dimensiones para que el mínimo sea 0.
        """
        n = len(x)
        # Asumimos que V de la imagen era 418.9829 para cada dimensión
        # O implementamos literalmente 4189.829101 si d=10.
        V = 418.9829101
        
        suma = sum(-xi * math.sin(math.sqrt(abs(xi))) for xi in x)
        return (V * n) + suma

    @staticmethod
    def griewank(x):
        """
        Función Griewank
        Mínimo global: 0.0 en x = [0, 0, ...]
        Límites: -600 <= xi <= 600
        """
        suma = sum((xi**2) / 4000.0 for xi in x)
        producto = 1.0
        for i, xi in enumerate(x):
            # El índice en la fórmula de la imagen empieza en 1, 
            # así que usamos (i+1)
            producto *= math.cos(xi / math.sqrt(i + 1))
            
        return 1 + suma - producto

# Diccionario para acceder a la configuración de las funciones fácilmente
CONFIG_FUNCIONES = {
    'Sphere': {
        'funcion': FuncionesPrueba.sphere,
        'limite_inf': -5.0,
        'limite_sup': 5.0,
        'dimensiones': 10
    },
    'Schwefel': {
        'funcion': FuncionesPrueba.schwefel,
        'limite_inf': -500.0,
        'limite_sup': 500.0,
        'dimensiones': 10
    },
    'Griewank': {
        'funcion': FuncionesPrueba.griewank,
        'limite_inf': -600.0,
        'limite_sup': 600.0,
        'dimensiones': 10
    }
}

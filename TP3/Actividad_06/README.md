# Actividad 6 - Búsqueda Informada: 8-Puzzle

Compara **cuatro variantes** de búsqueda informada sobre el 8-puzzle combinando dos heurísticas con dos algoritmos:

| Heurística | Descripción |
|---|---|
| **h1** – Fichas mal colocadas | Cuenta cuántas fichas no están en su posición objetivo |
| **h2** – Distancia de Manhattan | Suma de distancias horizontales + verticales de cada ficha a su meta |

| Algoritmo | Criterio de expansión |
|---|---|
| **Búsqueda Voraz** | `h(n)` — solo la heurística |
| **A\*** | `f(n) = g(n) + h(n)` — costo real + heurística |

## Métricas comparadas

- Pasos hasta la meta (longitud del camino)
- Nodos expandidos
- Nodos generados
- Tiempo de ejecución (ms)

## Ejecución

```bash
python solucion.py
```

## Resultados esperados

- h2 **domina** a h1: `h2(n) ≥ h1(n)` para todo estado `n`.
- A* con h2 expande la **menor cantidad de nodos** y garantiza optimalidad.
- A* con h1 también es óptimo pero menos eficiente.
- Búsqueda Voraz es más rápida pero **no garantiza** caminos óptimos.

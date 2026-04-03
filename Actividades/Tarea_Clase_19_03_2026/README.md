# Tarea de Clase — 19/03/2026
## Resolución de Problemas mediante Búsqueda No Informada

**Materia:** Inteligencia Artificial  
**Profesores:** Dr. Ing. Carlos Casanova — Mg. Ing. Ulises Rapallini  
**Bibliografía:** Russell & Norvig — *IA: Un Enfoque Moderno* (2da ed.), Capítulo 3

---

## Archivos

| Archivo | Descripción |
|---|---|
| `busqueda_no_informada.py` | Solución completa con BFS y DFS, salida en consola |
| `visualizacion.py` | Visualización gráfica del árbol + BFS + DFS con matplotlib |

## Ejecución

```bash
# Solución en consola
python busqueda_no_informada.py

# Visualización gráfica
python visualizacion.py
```

---

## 1. Elementos del Problema

| Elemento | Descripción |
|---|---|
| **Estado inicial** | `A` = TuCasa |
| **Estado objetivo** | `G` = Biblioteca |
| **Acciones** | Caminar por un camino que conecta dos puntos |
| **Espacio de estados** | Árbol de 19 nodos, 4 niveles de profundidad |

---

## 2. Árbol de Búsqueda — Tabla de Alias

| Alias | Nombre | Nivel |
|---|---|---|
| A | TuCasa *(raíz)* | 0 |
| B | Plaza | 1 |
| C | Parque | 1 |
| D | Supermercado | 1 |
| E | Escuela | 2 |
| F | Cine | 2 |
| **G** | **Biblioteca *(OBJETIVO)*** | **2** |
| H | Kiosco | 2 |
| I | Restaurante | 2 |
| J | Hospital | 2 |
| K | Banco | 2 |
| L | Museo | 3 |
| M | Farmacia | 3 |
| N | Café | 3 |
| O | Correo | 3 |
| P | Estadio | 4 |
| Q | Zoológico | 4 |
| R | Terminal | 4 |
| S | Liceo | 4 |

### Caminos hacia la Biblioteca

- **Camino 1:** A → B → G *(profundidad 2)*
- **Camino 2:** A → C → H → G *(profundidad 3)*
- **Camino 3:** A → D → J → G *(profundidad 3)*

---

## 3. BFS — Resultado

**Nodos explorados (orden):** A, B, C, D, E, F, **G** *(7 nodos)*  
**Camino encontrado:** `A → B → G` *(2 pasos)*  
**✅ BFS garantiza el camino más corto**

---

## 4. DFS — Resultado

**Nodos explorados (orden):** A, B, E, L, P, F, M, Q, **G** *(9 nodos)*  
**Camino encontrado:** `A → B → G` *(2 pasos)*  
**⚠️ DFS no garantiza el camino más corto**

---

## Comparación

| Criterio | BFS | DFS |
|---|---|---|
| Nodos explorados | 7 | 9 |
| Pasos del camino | 2 | 2 |
| ¿Camino óptimo? | ✅ Sí | ❌ No garantiza |
| Estructura | Cola (FIFO) | Pila (LIFO) |
| Estrategia | Por niveles | Por ramas |

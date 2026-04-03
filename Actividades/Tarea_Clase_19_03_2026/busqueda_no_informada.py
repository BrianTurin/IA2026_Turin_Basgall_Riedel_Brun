"""
=============================================================================
TAREA DE CLASE - 19/03/2026
Resolución de Problemas mediante Búsqueda No Informada
Materia: Inteligencia Artificial
Profesores: Dr. Ing. Carlos Casanova - Mg. Ing. Ulises Rapallini
=============================================================================

PROBLEMA:
  Encontrar un camino desde TuCasa hasta la Biblioteca en una ciudad X
  utilizando métodos de búsqueda no informada (BFS y DFS).

ELEMENTOS DEL PROBLEMA:
  - Estado inicial  : TuCasa (A)
  - Estado objetivo : Biblioteca (G)
  - Acciones        : Caminar por un camino que conecta dos puntos
  - Espacio estados : Grafo no dirigido con los puntos de la ciudad
=============================================================================
"""

from collections import deque

# ---------------------------------------------------------------------------
# 1. DEFINICIÓN DEL ÁRBOL DE BÚSQUEDA
#    Cada nodo tiene un alias (letra) y un nombre completo.
#    El árbol está representado como un diccionario de adyacencia.
#
#    NIVEL 0 (raíz):
#       A = TuCasa
#
#    NIVEL 1 (3 caminos desde la raíz):
#       B = Plaza
#       C = Parque
#       D = Supermercado
#
#    NIVEL 2 (2-3 alternativas por nodo intermedio):
#       E = Escuela        (desde B-Plaza)
#       F = Cine           (desde B-Plaza)
#       G = Biblioteca     (desde B-Plaza) ← OBJETIVO (camino 1)
#       H = Kiosco         (desde C-Parque)
#       I = Restaurante    (desde C-Parque)
#       J = Hospital       (desde D-Supermercado)
#       K = Banco          (desde D-Supermercado)
#
#    NIVEL 3:
#       L = Museo          (desde E-Escuela)
#       M = Farmacia       (desde F-Cine)
#       G = Biblioteca     (desde H-Kiosco)   ← OBJETIVO (camino 2)
#       N = Café           (desde I-Restaurante)
#       G = Biblioteca     (desde J-Hospital) ← OBJETIVO (camino 3)
#       O = Correo         (desde K-Banco)
#
#    NIVEL 4:
#       P = Estadio        (desde L-Museo)
#       Q = Zoológico      (desde M-Farmacia)
#       R = Terminal       (desde N-Café)
#       S = Liceo          (desde O-Correo)
# ---------------------------------------------------------------------------

# Alias → Nombre completo
NOMBRES = {
    "A": "TuCasa",
    "B": "Plaza",
    "C": "Parque",
    "D": "Supermercado",
    "E": "Escuela",
    "F": "Cine",
    "G": "Biblioteca",   # ← OBJETIVO
    "H": "Kiosco",
    "I": "Restaurante",
    "J": "Hospital",
    "K": "Banco",
    "L": "Museo",
    "M": "Farmacia",
    "N": "Café",
    "O": "Correo",
    "P": "Estadio",
    "Q": "Zoológico",
    "R": "Terminal",
    "S": "Liceo",
}

# Árbol de búsqueda (lista de adyacencia — árbol, sin ciclos)
# Sólo se definen los hijos de cada nodo para preservar la estructura de árbol.
ARBOL = {
    "A": ["B", "C", "D"],          # Nivel 0 → Nivel 1  (3 hijos)
    "B": ["E", "F", "G"],          # Nivel 1 → Nivel 2  (3 hijos) — G = camino 1
    "C": ["H", "I"],               # Nivel 1 → Nivel 2  (2 hijos)
    "D": ["J", "K"],               # Nivel 1 → Nivel 2  (2 hijos)
    "E": ["L"],                    # Nivel 2 → Nivel 3  (1 hijo)
    "F": ["M"],                    # Nivel 2 → Nivel 3  (1 hijo)
    "G": [],                       # OBJETIVO — hoja
    "H": ["G"],                    # Nivel 2 → Nivel 3  (1 hijo) — G = camino 2
    "I": ["N"],                    # Nivel 2 → Nivel 3  (1 hijo)
    "J": ["G"],                    # Nivel 2 → Nivel 3  (1 hijo) — G = camino 3
    "K": ["O"],                    # Nivel 2 → Nivel 3  (1 hijo)
    "L": ["P"],                    # Nivel 3 → Nivel 4
    "M": ["Q"],                    # Nivel 3 → Nivel 4
    "N": ["R"],                    # Nivel 3 → Nivel 4
    "O": ["S"],                    # Nivel 3 → Nivel 4
    "P": [], "Q": [], "R": [], "S": [],  # Hojas nivel 4
}

INICIO   = "A"   # TuCasa
OBJETIVO = "G"   # Biblioteca


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

def nombre(nodo: str) -> str:
    """Devuelve 'Alias (Nombre completo)'."""
    return f"{nodo} ({NOMBRES[nodo]})"


def reconstruir_camino(padres: dict, objetivo: str) -> list:
    """Reconstruye el camino desde la raíz hasta el objetivo usando el dict de padres."""
    camino = []
    nodo = objetivo
    while nodo is not None:
        camino.append(nodo)
        nodo = padres[nodo]
    camino.reverse()
    return camino


def imprimir_camino(camino: list) -> str:
    """Formatea el camino como cadena legible."""
    return " → ".join(f"{n}({NOMBRES[n]})" for n in camino)


# ---------------------------------------------------------------------------
# 2. BÚSQUEDA EN ANCHURA — BFS
#    Explora nivel por nivel.  Garantiza encontrar el CAMINO MÁS CORTO
#    (en número de pasos) dentro del árbol de búsqueda.
# ---------------------------------------------------------------------------

def bfs(inicio: str, objetivo: str) -> tuple[list, list]:
    """
    Breadth-First Search sobre el árbol de búsqueda.

    Retorna:
        orden_exploracion : lista de nodos en el orden en que fueron expandidos
        camino            : lista de nodos desde inicio hasta objetivo
    """
    cola          = deque([inicio])
    padres        = {inicio: None}
    orden         = []          # orden de exploración (nodos expandidos)
    visitados     = {inicio}

    while cola:
        nodo = cola.popleft()
        orden.append(nodo)

        if nodo == objetivo:
            return orden, reconstruir_camino(padres, objetivo)

        for hijo in ARBOL.get(nodo, []):
            if hijo not in visitados:
                visitados.add(hijo)
                padres[hijo] = nodo
                cola.append(hijo)

    return orden, []   # No encontrado


# ---------------------------------------------------------------------------
# 3. BÚSQUEDA EN PROFUNDIDAD — DFS
#    Explora tan profundo como sea posible antes de retroceder.
#    Encuentra el PRIMER camino según el orden de los hijos definido.
# ---------------------------------------------------------------------------

def dfs(inicio: str, objetivo: str) -> tuple[list, list]:
    """
    Depth-First Search sobre el árbol de búsqueda (versión iterativa con pila).

    Retorna:
        orden_exploracion : lista de nodos en el orden en que fueron expandidos
        camino            : lista de nodos desde inicio hasta objetivo
    """
    pila      = [(inicio, [inicio])]   # (nodo_actual, camino_hasta_aqui)
    visitados = set()
    orden     = []

    while pila:
        nodo, camino_actual = pila.pop()

        if nodo in visitados:
            continue
        visitados.add(nodo)
        orden.append(nodo)

        if nodo == objetivo:
            return orden, camino_actual

        # Añadir hijos en orden inverso para explorar primero el primero definido
        for hijo in reversed(ARBOL.get(nodo, [])):
            if hijo not in visitados:
                pila.append((hijo, camino_actual + [hijo]))

    return orden, []   # No encontrado


# ---------------------------------------------------------------------------
# VISUALIZACIÓN DEL ÁRBOL (texto)
# ---------------------------------------------------------------------------

def dibujar_arbol():
    """Imprime el árbol de búsqueda en formato jerárquico."""
    separador = "=" * 70
    print(separador)
    print("  ÁRBOL DE BÚSQUEDA — Ciudad X")
    print(separador)

    arbol_visual = """
    NIVEL 0 (Raíz):
         A(TuCasa)
        /    |    \\
       /     |     \\
    NIVEL 1:
    B(Plaza)  C(Parque)  D(Supermercado)
    / | \\      |  \\        |    \\
    /  |  \\     |   \\       |     \\
    NIVEL 2:
E(Escuela) F(Cine) G(Biblioteca)★  H(Kiosco) I(Restaurante)  J(Hospital) K(Banco)
    |          |                      |            |               |           |
    NIVEL 3:
L(Museo)  M(Farmacia)           G(Biblioteca)★  N(Café)     G(Biblioteca)★ O(Correo)
    |          |                                    |                          |
    NIVEL 4:
P(Estadio) Q(Zoológico)                       R(Terminal)               S(Liceo)

    ★ = Nodo objetivo (Biblioteca = G)

    CAMINOS hacia la Biblioteca:
      Camino 1: A → B → G           (profundidad 2)
      Camino 2: A → C → H → G       (profundidad 3)
      Camino 3: A → D → J → G       (profundidad 3)
"""
    print(arbol_visual)


def dibujar_arbol_con_alias():
    """Imprime la tabla de alias de todos los nodos."""
    separador = "=" * 70
    print(separador)
    print("  TABLA DE ALIAS — Nodos del árbol")
    print(separador)
    print(f"  {'Alias':<8} {'Nombre':<20} {'Nivel'}")
    print("  " + "-" * 38)
    niveles = {
        "A": 0,
        "B": 1, "C": 1, "D": 1,
        "E": 2, "F": 2, "G": 2, "H": 2, "I": 2, "J": 2, "K": 2,
        "L": 3, "M": 3, "N": 3, "O": 3,
        "P": 4, "Q": 4, "R": 4, "S": 4,
    }
    for alias, nombre_nodo in NOMBRES.items():
        objetivo_marker = "  ← OBJETIVO" if alias == OBJETIVO else ""
        print(f"  {alias:<8} {nombre_nodo:<20} {niveles.get(alias, '?')}{objetivo_marker}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    separador_grueso = "=" * 70
    separador_fino   = "-" * 70

    print()
    print(separador_grueso)
    print("  TAREA DE CLASE — 19/03/2026")
    print("  Resolución de Problemas mediante Búsqueda No Informada")
    print("  Materia: Inteligencia Artificial")
    print("  Profesores: Dr. Ing. Carlos Casanova - Mg. Ing. Ulises Rapallini")
    print(separador_grueso)

    # ── Árbol de búsqueda ──────────────────────────────────────────────────
    dibujar_arbol_con_alias()
    dibujar_arbol()

    # ── BFS ───────────────────────────────────────────────────────────────
    print(separador_grueso)
    print("  PUNTO 3 — BÚSQUEDA EN ANCHURA (BFS - Breadth-First Search)")
    print(separador_grueso)
    print()
    print("  Concepto: Explora todos los nodos de un nivel antes de avanzar")
    print("  al siguiente. Utiliza una COLA (FIFO). Garantiza el camino más")
    print("  corto en grafos/árboles no ponderados.")
    print()

    orden_bfs, camino_bfs = bfs(INICIO, OBJETIVO)

    print("  Orden de exploración (nodos expandidos):")
    print()
    for i, nodo in enumerate(orden_bfs, 1):
        marca = "  ★ OBJETIVO ENCONTRADO" if nodo == OBJETIVO else ""
        print(f"    Paso {i:>2}: {nombre(nodo)}{marca}")

    print()
    print(separador_fino)
    print(f"  Total nodos explorados : {len(orden_bfs)}")
    print(f"  Camino encontrado      : {imprimir_camino(camino_bfs)}")
    print(f"  Longitud del camino    : {len(camino_bfs) - 1} paso(s)")
    print(separador_fino)
    print()
    print("  EXPLICACIÓN BFS paso a paso:")
    print()
    print("  Cola inicial: [A(TuCasa)]")
    print()
    print("  Paso 1: Extraer A → Expandir → Agregar B, C, D a la cola")
    print("          Cola: [B(Plaza), C(Parque), D(Supermercado)]")
    print()
    print("  Paso 2: Extraer B(Plaza) → Expandir → Agregar E, F, G a la cola")
    print("          Cola: [C(Parque), D(Supermercado), E(Escuela), F(Cine), G(Biblioteca)]")
    print()
    print("  Paso 3: Extraer C(Parque) → Expandir → Agregar H, I a la cola")
    print("          Cola: [D(Supermercado), E(Escuela), F(Cine), G(Biblioteca), H(Kiosco), I(Restaurante)]")
    print()
    print("  Paso 4: Extraer D(Supermercado) → Expandir → Agregar J, K a la cola")
    print("          Cola: [E(Escuela), F(Cine), G(Biblioteca), H(Kiosco), I(Restaurante), J(Hospital), K(Banco)]")
    print()
    print("  Paso 5: Extraer E(Escuela) → Expandir → Agregar L a la cola")
    print("          Cola: [F(Cine), G(Biblioteca), H(Kiosco), I(Restaurante), J(Hospital), K(Banco), L(Museo)]")
    print()
    print("  Paso 6: Extraer F(Cine) → Expandir → Agregar M a la cola")
    print("          Cola: [G(Biblioteca), H(Kiosco), I(Restaurante), J(Hospital), K(Banco), L(Museo), M(Farmacia)]")
    print()
    print("  Paso 7: Extraer G(Biblioteca) → ¡OBJETIVO ENCONTRADO!")
    print()
    print(f"  ✅ BFS encuentra el camino MÁS CORTO: {imprimir_camino(camino_bfs)}")
    print(f"     ({len(camino_bfs)-1} paso(s) desde TuCasa hasta Biblioteca)")

    print()

    # ── DFS ───────────────────────────────────────────────────────────────
    print(separador_grueso)
    print("  PUNTO 4 — BÚSQUEDA EN PROFUNDIDAD (DFS - Depth-First Search)")
    print(separador_grueso)
    print()
    print("  Concepto: Explora tan profundo como sea posible por cada rama")
    print("  antes de retroceder (backtracking). Utiliza una PILA (LIFO).")
    print("  No garantiza el camino más corto, pero usa menos memoria.")
    print()

    orden_dfs, camino_dfs = dfs(INICIO, OBJETIVO)

    print("  Orden de exploración (nodos expandidos):")
    print()
    for i, nodo in enumerate(orden_dfs, 1):
        marca = "  ★ OBJETIVO ENCONTRADO" if nodo == OBJETIVO else ""
        print(f"    Paso {i:>2}: {nombre(nodo)}{marca}")

    print()
    print(separador_fino)
    print(f"  Total nodos explorados : {len(orden_dfs)}")
    print(f"  Camino encontrado      : {imprimir_camino(camino_dfs)}")
    print(f"  Longitud del camino    : {len(camino_dfs) - 1} paso(s)")
    print(separador_fino)
    print()
    print("  EXPLICACIÓN DFS paso a paso:")
    print()
    print("  Pila inicial: [A(TuCasa)]")
    print()
    print("  Paso 1: Sacar A → Expandir → Agregar D, C, B (orden inverso para PILA)")
    print("          Pila: [D(Supermercado), C(Parque), B(Plaza)]")
    print()
    print("  Paso 2: Sacar B(Plaza) → Expandir → Agregar G, F, E (orden inverso)")
    print("          Pila: [D(Supermercado), C(Parque), G(Biblioteca), F(Cine), E(Escuela)]")
    print()
    print("  Paso 3: Sacar E(Escuela) → Expandir → Agregar L")
    print("          Pila: [D(Supermercado), C(Parque), G(Biblioteca), F(Cine), L(Museo)]")
    print()
    print("  Paso 4: Sacar L(Museo) → Expandir → Agregar P")
    print("          Pila: [D(Supermercado), C(Parque), G(Biblioteca), F(Cine), P(Estadio)]")
    print()
    print("  Paso 5: Sacar P(Estadio) → hoja, sin hijos → backtrack")
    print("          Pila: [D(Supermercado), C(Parque), G(Biblioteca), F(Cine)]")
    print()
    print("  Paso 6: Sacar F(Cine) → Expandir → Agregar M")
    print("          Pila: [D(Supermercado), C(Parque), G(Biblioteca), M(Farmacia)]")
    print()
    print("  Paso 7: Sacar M(Farmacia) → Expandir → Agregar Q")
    print("          Pila: [D(Supermercado), C(Parque), G(Biblioteca), Q(Zoológico)]")
    print()
    print("  Paso 8: Sacar Q(Zoológico) → hoja → backtrack")
    print("          Pila: [D(Supermercado), C(Parque), G(Biblioteca)]")
    print()
    print("  Paso 9: Sacar G(Biblioteca) → ¡OBJETIVO ENCONTRADO!")
    print()
    print(f"  ✅ DFS encuentra el PRIMER camino en su recorrido: {imprimir_camino(camino_dfs)}")
    print(f"     ({len(camino_dfs)-1} paso(s) desde TuCasa hasta Biblioteca)")

    print()

    # ── COMPARACIÓN FINAL ─────────────────────────────────────────────────
    print(separador_grueso)
    print("  COMPARACIÓN FINAL: BFS vs DFS")
    print(separador_grueso)
    print()
    print(f"  {'Criterio':<30} {'BFS':^20} {'DFS':^20}")
    print("  " + "-" * 70)
    print(f"  {'Nodos explorados':<30} {len(orden_bfs):^20} {len(orden_dfs):^20}")
    print(f"  {'Pasos del camino':<30} {len(camino_bfs)-1:^20} {len(camino_dfs)-1:^20}")
    print(f"  {'Camino óptimo?':<30} {'✅ Sí (más corto)':^20} {'❌ No garantiza':^20}")
    print(f"  {'Estructura':<30} {'Cola (FIFO)':^20} {'Pila (LIFO)':^20}")
    print(f"  {'Estrategia':<30} {'Por niveles':^20} {'Por ramas':^20}")
    print()
    print(f"  Camino BFS: {imprimir_camino(camino_bfs)}")
    print(f"  Camino DFS: {imprimir_camino(camino_dfs)}")
    print()
    print(separador_grueso)
    print("  FIN DE LA TAREA — 19/03/2026")
    print(separador_grueso)
    print()


if __name__ == "__main__":
    main()

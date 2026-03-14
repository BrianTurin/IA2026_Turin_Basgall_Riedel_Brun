# Actividad 02 — Agente Reactivo Aspiradora

## Descripcion

Implementacion y comparacion de dos variantes del **agente aspiradora** sobre una grilla bidimensional (TP1 - Ejercicio B).

Se programan ambos agentes sobre el **mismo entorno inicial** para que los resultados sean comparables.

---

## Parte 1 — Agente Reactivo Simple

El agente toma decisiones basandose **unicamente en la percepcion del instante actual**: posicion y estado de la celda (sucia/limpia). No guarda ninguna informacion sobre el entorno.

### Reglas condicion-accion

| Percepcion             | Accion  |
|------------------------|---------|
| Celda actual sucia     | LIMPIAR |
| Celda actual limpia    | MOVER   |

### Movimiento

Patron fijo izquierda → derecha, fila por fila (top-down). Al llegar a la ultima celda, reinicia desde el origen porque **no sabe si quedan celdas sucias**.

### Limitaciones

- No sabe cuales celdas ya limpio.
- Puede recorrer celdas repetidas.
- Solo se detiene cuando el entorno esta 100% limpio (verificacion externa en cada paso).

---

## Parte 2 — Agente Reactivo con Estados

El agente mantiene un **mapa interno** del entorno que actualiza en cada paso. Este estado le permite tomar decisiones mas informadas.

### Estado interno

```python
mapa: dict[(fila, col) -> 'sucia' | 'limpia']
```

### Logica de decision

1. Si la celda actual esta sucia → **LIMPIAR**.
2. Si ya visito todas las celdas y ninguna aparece como sucia en el mapa → **TERMINADO**.
3. En caso contrario → **MOVER** hacia la celda prioritaria.

### Navegacion

- Busca primero las **celdas sucias conocidas** mas cercanas (distancia Manhattan).
- Si no hay sucias conocidas, explora la **celda no visitada** mas cercana.

### Ventajas respecto al agente simple

- Detecta la finalizacion **sin volver a recorrer toda la grilla**.
- Reduce significativamente los movimientos al ir directamente a las celdas de interes.
- Menor cantidad total de pasos para completar la misma tarea.

---

## Ejecucion

```bash
python solucion.py
```

Se solicitaran parametros de configuracion (o se usan los valores por defecto con Enter):

| Parametro           | Default |
|---------------------|---------|
| Filas               | 4       |
| Columnas            | 5       |
| Prob. de suciedad   | 0.6     |
| Semilla aleatoria   | 42      |
| Velocidad (seg/paso)| 0.2     |

---

## Ejemplo de visualizacion

```
  ┌──────────────────────────────────────────────────────┐
  │ AGENTE REACTIVO CON ESTADOS  —  Parte 2              │
  ├──────────────────────────────────────────────────────┤
  │  [S]   [S]   .    .    .                             │
  │  [A]   [S]   [S]  .    .                             │
  │   .    .    .    .    .                              │
  │   .    .    .    .    .                              │
  ├──────────────────────────────────────────────────────┤
  │  Paso:  8    Accion: LIMPIAR     Sucias: 4           │
  │  Limpiezas: 4      Movimientos: 4                    │
  └──────────────────────────────────────────────────────┘

  Leyenda:  [A] Aspiradora    [S] Sucia    [ . ] Limpia
```

---

## Comparacion de resultados (ejemplo con grilla 4×5, semilla 42)

```
  ╔══════════════════════════════════════════════════════╗
  ║              COMPARACION DE RESULTADOS               ║
  ╠══════════════════════════════════════════════════════╣
  ║  Metrica                         Simple       Est.  ║
  ╠══════════════════════════════════════════════════════╣
  ║  Celdas sucias (inicial)             12          12  ║
  ║  Pasos totales                       32          24  ║
  ║  Limpiezas realizadas                12          12  ║
  ║  Movimientos realizados              20          12  ║
  ║  Tarea completada                    Si          Si  ║
  ╚══════════════════════════════════════════════════════╝

  → El agente con estados necesito 8 pasos menos (25.0% mas eficiente).
```

---

## Integrantes

- Turin
- Basgall
- Riedel
- Brun

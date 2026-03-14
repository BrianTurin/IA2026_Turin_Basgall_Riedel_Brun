"""
Actividad 02 - Agente Reactivo Aspiradora

Descripcion:
    Implementacion y comparacion de dos variantes del agente aspiradora
    sobre una grilla bidimensional de habitaciones:

    Parte 1 - Agente Reactivo Simple:
        Toma decisiones basandose unicamente en la percepcion actual
        (posicion actual + estado de la celda: sucia/limpia).
        No posee ninguna memoria del entorno. Sus reglas son:
          - si celda sucia  -> LIMPIAR
          - si celda limpia -> MOVER  (recorrido izq->der, fila por fila)
        No sabe si ya limpio todas las habitaciones sin verificarlo.

    Parte 2 - Agente Reactivo con Estados:
        Mantiene un mapa interno de las celdas visitadas y su estado.
        Navega directamente a la celda sucia o no visitada mas cercana.
        Detecta la finalizacion de la tarea cuando todo el mapa interno
        esta limpio, sin necesidad de recorrer toda la grilla otra vez.

Autores:
    Turin, Basgall, Riedel, Brun
Materia:
    Inteligencia Artificial - 2026
"""

import random
import time
import os
import sys

# Fuerza UTF-8 en la salida para que los caracteres de dibujo se muestren bien
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ══════════════════════════════════════════════════════════════════════════════
#  ENTORNO
# ══════════════════════════════════════════════════════════════════════════════

class Entorno:
    """
    Grilla bidimensional de habitaciones que pueden estar sucias o limpias.
    Representa el mundo donde opera la aspiradora.
    """

    SUCIA  = 1
    LIMPIA = 0

    def __init__(self, filas, columnas, prob_suciedad=0.6, semilla=None):
        if semilla is not None:
            random.seed(semilla)
        self.filas     = filas
        self.columnas  = columnas
        self.grilla    = [
            [self.SUCIA if random.random() < prob_suciedad else self.LIMPIA
             for _ in range(columnas)]
            for _ in range(filas)
        ]
        self.suciedad_inicial = self.contar_sucios()

    def esta_sucio(self, fila, col):
        return self.grilla[fila][col] == self.SUCIA

    def limpiar(self, fila, col):
        self.grilla[fila][col] = self.LIMPIA

    def todo_limpio(self):
        return all(
            self.grilla[f][c] == self.LIMPIA
            for f in range(self.filas)
            for c in range(self.columnas)
        )

    def contar_sucios(self):
        return sum(
            self.grilla[f][c]
            for f in range(self.filas)
            for c in range(self.columnas)
        )

    def copia(self):
        """Devuelve una copia independiente del entorno."""
        nuevo                  = object.__new__(Entorno)
        nuevo.filas            = self.filas
        nuevo.columnas         = self.columnas
        nuevo.grilla           = [fila[:] for fila in self.grilla]
        nuevo.suciedad_inicial = self.suciedad_inicial
        return nuevo


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACION
# ══════════════════════════════════════════════════════════════════════════════

def _limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')


def _dibujar(entorno, fila_ag, col_ag, titulo, paso, accion, limpiezas, movimientos):
    """Dibuja la grilla del entorno con el agente y estadisticas del paso."""
    CELDA        = 5              # chars por celda: " [X] "
    ancho_grilla = CELDA * entorno.columnas + 2
    ancho_info   = 54
    W            = max(ancho_grilla, ancho_info)
    sep          = "-" * W
    sep_doble    = "=" * W

    print()
    print(f"  +{sep_doble}+")
    print(f"  | {titulo:<{W - 1}}|")
    print(f"  +{sep}+")

    for f in range(entorno.filas):
        fila_str = "  |"
        for c in range(entorno.columnas):
            if f == fila_ag and c == col_ag:
                fila_str += " [A] "
            elif entorno.esta_sucio(f, c):
                fila_str += " [S] "
            else:
                fila_str += "  .  "
        fila_str += " " * (W - CELDA * entorno.columnas) + "|"
        print(fila_str)

    print(f"  +{sep}+")
    linea1 = f"  Paso: {paso:<4}  Accion: {accion:<12}  Sucias: {entorno.contar_sucios()}"
    linea2 = f"  Limpiezas: {limpiezas:<5}  Movimientos: {movimientos}"
    print(f"  | {linea1:<{W - 1}}|")
    print(f"  | {linea2:<{W - 1}}|")
    print(f"  +{sep_doble}+")
    print("\n  Leyenda:  [A] Aspiradora    [S] Sucia    [ . ] Limpia")


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 1 — AGENTE REACTIVO SIMPLE
# ══════════════════════════════════════════════════════════════════════════════

class AgenteReactivoSimple:
    """
    Agente reactivo puro basado en una tabla de reglas condicion-accion.

    Percepcion : (posicion_actual, esta_sucio?)
    Reglas     : si sucio  -> LIMPIAR
                 si limpio -> MOVER
    Movimiento : izquierda a derecha, fila por fila (sin estado de direccion).

    Limitaciones respecto al agente con estados:
      - No sabe cuales habitaciones ya limpió.
      - Solo termina cuando verifica el entorno completo en cada iteracion.
      - Puede volver a pasar por celdas ya limpias.
    """

    def __init__(self, entorno):
        self.entorno     = entorno
        self.fila        = 0
        self.col         = 0
        self.limpiezas   = 0
        self.movimientos = 0

    # ── Percepcion ────────────────────────────────────────────────────────────
    def percibir(self):
        return (self.fila, self.col), self.entorno.esta_sucio(self.fila, self.col)

    # ── Tabla de reglas condicion-accion ──────────────────────────────────────
    def _regla(self, percepcion):
        _pos, sucio = percepcion
        return 'LIMPIAR' if sucio else 'MOVER'

    # ── Ejecucion de acciones ─────────────────────────────────────────────────
    def _accion_limpiar(self):
        self.entorno.limpiar(self.fila, self.col)
        self.limpiezas += 1

    def _accion_mover(self):
        """
        Avanza segun posicion actual (sin memoria de direccion previa).
        Patron: izquierda -> derecha dentro de cada fila, de arriba a abajo.
        """
        if self.col + 1 < self.entorno.columnas:
            self.col += 1
        elif self.fila + 1 < self.entorno.filas:
            self.fila += 1
            self.col   = 0
        else:
            # Llego al final; reinicia recorrido (no sabe si quedan sucias)
            self.fila, self.col = 0, 0
        self.movimientos += 1

    # ── Ciclo del agente ──────────────────────────────────────────────────────
    def paso(self):
        percepcion = self.percibir()
        accion     = self._regla(percepcion)
        if accion == 'LIMPIAR':
            self._accion_limpiar()
        else:
            self._accion_mover()
        return accion

    def ejecutar(self, max_pasos=500, velocidad=0.15):
        """Ejecuta hasta que el entorno este limpio o se agoten los pasos."""
        paso = 0
        while not self.entorno.todo_limpio() and paso < max_pasos:
            accion = self.paso()
            paso  += 1
            _limpiar_pantalla()
            _dibujar(
                self.entorno, self.fila, self.col,
                "AGENTE REACTIVO SIMPLE  —  Parte 1",
                paso, accion, self.limpiezas, self.movimientos,
            )
            time.sleep(velocidad)
        return {
            'pasos'       : paso,
            'limpiezas'   : self.limpiezas,
            'movimientos' : self.movimientos,
            'exito'       : self.entorno.todo_limpio(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 2 — AGENTE REACTIVO CON ESTADOS
# ══════════════════════════════════════════════════════════════════════════════

class AgenteReactivoConEstados:
    """
    Agente reactivo que mantiene un modelo del entorno como estado interno.

    Percepcion       : (posicion_actual, esta_sucio?)
    Estado interno   : mapa {(fila, col) -> 'sucia' | 'limpia'}

    Logica de decision:
      1. Si la celda actual esta sucia -> LIMPIAR.
      2. Si ya visito todas las celdas y ninguna esta sucia -> TERMINADO.
      3. En caso contrario -> MOVER hacia la celda prioritaria.

    Navegacion:
      - Primero va a las celdas sucias conocidas (mas cercana por distancia
        Manhattan).
      - Si no hay sucias conocidas, explora la celda no visitada mas cercana.

    Ventajas respecto al agente simple:
      - Detecta la finalizacion sin recorrer toda la grilla nuevamente.
      - Reduce movimientos al ir directo a las celdas de interes.
    """

    _SUCIA  = 'sucia'
    _LIMPIA = 'limpia'

    def __init__(self, entorno):
        self.entorno     = entorno
        self.fila        = 0
        self.col         = 0
        self.limpiezas   = 0
        self.movimientos = 0
        self.mapa        = {}    # estado interno: modelo del entorno conocido

    # ── Percepcion ────────────────────────────────────────────────────────────
    def percibir(self):
        return (self.fila, self.col), self.entorno.esta_sucio(self.fila, self.col)

    # ── Actualizacion del estado interno ──────────────────────────────────────
    def _actualizar_mapa(self, pos, sucio):
        self.mapa[pos] = self._SUCIA if sucio else self._LIMPIA

    # ── Seleccion de accion (usa percepcion + estado interno) ─────────────────
    def _seleccionar_accion(self, percepcion):
        pos, sucio = percepcion
        self._actualizar_mapa(pos, sucio)

        if sucio:
            return 'LIMPIAR'

        total_celdas = self.entorno.filas * self.entorno.columnas
        if len(self.mapa) == total_celdas:
            if all(estado == self._LIMPIA for estado in self.mapa.values()):
                return 'TERMINADO'

        return 'MOVER'

    # ── Navegacion inteligente basada en el mapa interno ─────────────────────
    def _destino(self):
        """
        Elige la proxima celda objetivo:
          1. Celda sucia conocida mas cercana (distancia Manhattan).
          2. Celda no visitada mas cercana (exploracion sistematica).
        """
        sucias = [
            (f, c) for (f, c), e in self.mapa.items()
            if e == self._SUCIA
        ]
        no_visitadas = [
            (f, c)
            for f in range(self.entorno.filas)
            for c in range(self.entorno.columnas)
            if (f, c) not in self.mapa
        ]
        candidatas = sucias + no_visitadas
        if not candidatas:
            return None
        return min(
            candidatas,
            key=lambda p: abs(p[0] - self.fila) + abs(p[1] - self.col),
        )

    def _mover_hacia(self, destino):
        """Un paso hacia el destino (prioriza movimiento vertical luego horizontal)."""
        df, dc = destino
        if   df > self.fila: self.fila += 1
        elif df < self.fila: self.fila -= 1
        elif dc > self.col:  self.col  += 1
        else:                self.col  -= 1
        self.movimientos += 1

    # ── Ciclo del agente ──────────────────────────────────────────────────────
    def paso(self):
        percepcion = self.percibir()
        accion     = self._seleccionar_accion(percepcion)

        if accion == 'LIMPIAR':
            self.entorno.limpiar(self.fila, self.col)
            self.mapa[(self.fila, self.col)] = self._LIMPIA
            self.limpiezas += 1
        elif accion == 'MOVER':
            dest = self._destino()
            if dest:
                self._mover_hacia(dest)

        return accion

    def ejecutar(self, max_pasos=500, velocidad=0.15):
        """Ejecuta hasta detectar TERMINADO o agotar los pasos."""
        paso   = 0
        accion = ''
        while accion != 'TERMINADO' and paso < max_pasos:
            accion = self.paso()
            paso  += 1
            _limpiar_pantalla()
            _dibujar(
                self.entorno, self.fila, self.col,
                "AGENTE REACTIVO CON ESTADOS  —  Parte 2",
                paso, accion, self.limpiezas, self.movimientos,
            )
            time.sleep(velocidad)
        return {
            'pasos'       : paso,
            'limpiezas'   : self.limpiezas,
            'movimientos' : self.movimientos,
            'exito'       : self.entorno.todo_limpio(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  TABLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════

def _mostrar_comparacion(r_simple, r_estados, suciedad_inicial):
    W   = 54
    sep = "=" * W
    print(f"\n  +{sep}+")
    print(f"  |{'COMPARACION DE RESULTADOS':^{W}}|")
    print(f"  +{sep}+")
    print(f"  |  {'Metrica':<30}{'Simple':>10}{'Est.':>12}  |")
    print(f"  +{sep}+")

    filas_tabla = [
        ("Celdas sucias (inicial)",
         str(suciedad_inicial),          str(suciedad_inicial)),
        ("Pasos totales",
         str(r_simple['pasos']),         str(r_estados['pasos'])),
        ("Limpiezas realizadas",
         str(r_simple['limpiezas']),     str(r_estados['limpiezas'])),
        ("Movimientos realizados",
         str(r_simple['movimientos']),   str(r_estados['movimientos'])),
        ("Tarea completada",
         "Si" if r_simple['exito']  else "No",
         "Si" if r_estados['exito'] else "No"),
    ]

    for metrica, v1, v2 in filas_tabla:
        print(f"  |  {metrica:<30}{v1:>10}{v2:>12}  |")

    print(f"  +{sep}+")

    if r_simple['pasos'] > 0 and r_simple['pasos'] != r_estados['pasos']:
        dif = r_simple['pasos'] - r_estados['pasos']
        pct = abs(dif) / r_simple['pasos'] * 100
        if dif > 0:
            print(f"\n  -> El agente con estados necesito {dif} pasos menos ({pct:.1f}% mas eficiente).")
        else:
            print(f"\n  -> El agente simple necesito {-dif} pasos menos en este caso.")
    else:
        print("\n  -> Ambos agentes usaron la misma cantidad de pasos.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 58)
    print("  Actividad 02  --  Agente Reactivo Aspiradora")
    print("  Inteligencia Artificial 2026")
    print("  Turin  Basgall  Riedel  Brun")
    print("=" * 58)

    print("\n  Configuracion del entorno (Enter = valor por defecto):\n")
    try:
        filas     = int(  input("    Filas                [default 4 ]: ").strip() or 4)
        columnas  = int(  input("    Columnas             [default 5 ]: ").strip() or 5)
        prob      = float(input("    Prob. de suciedad    [default 0.6]: ").strip() or 0.6)
        semilla   = int(  input("    Semilla aleatoria    [default 42]: ").strip() or 42)
        velocidad = float(input("    Velocidad (seg/paso) [default 0.2]: ").strip() or 0.2)
    except ValueError:
        print("  Valor invalido detectado. Se usan los valores por defecto.")
        filas, columnas, prob, semilla, velocidad = 4, 5, 0.6, 42, 0.2

    # Mismo entorno inicial para los dos agentes
    entorno_base    = Entorno(filas, columnas, prob_suciedad=prob, semilla=semilla)
    entorno_simple  = entorno_base.copia()
    entorno_estados = entorno_base.copia()

    print(f"\n  Entorno generado: {filas}x{columnas}  |  Celdas sucias: {entorno_base.suciedad_inicial} / {filas * columnas}")

    # ── Parte 1: Agente Reactivo Simple ───────────────────────────────────────
    input("\n  [Enter] para iniciar PARTE 1: Agente Reactivo Simple...")
    agente_simple = AgenteReactivoSimple(entorno_simple)
    res_simple    = agente_simple.ejecutar(max_pasos=500, velocidad=velocidad)

    # ── Parte 2: Agente Reactivo con Estados ──────────────────────────────────
    input("\n  [Enter] para iniciar PARTE 2: Agente Reactivo con Estados...")
    agente_estados = AgenteReactivoConEstados(entorno_estados)
    res_estados    = agente_estados.ejecutar(max_pasos=500, velocidad=velocidad)

    # ── Comparacion ───────────────────────────────────────────────────────────
    input("\n  [Enter] para ver la comparacion de resultados...")
    _limpiar_pantalla()
    _mostrar_comparacion(res_simple, res_estados, entorno_base.suciedad_inicial)


if __name__ == '__main__':
    main()

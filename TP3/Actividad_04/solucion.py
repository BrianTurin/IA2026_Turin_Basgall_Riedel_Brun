# -*- coding: utf-8 -*-
"""
Actividad 4 - Busqueda Informada: Romania (Arad -> Bucarest)
============================================================
Punto de entrada principal. Ejecuta la comparacion entre:
  - Busqueda Voraz (Greedy Best-First Search)
  - A* (A-star)
usando la heuristica SLD (distancia en linea recta a Bucarest).

Delegamos la logica completa a busqueda_informada_rumania.py
"""
from busqueda_informada_rumania import comparar

if __name__ == '__main__':
    comparar('Arad', 'Bucarest', repeticiones=5000)

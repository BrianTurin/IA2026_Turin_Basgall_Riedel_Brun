# Actividad A – Diagnóstico de Enfermedades

## Archivos
- `../experto.pl` — Shell del sistema experto (compartido por todos los ejercicios)
- `BaseConocimientos1.pl` — Base de conocimientos médica

## Cómo ejecutar en SWI-Prolog

```prolog
?- consult('../experto.pl').
?- consult('BaseConocimientos1.pl').
?- consulta.
```

## Consultas requeridas y resultados

### Consulta i — temperatura alta, dolor de cabeza, cuerpo cortado
**Diagnóstico: gripe**

Síntomas confirmados:
- el paciente tiene cuerpo cortado → si
- el paciente tiene dolor de cabeza → si
- el paciente tiene temperatura alta → si

### Consulta ii — dolor en articulaciones, temblor violento, escalofríos
**Diagnóstico: malaria**

Síntomas confirmados:
- el paciente tiene temperatura alta → si
- el paciente tiene dolor en las articulaciones → si
- el paciente tiembla violentamente → si
- el paciente tiene escalofrios → si

### Consulta iii — dolor de cabeza, estornudos, dolor en articulaciones
**Diagnóstico: influenza**

Síntomas confirmados:
- el paciente tiene dolor en las articulaciones → si
- el paciente tiene mucho estornudo → si
- el paciente tiene dolor de cabeza → si

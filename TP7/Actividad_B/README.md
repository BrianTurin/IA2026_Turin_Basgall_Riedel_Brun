# Actividad B – Sistema Experto Automotriz

## Archivos
- `../experto.pl` — Shell del sistema experto (compartido)
- `BaseConocimientos2.pl` — Base de conocimientos automotriz

## Cómo ejecutar en SWI-Prolog

```prolog
?- consult('../experto.pl').
?- consult('BaseConocimientos2.pl').
?- consulta.
```

## 4 Consultas requeridas y resultados

### Consulta 1 — Banda engrasada que rechina
**Diagnóstico: banda del alternador defectuosa (engrasada)**

| Pregunta | Respuesta |
|---|---|
| la banda del alternador esta engrasada | si |
| la banda del alternador rechina al girar | si |

### Consulta 2 — Batería con voltaje bajo y motor apagado
**Diagnóstico: bateria defectuosa**

| Pregunta | Respuesta |
|---|---|
| las luces y el ventilador estan encendidos | si |
| el motor esta apagado | si |
| el voltaje de la bateria es menor a 10.5 volts | si |

### Consulta 3 — Luces que se intensifican al acelerar
**Diagnóstico: regulador defectuoso**

| Pregunta | Respuesta |
|---|---|
| el motor esta en marcha | si |
| las luces estan encendidas | si |
| las luces se intensifican al acelerar | si |

### Consulta 4 — Sistema de frenos con tirón y ruedas que rechinan
**Diagnóstico: sistema de frenos defectuoso**

| Pregunta | Respuesta |
|---|---|
| el pedal del freno esta duro | si |
| al frenar se produce un tiron lateral | si |
| las ruedas rechinan al frenar | si |

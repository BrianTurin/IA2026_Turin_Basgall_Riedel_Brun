# Actividad C – Identificación de Animales

## Archivos
- `../experto.pl` — Shell del sistema experto (compartido)
- `BaseConocimientos3.pl` — Base de conocimientos de animales

## Cómo ejecutar en SWI-Prolog

```prolog
?- consult('../experto.pl').
?- consult('BaseConocimientos3.pl').
?- consulta.
```

## 4 Consultas requeridas y resultados

### Consulta 1 — Mamífero carnívoro de color leonado con manchas
**Diagnóstico: cheeta**

| Pregunta | Respuesta |
|---|---|
| el animal es mamifero | si |
| el animal es carnivoro | si |
| el animal tiene color leonado | si |
| el animal tiene puntos negros | si |

### Consulta 2 — Mamífero carnívoro de color leonado con rayas
**Diagnóstico: tigre**

| Pregunta | Respuesta |
|---|---|
| el animal es mamifero | si |
| el animal es carnivoro | si |
| el animal tiene color leonado | si |
| el animal tiene puntos negros | no |
| el animal tiene rayas negras | si |

### Consulta 3 — Pájaro que no vuela y nada, blanco con negro
**Diagnóstico: pinguino**

| Pregunta | Respuesta |
|---|---|
| el animal es pajaro | si |
| el animal no vuela | si |
| el animal sabe nadar | si |
| el animal es blanco con negro | si |

### Consulta 4 — Ungulado con rayas negras
**Diagnóstico: zebra**

| Pregunta | Respuesta |
|---|---|
| el animal es ungulado | si |
| el animal tiene rayas negras | si |

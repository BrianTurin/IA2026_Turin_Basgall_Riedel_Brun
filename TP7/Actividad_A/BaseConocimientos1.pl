/* =============================================================
   BaseConocimientos1.pl  –  Diagnostico medico
   =============================================================
   Dominio: enfermedades comunes.
   Formato: conocimiento(Diagnostico, [Sintoma1, Sintoma2, ...]).

   Uso junto con el shell:
       ?- consult('../experto.pl').
       ?- consult('BaseConocimientos1.pl').
       ?- consulta.
   ============================================================= */

conocimiento(sarampion, [
    'el paciente esta cubierto de puntos',
    'el paciente tiene temperatura alta',
    'el paciente tiene ojos rojos',
    'el paciente tiene tos seca'
]).

conocimiento(influenza, [
    'el paciente tiene dolor en las articulaciones',
    'el paciente tiene mucho estornudo',
    'el paciente tiene dolor de cabeza'
]).

conocimiento(malaria, [
    'el paciente tiene temperatura alta',
    'el paciente tiene dolor en las articulaciones',
    'el paciente tiembla violentamente',
    'el paciente tiene escalofrios'
]).

conocimiento(gripe, [
    'el paciente tiene cuerpo cortado',
    'el paciente tiene dolor de cabeza',
    'el paciente tiene temperatura alta'
]).

conocimiento(tifoidea, [
    'el paciente tiene falta de apetito',
    'el paciente tiene temperatura alta',
    'el paciente tiene dolor abdominal',
    'el paciente tiene dolor de cabeza',
    'el paciente tiene diarrea'
]).

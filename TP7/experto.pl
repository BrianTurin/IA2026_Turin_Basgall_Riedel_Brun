/* =============================================================
   experto.pl  –  Shell generico del Sistema Experto
   =============================================================
   Funciona con cualquier base de conocimientos que defina:
       conocimiento(Diagnostico, [Sintoma1, Sintoma2, ...]).

   Uso:
       ?- consult('experto.pl').
       ?- consult('Actividad_X/BaseConocimientosX.pl').
       ?- consulta.

   El usuario responde  si / no / porque  a cada pregunta.
   Al final el sistema ofrece justificar el diagnostico.
   ============================================================= */

:- dynamic conocido/1.

/* --- Punto de entrada principal --- */

consulta :-
    haz_diagnostico(X),
    escribe_diagnostico(X),
    ofrece_explicacion_diagnostico(X),
    clean_scratchpad.

consulta :-
    write('No hay suficiente conocimiento para elaborar un diagnostico.'), nl,
    clean_scratchpad.

/* --- Motor de inferencia --- */

haz_diagnostico(Diagnostico) :-
    obten_hipotesis_y_sintomas(Diagnostico, ListaDeSintomas),
    prueba_presencia_de(Diagnostico, ListaDeSintomas).

obten_hipotesis_y_sintomas(Diagnostico, ListaDeSintomas) :-
    conocimiento(Diagnostico, ListaDeSintomas).

% Caso base: lista de sintomas vacia => hipotesis confirmada
prueba_presencia_de(_, []).

% Caso recursivo: verificar cada sintoma de la lista
prueba_presencia_de(Diagnostico, [Cabeza | Cola]) :-
    prueba_verdad_de(Diagnostico, Cabeza),
    prueba_presencia_de(Diagnostico, Cola).

% Un sintoma es verdadero si ya fue confirmado antes
prueba_verdad_de(_, Sintoma) :- conocido(Sintoma).

% Un sintoma es verdadero si no fue negado Y el usuario responde "si"
prueba_verdad_de(Diagnostico, Sintoma) :-
    not(conocido(is_false(Sintoma))),
    pregunta_sobre(Diagnostico, Sintoma, Respuesta),
    Respuesta = si.

/* --- Interaccion con el usuario --- */

pregunta_sobre(Diagnostico, Sintoma, Respuesta) :-
    write('Es verdad que '), write(Sintoma), write('? '),
    read(RespuestaUsuario),
    process(Diagnostico, Sintoma, RespuestaUsuario, Respuesta).

% Procesar respuesta "si": guardar sintoma como conocido
process(_, Sintoma, si, si) :-
    asserta(conocido(Sintoma)).

% Procesar respuesta "no": marcar sintoma como falso
process(_, Sintoma, no, no) :-
    asserta(conocido(is_false(Sintoma))).

% Procesar respuesta "porque": explicar que hipotesis se esta evaluando
process(Diagnostico, Sintoma, porque, Respuesta) :-
    nl,
    write('Estoy investigando la hipotesis: '), write(Diagnostico), write('.'), nl,
    write('Para esto necesito saber si '), write(Sintoma), write('.'), nl,
    pregunta_sobre(Diagnostico, Sintoma, Respuesta).

% Respuesta invalida: pedir de nuevo
process(Diagnostico, Sintoma, _, Respuesta) :-
    nl,
    write('Debes contestar si, no o porque.'), nl,
    pregunta_sobre(Diagnostico, Sintoma, Respuesta).

/* --- Salida del diagnostico --- */

escribe_diagnostico(Diagnostico) :-
    nl,
    write('El diagnostico es: '), write(Diagnostico), write('.'), nl.

/* --- Explicacion del diagnostico --- */

ofrece_explicacion_diagnostico(Diagnostico) :-
    pregunta_si_necesita_explicacion(Respuesta),
    actua_consecuentemente(Diagnostico, Respuesta).

pregunta_si_necesita_explicacion(Respuesta) :-
    write('Quieres que justifique este diagnostico? '),
    read(RespuestaUsuario),
    asegura_si_o_no(RespuestaUsuario, Respuesta).

asegura_si_o_no(si, si).
asegura_si_o_no(no, no).
asegura_si_o_no(_, Respuesta) :-
    write('Debes contestar si o no.'), nl,
    pregunta_si_necesita_explicacion(Respuesta).

actua_consecuentemente(_, no).

actua_consecuentemente(Diagnostico, si) :-
    conocimiento(Diagnostico, ListaDeSintomas),
    write('Se determino este diagnostico porque se encontraron los siguientes sintomas:'), nl,
    escribe_lista_de_sintomas(ListaDeSintomas).

escribe_lista_de_sintomas([]).
escribe_lista_de_sintomas([Cabeza | Cola]) :-
    write('  - '), write(Cabeza), nl,
    escribe_lista_de_sintomas(Cola).

/* --- Limpieza del estado dinamico entre consultas --- */

% Elimina todos los hechos dinamicos acumulados en la sesion
clean_scratchpad :- retract(conocido(_)), fail.
clean_scratchpad.

/* --- Negacion por fallo --- */
not(X) :- X, !, fail.
not(_).

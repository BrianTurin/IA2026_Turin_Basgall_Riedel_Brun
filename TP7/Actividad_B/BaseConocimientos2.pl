/* =============================================================
   BaseConocimientos2.pl  –  Diagnostico automotriz
   =============================================================
   Dominio: fallas electricas y mecanicas del automovil.
   Formato: conocimiento(Diagnostico, [Sintoma1, Sintoma2, ...]).

   Uso junto con el shell:
       ?- consult('../experto.pl').
       ?- consult('BaseConocimientos2.pl').
       ?- consulta.
   ============================================================= */

/* Banda del alternador con grietas / cristalizada / floja */
conocimiento('banda del alternador defectuosa (grietas)', [
    'la banda del alternador tiene grietas',
    'la banda del alternador esta cristalizada',
    'la banda del alternador esta floja'
]).

/* Banda del alternador engrasada que rechina */
conocimiento('banda del alternador defectuosa (engrasada)', [
    'la banda del alternador esta engrasada',
    'la banda del alternador rechina al girar'
]).

/* Bateria descargada o en mal estado */
conocimiento('bateria defectuosa', [
    'las luces y el ventilador estan encendidos',
    'el motor esta apagado',
    'el voltaje de la bateria es menor a 10.5 volts'
]).

/* Regulador de voltaje fuera de rango */
conocimiento('regulador defectuoso', [
    'el motor esta en marcha',
    'las luces estan encendidas',
    'las luces se intensifican al acelerar'
]).

/* Alternador que no genera corriente suficiente */
conocimiento('alternador defectuoso', [
    'la bateria esta en buenas condiciones',
    'la luz de advertencia permanece encendida',
    'el motor de arranque gira lentamente'
]).

/* Sistema de frenos con problemas mecanicos */
conocimiento('sistema de frenos defectuoso', [
    'el pedal del freno esta duro',
    'al frenar se produce un tiron lateral',
    'las ruedas rechinan al frenar'
]).

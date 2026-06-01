/* =============================================================
   BaseConocimientos3.pl  –  Identificacion de animales
   =============================================================
   Dominio: clasificacion de animales segun caracteristicas.
   Formato: conocimiento(Diagnostico, [Sintoma1, Sintoma2, ...]).

   Uso junto con el shell:
       ?- consult('../experto.pl').
       ?- consult('BaseConocimientos3.pl').
       ?- consulta.
   ============================================================= */

/* Felino africano de manchas */
conocimiento(cheeta, [
    'el animal es mamifero',
    'el animal es carnivoro',
    'el animal tiene color leonado',
    'el animal tiene puntos negros'
]).

/* Felino de rayas */
conocimiento(tigre, [
    'el animal es mamifero',
    'el animal es carnivoro',
    'el animal tiene color leonado',
    'el animal tiene rayas negras'
]).

/* Rumiante de cuello largo */
conocimiento(jirafa, [
    'el animal es ungulado',
    'el animal tiene cuello largo',
    'el animal tiene piernas largas'
]).

/* Equido de rayas */
conocimiento(zebra, [
    'el animal es ungulado',
    'el animal tiene rayas negras'
]).

/* Ave corredora de cuello largo */
conocimiento(aveztruz, [
    'el animal es pajaro',
    'el animal no vuela',
    'el animal tiene cuello largo'
]).

/* Ave acuatica de color blanco y negro */
conocimiento(pinguino, [
    'el animal es pajaro',
    'el animal no vuela',
    'el animal sabe nadar',
    'el animal es blanco con negro'
]).

/* Ave marina de vuelo planeo */
conocimiento(albatros, [
    'el animal es pajaro',
    'el animal aparece en historias marinas',
    'el animal vuela bien'
]).

% Hechos
mamifero(perro).
mamifero(gato).
mamifero(leon).

% Hechos adicionales
come_carne(perro).
come_carne(gato).
come_carne(leon).

% Regla
carnivoro(X) :- mamifero(X), come_carne(X).

%Ejemplo 2
masgrande(elefante, caballo).
masgrande(caballo, perro).

muchomasgrande(A, C) :- masgrande(A, B), masgrande(B, C).
muchomasgrande(A,B) :- masgrande(A,B).
muchomasgrande(A,B) :- masgrande(A,X), masgrande(B,X).

%Ejemplo 3
% Definir hechos
padre_de('Juan', 'Maria').
padre_de('Pablo', 'Juan').
padre_de('Pablo', 'Marcela').
padre_de('Carlos', 'Debora').

%A es hijo de B si B es padre de A
hijo_de(A, B) :- padre_de(B, A).

%A es abuelo de B si A es padre de C y C es padre de B
abuelo_de(A, B) :- padre_de(A, C), padre_de(C, B).

% Definir la regla para hermano
hermano_de(A, B) :- padre_de(C, A), padre_de(C, B), A\==B.
% Definir la regla para tio_de
tio_de(A, B) :- padre_de(C, B), hermano_de(A, C).


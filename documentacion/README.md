# Documentacion LaTeX de goofytex

Este directorio contiene el paper tecnico en LaTeX alineado con la estructura actual del repositorio.

## Estructura

- `main.tex`: documento principal.
- `sections/`: secciones modulares.
- `references.bib`: bibliografia.
- `Makefile`: comandos de compilacion.

## Compilacion

```bash
cd documentacion
make
```

Salida esperada: `documentacion/main.pdf`.

Para recompilar limpio:

```bash
cd documentacion
make clean
make
```

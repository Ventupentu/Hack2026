# Documentacion LaTeX del Proyecto goofytex

Este directorio contiene el paper tecnico en LaTeX alineado con el estado actual del codigo del repositorio (entrenamiento OpenCLIP, variantes de inferencia y scripts de analisis).

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

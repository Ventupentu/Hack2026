# Hack2026 - Proyecto de Reto Inditex

Este repositorio contiene la solución para el reto de Visual Product Retrieval propuesto por Inditex.

El objetivo, dado un "bundle" (una imagen que muestra a un modelo vistiendo un conjunto de prendas o varias prendas agrupadas), es identificar qué artículos individuales (productos del catálogo) están presentes en esa composición.

## Estructura del Proyecto

La estructura de directorios y código del proyecto se organiza de la siguiente manera:

```text
Hack2026/
|-- data/
|   |-- raw/                  Datos originales de la descarga (ficheros CSV, imágenes de productos y bundles).
|   |-- processed/            Imágenes procesadas listas para entrenamiento (redimensionadas, recortes).
|   |-- embeddings/           Vectores de características calculados previamente para agilizar la inferencia.
|-- notebooks/
|   |-- 01_EDA.ipynb          Libreta para la exploración estadística y limpieza visual de los datos.
|   |-- 02_Baseline.ipynb     Libreta para pruebas rápidas de inferencia con modelos base (zero-shot).
|-- src/                      Código fuente principal y lógica del modelo.
|   |-- config.py             Archivo centralizado para hiperparámetros, rutas del sistema y configuraciones generales.
|   |-- data/                 Módulo de gestión de datos.
|   |   |-- dataset.py        Clases para estructurar los datos (PyTorch Datasets para productos y bundles).
|   |   |-- transforms.py     Lógica asociada a la aumentación y transformación de las imágenes de entrada.
|   |-- models/               Módulo de arquitecturas y aprendizaje.
|   |   |-- feature_extractors.py Envolturas para cargar modelos extractores de características (ej. ViT, ResNet, CLIP).
|   |   |-- metric_learning.py    Implementación de funciones de pérdida y cabezas de aprendizaje métrico (Contrastive, etc).
|   |-- utils/                Módulo de utilidades generales del proyecto.
|   |   |-- metrics.py        Funciones para el cálculo del rendimiento en base a las métricas requeridas (Recall@K, mAP).
|   |   |-- retrieval.py      Lógica para la indexación y búsqueda rápida de vectores de similitud.
|   |   |-- logger.py         Utilidades de registro de eventos y métricas de entrenamiento.
|   |-- train.py              Script principal que coordina el ciclo de entrenamiento o fine-tuning.
|   |-- infer.py              Script de inferencia que evalúa el conjunto de test y genera el documento a entregar.
|-- tests/                    Módulo de pruebas unitarias para corroborar elementos puntuales como transformaciones.
|-- submissions/              Carpeta destino para guardar de forma ordenada los CSV con predicciones generadas.
|-- requirements.txt          Lista de dependencias y versiones en Python para asegurar la reproducibilidad.
|-- README.md                 Este documento central.
```

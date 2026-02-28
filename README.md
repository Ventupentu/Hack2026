# Hack2026

## Objetivo del Proyecto
Construir un modelo que prediga coincidencias de productos para bundles de moda.

Debes entrenar usando los CSV de entrenamiento proporcionados y generar la salida final completando los valores `product_asset_id` para los bundles de test.

## Resumen del Dataset

### 1) `data/bundles_dataset.csv`
Catalogo de bundles (lado de consulta).

Columnas:
- `bundle_asset_id`: Identificador unico de la imagen del bundle.
- `bundle_id_section`: ID de seccion/categoria del bundle.
- `bundle_image_url`: URL de la imagen del bundle.

### 2) `data/product_dataset.csv`
Catalogo de productos (lado candidato).

Columnas:
- `product_asset_id`: Identificador unico de la imagen del producto.
- `product_image_url`: URL de la imagen del producto.
- `product_description`: Descripcion textual del producto.

### 3) `data/bundles_product_match_train.csv`
Pares supervisados de entrenamiento (coincidencias reales).

Columnas:
- `bundle_asset_id`
- `product_asset_id`

Cada fila es una coincidencia valida bundle-producto usada para entrenamiento.

### 4) `data/bundles_product_match_test.csv`
Plantilla de test para la prediccion final.

Columnas:
- `bundle_asset_id`: Bundle a evaluar.
- `product_asset_id`: Campo vacio que debes completar con las predicciones del modelo.

## Como se Conectan los Archivos
- Une los IDs de bundle de train/test con `bundles_dataset.csv` usando `bundle_asset_id`.
- Une los IDs de producto (predichos o de train) con `product_dataset.csv` usando `product_asset_id`.
- La senal de entrenamiento viene de `bundles_product_match_train.csv`.
- La entrega final se basa en `bundles_product_match_test.csv`.

## Reglas de Entrega Final (Importante)

### Proposito
Usa este dataset para entregar tus resultados finales.

### Columnas Obligatorias
- `bundle_asset_id`: Identificador de la imagen del bundle (grupo de productos).
- `product_asset_id`: Identificador de la imagen del producto. Debes completar esta columna.

### Reglas de Formato
- Aunque un ejemplo muestre una fila por bundle, el resultado final debe incluir una fila por cada producto reconocido dentro de cada bundle.
- Varias filas pueden compartir el mismo `bundle_asset_id` (una por producto predicho).
- Se evaluara un maximo de 15 productos por bundle.
- Para cada bundle, solo se consideran las primeras 15 filas durante la evaluacion.

## Comportamiento Sugerido de Salida
- Mantener las filas agrupadas por `bundle_asset_id`.
- Dentro de cada bundle, ordenar las filas por confianza del modelo (mejor prediccion primero).
- Limitar las predicciones a un maximo de 15 filas por bundle para cumplir las restricciones de evaluacion.

## Linea Base: Recuperacion Preentrenada

Este repositorio incluye un baseline simple en `src/infer.py`:

- Encoder: modelo preentrenado de `torchvision` (`resnet50` por defecto).
- Metodo: extrae embeddings normalizados para todos los productos y luego recupera los top-K productos mas cercanos para cada bundle.
- Validacion: calcula `hit@K` y `recall@K` en una particion aleatoria de validacion por `bundle_asset_id`.
- Entrega: escribe una fila por producto predicho y limita a 15 productos por bundle.

### Ejecucion

```bash
python -m src.infer \
  --model-name resnet50 \
  --batch-size 64 \
  --val-ratio 0.2 \
  --top-n-submit 15 \
  --submission-out outputs/test_submission.csv \
  --metrics-out outputs/val_metrics.json
```

### Salidas principales

- `outputs/test_submission.csv`: archivo listo para subir (`bundle_asset_id,product_asset_id`).
- `outputs/val_metrics.json`: metricas de validacion local y resumen de ejecucion.

## Ejemplo de Salida (Solo Formato)

```csv
bundle_asset_id,product_asset_id
B_xxxxx,I_aaaaa
B_xxxxx,I_bbbbb
B_yyyyy,I_ccccc
```

En este ejemplo, el bundle `B_xxxxx` tiene dos productos reconocidos, por eso aparece en dos filas.

## Pipeline de IA Recomendado

Este reto se resuelve mejor como un problema de **recuperacion visual multiobjeto**, no como un clasificador de conjunto cerrado.
Ademas, cada `product_asset_id` tiene una sola imagen, por lo que el pipeline debe ser robusto a representaciones de producto de vista unica.

### 1) Preparacion de Datos
- Validar IDs y uniones entre todos los CSV.
- Construir particiones train/validacion por `bundle_asset_id` (sin fuga de informacion).
- Generar tablas de metadatos de producto (`product_asset_id`, ruta de imagen, `product_description`).

### 2) Indice de Embeddings de Producto (Offline)
- Codificar todas las imagenes de productos en embeddings usando backbones visuales potentes.
- Usar test-time augmentation (TTA) fuerte en imagenes de producto (multi-crop/flip/color jitter) y promediar embeddings para crear un vector unico mas robusto por producto.
- Guardar vectores y construir un indice ANN para busqueda rapida de vecinos mas cercanos.
- Mantener el indice persistente para reutilizarlo en inferencia.

```bash
python preprocess_data.py --skip_download --out_dir data/preprocessed --val_ratio 0.1 --seed 42
```

### Aumento de Datos Offline (Guardado en Disco)

Usa `offline_augment.py` para generar vistas adicionales manteniendo los IDs (`bundle_asset_id` / `product_asset_id`):

```bash
python offline_augment.py \
  --bundles_manifest data/manifests/train_manifest.jsonl \
  --products_manifest data/product_dataset.csv \
  --products_images_dir data/product_images \
  --out_dir data/offline_aug \
  --bundles_num_augs 4 \
  --products_num_augs 2 \
  --img_size 224 \
  --seed 42 \
  --workers 8
```

Salidas:
- `data/offline_aug/bundles_aug/*.jpg`
- `data/offline_aug/products_aug/*.jpg`
- `data/offline_aug/bundles_aug_manifest.jsonl`
- `data/offline_aug/products_aug_manifest.jsonl`

### 3) Deteccion de Items en Bundle
- Detectar regiones/recortes de items en cada imagen de bundle.
- Mantener un recorte de imagen completa como respaldo cuando la deteccion falle en items pequenos.

### 4) Recuperacion de Candidatos
- Para cada recorte de bundle, recuperar los top-K productos candidatos desde el indice ANN.
- Usar priors de categoria de `product_description` y `bundle_id_section` para reducir falsos positivos.
- Aplicar augmentacion en tiempo de consulta sobre recortes de bundle y fusionar puntajes para compensar diferencias de punto de vista/fondo frente a imagenes de producto de vista unica.

### 5) Reordenamiento
- Reordenar candidatos recuperados con un scorer por pares usando:
- Caracteristicas de similitud visual.
- Confianza del detector.
- Compatibilidad categoria/seccion.
- Priorizar perdidas de metric learning con negativos dificiles para mejorar la discriminacion cuando solo existe una imagen de referencia por producto.

### 6) Prediccion Final y Entrega
- Fusionar candidatos de todos los recortes de un bundle.
- Eliminar duplicados de `product_asset_id`.
- Ordenar por confianza y conservar hasta las primeras 15 predicciones por bundle.
- Exportar un CSV de entrega con una fila por producto predicho.

### Restriccion de Producto con Imagen Unica (Importante)
- Solo hay una imagen por ID de producto, por lo que no debes depender de aprendizaje multivista a nivel de producto.
- Prioriza estrategias de robustez de embeddings: promediado con TTA, normalizacion fuerte de imagen y fusion de caracteristicas de dos encoders complementarios.
- Usa recuperacion + reordenamiento en lugar de clasificacion directa sobre todos los productos, ya que el soporte por clase es extremadamente escaso.

## Tecnologias Recomendadas

### Framework Base
- `Python 3.11+`
- `PyTorch`
- `torchvision`
- `pandas`, `numpy`

### Extraccion de Caracteristicas / Encoders
- `transformers` + `timm`
- `SigLIP` (embeddings alineados imagen-texto)
- `DINOv2` (embeddings visuales robustos)

### Deteccion y Localizacion
- `GroundingDINO` (deteccion de vocabulario abierto), o `OWL-ViT` como alternativa
- Opcional: `segment-anything` para recortes mas ajustados si hace falta

### Recuperacion
- `FAISS` para indice ANN y busqueda por similitud
- Fusion TTA en tiempo de embedding y en tiempo de consulta para estabilizar el ranking de vecinos cercanos bajo condiciones de una sola imagen por producto

### Reordenamiento
- `LightGBM` como ranker o un scorer MLP pequeno en `PyTorch`

### Entrenamiento / Experimentacion
- `Hydra` para gestion de configuracion
- `Weights & Biases` o `MLflow` para seguimiento de experimentos

### Inferencia y Salida
- Inferencia en GPU por lotes para embeddings y deteccion
- Postprocesado determinista para forzar la restriccion de top-15 por bundle en la salida

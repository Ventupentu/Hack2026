# Goofytex

Repositorio para retrieval de productos de moda a partir de imagenes de bundles.

Actualizado con inspeccion local del repo en fecha **01-03-2026**.

## Pesos

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Sergiugz/GoofyTex)

Este repositorio contiene el código fuente para el proyecto Hack2026. Los pesos del modelo están alojados en Hugging Face: [Sergiugz/GoofyTex](https://huggingface.co/Sergiugz/GoofyTex).

## Resumen

El objetivo es predecir `product_asset_id` para cada `bundle_asset_id` de test, con ranking de hasta 15 productos por bundle.

El stack actual esta centrado en **OpenCLIP (Marqo Fashion SigLIP)**, con variantes de inferencia que combinan:

- deteccion YOLO de prendas en bundles,
- retrieval por similitud coseno,
- filtros de post-procesado (gender, categoria, umbral de score),
- reranking opcional con timestamps y modelos ligeros/pesados.

## Inventario de ideas implementadas en codigo

### Entrenamiento y representacion

- **Entrenamiento contrastivo bundle-producto** con OpenCLIP multimodal (imagen + texto de producto), scheduler coseno con warmup, AMP y acumulacion de gradiente (`src/models/retrieval_openclip.py`).
- **Hard negative mining configurable** (`mine_every`, `hard_neg_top_k`, `max_hard_negatives`) para reforzar discriminacion en candidatos dificiles (`config/config.yaml`, `src/models/retrieval_openclip.py`).
- **Entrenamiento/inferencia guiados por cajas**: opcion de usar boxes YOLO en bundles, con cache persistente de detecciones (`src/models/retrieval_openclip.py`, `src/infer.py`).
- **Compatibilidad multi-GPU (DataParallel)** y control de device/fallback CPU (`src/models/retrieval_openclip.py`, `src/train.py`).

### Inferencia (familia de pipelines)

- **Pipeline principal (`src.infer`)**: retrieval por boxes + fusion por score maximo, NMS de cajas, filtro de genero, deduplicacion por categoria, score threshold y fallback por productos populares cuando falta señal.
- **Pipeline per-crop avanzado (`src.new_infer`)**: TTA para productos y crops, agregacion por suma entre crops, indexado FAISS opcional (fallback numpy), y rerank temporal por `ts` (misma fecha/mes/trimestre).
- **Pipeline por fases (`src.infer_phase1`)**:
- indices de producto por seccion (1/2/3) aprendidos desde train;
- clasificacion zero-shot de categoria por crop via encoder de texto CLIP;
- boost por coincidencia de categoria y boost fuerte por coincidencia de `article`;
- export opcional de embeddings agregados de train para entrenar reranker MLP.
- **Pipeline top-1 por crop (`src.infer_top1`)**: una prediccion por prenda detectada, evitando inflar artificialmente a top-15 cuando la escena tiene pocas prendas.
- **Pipeline legacy (`src.infer_openclip`)**: ruta simple por argparse para checkpoints OpenCLIP.

### Reranking y fusion de modelos

- **Hubness penalty**: penaliza productos que aparecen con demasiada frecuencia entre bundles (`src/reranker.py`).
- **Heavy-model rerank (late interaction)**: re-score con un modelo visual mas pesado para refinar candidatos top (`src/reranker.py`).
- **Reranker MLP listwise entrenado**:
- objetivo listwise multi-positivo sobre top-k candidatos;
- features configurables (`sim`, `|q-p|`, `q*p`, `sq_diff`, concatenaciones, features extra de query);
- mezcla final `mlp_score + alpha * cosine_score`;
- cache de candidatos top-k para acelerar entrenamiento (`src/rerank/train_mlp.py`).
- **Ensemble de submissions por Reciprocal Rank Fusion (RRF)** para combinar multiples CSVs de prediccion (`src/ensemble_csv.py`).

### Utilidades de datos y analitica

- **Preprocesado robusto** con validacion de esquema, integridad referencial, limpieza de etiquetas y manifests train/val (`src/utils/preprocess_data.py`).
- **Augmentacion offline determinista** con semillas estables por asset+indice (`src/utils/offline_augment.py`).
- **Inferencia de genero de producto** por enlaces train + heuristicas de descripcion (`src/utils/add_gender.py`).
- **Analisis temporal bundle-producto** a partir de timestamps en URLs (`src/utils/check_link_timestamps.py`).

## Estructura Relevante (rutas actualizadas)

- `src/train.py`: entrypoint Hydra para entrenamiento OpenCLIP.
- `src/models/retrieval_openclip.py`: loop de train/val, checkpoints y `metrics.jsonl`.
- `src/infer.py`: inferencia principal (OpenCLIP + YOLO + filtros).
- `src/new_infer.py`: variante per-crop con TTA y FAISS opcional.
- `src/infer_phase1.py`: variante con filtrado por seccion + zero-shot por categoria.
- `src/infer_top1.py`: variante top-1 por crop.
- `src/infer_openclip.py`: inferencia simple por argparse desde checkpoint.
- `src/rerank/train_mlp.py`: entrenamiento de reranker MLP listwise.
- `src/reranker.py`: funciones de reranking de post-procesado.
- `src/ensemble_csv.py`: ensemble de submissions por RRF.
- `src/detection.py`: wrapper de YOLO clothing detection.
- `src/utils/add_gender.py`: genera `data/product_dataset_with_gender.csv`.
- `src/utils/check_link_timestamps.py`: reporte de alineacion temporal bundle-producto.
- `src/utils/preprocess_data.py`: validacion/split/manifests/stats.
- `src/utils/offline_augment.py`: augmentacion offline para bundles/productos.
- `src/utils/download.sh`: descarga de imagenes de bundles/productos.
- `src/notebooks/`: notebooks de exploracion.

## Instalacion

Dependencias base:

```bash
pip install -r requirements.txt
```

Dependencias usadas por pipelines avanzados (no incluidas en `requirements.txt`):

```bash
pip install open_clip_torch ultralyticsplus
```

Opcional para acelerar retrieval en `src.new_infer.py`:

```bash
pip install faiss-cpu
```

## Flujo Recomendado

### 1) Descargar imagenes

```bash
bash src/utils/download.sh
```

### 2) Preprocesar y generar manifests/stats

```bash
python src/utils/preprocess_data.py \
  --skip_download \
  --out_dir data \
  --val_ratio 0.1 \
  --seed 42
```

Genera/actualiza, entre otros:

- `data/manifests/train_manifest.jsonl`
- `data/manifests/val_manifest.jsonl`
- `data/labels.json`
- `data/label2idx.json`
- `data/stats.json`

### 3) (Opcional) Augmentacion offline

```bash
python src/utils/offline_augment.py \
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

### 4) Generar gender por producto (necesario para train por defecto)

```bash
python src/utils/add_gender.py
```

Esto crea `data/product_dataset_with_gender.csv`.

### 5) Entrenar

```bash
python -m src.train
```

Por defecto usa rutas de `config/files/file.yaml`:

- `data_dir: data`
- `bundles_images: data/bundle_images`
- `products_images: data/product_images`
- `yolo_detections_dir: data/yolo_detections`

Salida de entrenamiento en carpeta Hydra:

- `<hydra_run_dir>/retrieval_openclip/best.pt`
- `<hydra_run_dir>/retrieval_openclip/epoch_*.pt`
- `<hydra_run_dir>/retrieval_openclip/metrics.jsonl`

## Inferencia

### Opcion A: `src.infer` (pipeline principal)

```bash
python -m src.infer \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt \
  infer.val_ratio=0.1
```

Salida en carpeta Hydra de la corrida:

- `test_submission.csv`
- `val_metrics.json`

### Opcion B: `src.new_infer`

```bash
python -m src.new_infer \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt
```

### Opcion C: `src.infer_phase1`

```bash
python -m src.infer_phase1 \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt
```

### Opcion D: `src.infer_top1`

```bash
python -m src.infer_top1 \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt
```

### Opcion E: `src.infer_openclip` (legacy)

```bash
python -m src.infer_openclip \
  --checkpoint outputs/retrieval_openclip/best.pt \
  --submission-out outputs/retrieval_openclip/submission.csv
```

## Reranking y Ensemble

### Entrenar reranker MLP

```bash
python -m src.rerank.train_mlp \
  --query-embeddings artifacts/embeddings/train_bundle_embeddings.pt \
  --product-embeddings outputs/product_embeddings.pt \
  --train-csv data/bundles_product_match_train.csv \
  --output artifacts/rerank/mlp_reranker.pt
```

### Ensemble de CSVs (RRF)

```bash
python src/ensemble_csv.py outputs/sub1.csv outputs/sub2.csv -o outputs/ensemble_submission.csv
```

## Reportes Utiles

### Alineacion temporal bundle-producto

```bash
python src/utils/check_link_timestamps.py \
  --out-csv outputs/ts_date_alignment_report.csv \
  --timezone UTC
```

## Calidad y Limitaciones

- Los tests actuales en `tests/` son minimos (`tests/__init__.py` y `tests/test_submission.csv`).
- `requirements.txt` no incluye dependencias opcionales clave (`open_clip_torch`, `ultralyticsplus`, `faiss-cpu`).
- Algunos scripts asumen recursos externos (modelo YOLO y checkpoints OpenCLIP).
- Si activas `infer.gender_filter=true` y no existe `product_dataset_with_gender.csv`, el filtro se desactiva de facto.

## Documentacion Tecnica

- Paper tecnico LaTeX: [documentacion/README.md](documentacion/README.md)

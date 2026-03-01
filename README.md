# goofytex

Repositorio para retrieval de productos de moda a partir de imagenes de bundles.

Actualizado con inspeccion local del repo en fecha **2026-02-28**.

## Resumen

El objetivo es predecir `product_asset_id` para cada `bundle_asset_id` de test, con ranking de hasta 15 productos por bundle.

El stack actual esta centrado en **OpenCLIP (Marqo Fashion SigLIP)**, con variantes de inferencia que combinan:

- deteccion YOLO de prendas en bundles,
- retrieval por similitud coseno,
- filtros de post-procesado (gender, categoria, umbral de score),
- reranking opcional con timestamps extraidos de URLs.

## Estado Actual del Workspace

Snapshot local detectado en este repo:

- `data/bundles_dataset.csv`: 2,331 bundles
- `data/product_dataset.csv`: 27,688 productos
- `data/bundles_product_match_train.csv`: 6,493 relaciones train
- `data/bundles_product_match_test.csv`: 455 bundles de test (sin IDs faltantes respecto al catalogo)
- imagenes locales descargadas: 2,331 bundles y 27,688 productos
- train (unicos): 1,876 bundles, 4,012 productos
- promedio de productos por bundle en train: 3.461 (min=1, max=11)

Artefactos de analisis presentes en `outputs/`:

- `ts_date_alignment_report.csv` (1,876 bundles analizados)
- `category_section_consistency.csv`
- `category_section_summary.json`

## Estructura Relevante

- `src/train.py`: entrypoint Hydra para entrenamiento OpenCLIP.
- `src/models/retrieval_openclip.py`: loop de train/val, checkpoints y `metrics.jsonl`.
- `src/infer.py`: inferencia principal (OpenCLIP + YOLO + filtros).
- `src/new_infer.py`: variante "v8" per-crop con TTA y FAISS opcional.
- `src/infer_phase1.py`: variante con filtrado por seccion + zero-shot por categoria.
- `src/infer_top1.py`: variante top-1 por crop (salidas mas cortas por bundle).
- `src/infer_openclip.py`: inferencia simple por argparse desde checkpoint.
- `src/detection.py`: wrapper de YOLO clothing detection.
- `src/utils/add_gender.py`: genera `product_dataset_with_gender.csv`.
- `src/utils/check_link_timestamps.py`: reporte de alineacion temporal bundle-producto.
- `preprocess_data.py`: validacion/split/manifests/stats.
- `offline_augment.py`: augmentacion offline para bundles/productos.

## Instalacion

Dependencias base:

```bash
pip install -r requirements.txt
```

Dependencias usadas por los pipelines avanzados (no incluidas en `requirements.txt`):

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
bash download.sh
```

### 2) Preprocesar y generar manifests/stats

```bash
python preprocess_data.py \
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

### 3) Generar gender por producto (necesario para train por defecto)

```bash
python src/utils/add_gender.py
```

Esto crea `data/product_dataset_with_gender.csv`.

Importante: `src/train.py` usa ese archivo por defecto como `products_manifest`.

### 4) Entrenar

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

Nota tecnica: actualmente `src/train.py` apunta train y val al mismo CSV (`bundles_product_match_train.csv`).

## Inferencia

### Opcion A: `src.infer` (pipeline principal)

## Quick start
```bash
python -m src.infer \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt \
  infer.val_ratio=0.1
```

Salida en carpeta Hydra de la corrida:

- `test_submission.csv`
- `val_metrics.json`

### Opcion B: `src.new_infer` (v8)

```bash
python -m src.new_infer \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt
```

Salida en `outputs/`:

- `test_submission_v8.csv`
- `val_metrics_v8.json` (si `infer.val_ratio > 0`)

### Opcion C: `src.infer_phase1`

```bash
python -m src.infer_phase1 \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt
```

Salida en `outputs/`:

- `test_submission_phase1.csv`
- `val_metrics_phase1.json` (si `infer.val_ratio > 0`)

### Opcion D: `src.infer_top1`

```bash
python -m src.infer_top1 \
  infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt
```

Salida en `outputs/`:

- `test_submission_top1.csv`
- `val_metrics_top1.json` (si `infer.val_ratio > 0`)

Esta variante devuelve menos items por bundle (top-1 por crop, cap ~6), no fuerza 15 filas.

### Opcion E: `src.infer_openclip` (script argparse legacy)

```bash
python -m src.infer_openclip \
  --checkpoint outputs/retrieval_openclip/best.pt \
  --submission-out outputs/retrieval_openclip/submission.csv
```

## Reportes Utiles

### Alineacion temporal bundle-producto

```bash
python src/utils/check_link_timestamps.py \
  --out-csv outputs/ts_date_alignment_report.csv \
  --timezone UTC
```

Resumen del reporte existente (`outputs/ts_date_alignment_report.csv`):

- bundles analizados: 1,876
- productos vinculados: 6,493
- same date: 796 (12.26%)
- same month: 3,543 (54.57%)
- same quarter: 5,260 (81.01%)

### Consistencia categoria-seccion

`outputs/category_section_summary.json` existente indica:

- 88 categorias totales
- 38 categorias con >=30 muestras
- pureza 0.95 usada en el analisis
- 9 categorias "single-section" en el subset >=30
- 10 categorias "almost single-section" en el subset >=30

## Calidad y Limitaciones

- No hay tests unitarios/integ activos en `tests/` (solo `__init__.py`).
- `requirements.txt` no incluye dependencias opcionales clave (`open_clip_torch`, `ultralyticsplus`, `faiss-cpu`).
- Algunos scripts asumen recursos externos (modelo YOLO y checkpoints OpenCLIP).
- Si activas `infer.gender_filter=true` y no existe `product_dataset_with_gender.csv`, el filtro se desactiva de facto.

## Documentacion Tecnica

- Paper tecnico LaTeX: [documentacion/README.md](documentacion/README.md)

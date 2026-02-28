# Hack2026

Pipeline de retrieval para predecir `product_asset_id` por `bundle_asset_id` en moda.

Este refactor es **estructural**: se reorganizaron entrypoints en `src/workflows/` y se mantuvieron wrappers en rutas antiguas para compatibilidad. No se cambió la lógica funcional de entrenamiento/inferencia/utilidades.

## Estructura del proyecto

```text
.
├── config/
│   ├── config.yaml
│   └── files/file.yaml
├── data/
├── outputs/
├── src/
│   ├── config.py
│   ├── detection.py
│   ├── models/
│   ├── utils/
│   │   ├── add_gender.py
│   │   └── check_link_timestamps.py        # wrapper legacy
│   ├── workflows/
│   │   ├── data_preprocess.py
│   │   ├── offline_augment.py
│   │   ├── train_retrieval.py
│   │   ├── inference_baseline.py
│   │   ├── inference_advanced.py
│   │   ├── inference_phase1.py
│   │   ├── inference_top1.py
│   │   ├── inference_openclip.py
│   │   └── check_link_timestamps.py
│   ├── infer.py                            # wrapper legacy
│   ├── new_infer.py                        # wrapper legacy
│   ├── infer_phase1.py                     # wrapper legacy
│   ├── infer_top1.py                       # wrapper legacy
│   ├── infer_openclip.py                   # wrapper legacy
│   └── train.py                            # wrapper legacy
├── preprocess_data.py                      # wrapper legacy
└── offline_augment.py                      # wrapper legacy
```

## Requisitos

- Python 3.11+
- Dependencias en `requirements.txt`

Instalación:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Los scripts Hydra leen:

- `config/config.yaml`
- `config/files/file.yaml`

Ajusta `data_dir`, `bundles_images`, `products_images`, `yolo_detections_dir` según tu entorno.

## Cómo ejecutar cada archivo (canónico)

Comandos recomendados tras el refactor:

| Archivo | Propósito | Comando |
|---|---|---|
| `src/workflows/data_preprocess.py` | Preprocesado y manifests | `python -m src.workflows.data_preprocess --help` |
| `src/workflows/offline_augment.py` | Augmentación offline | `python -m src.workflows.offline_augment --help` |
| `src/workflows/train_retrieval.py` | Entrenamiento retrieval (Hydra) | `python -m src.workflows.train_retrieval` |
| `src/workflows/inference_baseline.py` | Inferencia baseline (Hydra) | `python -m src.workflows.inference_baseline` |
| `src/workflows/inference_advanced.py` | Inferencia avanzada (Hydra) | `python -m src.workflows.inference_advanced` |
| `src/workflows/inference_phase1.py` | Inferencia fase 1 (Hydra) | `python -m src.workflows.inference_phase1` |
| `src/workflows/inference_top1.py` | Inferencia top-1 por crop (Hydra) | `python -m src.workflows.inference_top1` |
| `src/workflows/inference_openclip.py` | Inferencia OpenCLIP (argparse) | `python -m src.workflows.inference_openclip --help` |
| `src/workflows/check_link_timestamps.py` | Auditoría de timestamps | `python -m src.workflows.check_link_timestamps --help` |
| `src/utils/add_gender.py` | Enriquecer productos con `gender` | `python src/utils/add_gender.py` |

## Comandos legacy (compatibles)

Estos archivos siguen funcionando y delegan al módulo canónico:

| Archivo legacy | Comando |
|---|---|
| `preprocess_data.py` | `python preprocess_data.py --help` |
| `offline_augment.py` | `python offline_augment.py --help` |
| `src/train.py` | `python -m src.train` |
| `src/infer.py` | `python -m src.infer` |
| `src/new_infer.py` | `python -m src.new_infer` |
| `src/infer_phase1.py` | `python -m src.infer_phase1` |
| `src/infer_top1.py` | `python -m src.infer_top1` |
| `src/infer_openclip.py` | `python -m src.infer_openclip --help` |
| `src/utils/check_link_timestamps.py` | `python src/utils/check_link_timestamps.py --help` |

## Flujo recomendado end-to-end

1. Preparar datos:

```bash
python -m src.workflows.data_preprocess --skip_download --out_dir data/preprocessed --val_ratio 0.1 --seed 42
```

2. (Opcional) Generar `gender` por producto:

```bash
python src/utils/add_gender.py
```

3. (Opcional) Augmentación offline:

```bash
python -m src.workflows.offline_augment \
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

4. Entrenar:

```bash
python -m src.workflows.train_retrieval
```

5. Inferir (elige variante):

```bash
python -m src.workflows.inference_baseline
python -m src.workflows.inference_advanced infer.checkpoint_path=outputs/retrieval_openclip/best.pt
python -m src.workflows.inference_phase1 infer.checkpoint_path=outputs/retrieval_openclip/best.pt
python -m src.workflows.inference_top1 infer.checkpoint_path=outputs/retrieval_openclip/best.pt
python -m src.workflows.inference_openclip --checkpoint outputs/retrieval_openclip/best.pt
```

6. Auditoría de consistencia temporal de links:

```bash
python -m src.workflows.check_link_timestamps --out-csv outputs/ts_date_alignment_report.csv --timezone UTC
```

## Outputs principales

- `outputs/test_submission.csv` o `outputs/retrieval_openclip/submission.csv`
- `outputs/val_metrics.json`
- `outputs/ts_date_alignment_report.csv`
- `data/product_dataset_with_gender.csv`

## Notas

- En inferencia final, respeta máximo 15 filas por `bundle_asset_id`.
- Para Hydra overrides usa sintaxis `clave=valor`, por ejemplo: `infer.val_ratio=0.2`.
- Si quieres inspeccionar flags de argparse: añade `--help`.

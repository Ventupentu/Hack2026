# Bundle-to-Product Retrieval (Pretrained-Only)

Pipeline para reconocer qué productos/prendas aparecen en una imagen de bundle y generar `submission.csv` con hasta 15 `product_asset_id` por `bundle_asset_id`.

Configuración actual: **sin fine-tuning**.
- Detector: GroundingDINO (`IDEA-Research/grounding-dino-base`)
- Embeddings: FashionSigLIP (`hf-hub:Marqo/marqo-fashionSigLIP`) preentrenado
- Retrieval: índice por similitud (brute o FAISS)
- Reranker: desactivado por defecto (no se entrena)

## Estructura

- `scripts/00_download_assets.py`: descarga assets con cache/retries/concurrencia.
- `scripts/01_build_manifests.py`: crea manifests y split 90/10 estratificado.
- `scripts/02_train_retrieval.py`: **prepara retrieval zero-shot** (embeddings + índice + eval opcional), sin entrenamiento.
- `scripts/03_train_reranker.py`: en modo pretrained-only deja constancia de que reranker está desactivado.
- `scripts/04_infer_and_submit.py`: inferencia en test y generación de submission.
- `scripts/05_eval.py`: evaluación en validación (Recall@5/10/15 + tiempo).

## Requisitos

- Python 3.10+
- GPU CUDA recomendada

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quickstart

1. Descargar assets

```bash
python scripts/00_download_assets.py \
  --bundles_csv data/bundles_dataset.csv \
  --products_csv data/product_dataset.csv \
  --bundle_out_dir data/bundle_images \
  --product_out_dir data/product_images \
  --bundle_index_out artifacts/paths/bundle_paths.csv \
  --product_index_out artifacts/paths/product_paths.csv
```

2. Construir manifests

```bash
python scripts/01_build_manifests.py \
  --bundles_csv data/bundles_dataset.csv \
  --products_csv data/product_dataset.csv \
  --train_relations_csv data/bundles_product_match_train.csv \
  --bundle_paths_csv artifacts/paths/bundle_paths.csv \
  --product_paths_csv artifacts/paths/product_paths.csv \
  --output_dir artifacts/manifests \
  --seed 42
```

3. Preparar retrieval con modelo preentrenado (sin fine-tuning)

```bash
python scripts/02_train_retrieval.py \
  --train_manifest artifacts/manifests/train_manifest.jsonl \
  --val_manifest artifacts/manifests/val_manifest.jsonl \
  --products_manifest artifacts/manifests/products_manifest.jsonl \
  --output_dir artifacts/retrieval \
  --product_embeddings artifacts/retrieval/product_embeddings.npz \
  --index_dir artifacts/retrieval/index \
  --index_mode brute \
  --run_val_eval
```

4. (Opcional) registrar estado de reranker desactivado

```bash
python scripts/03_train_reranker.py --output_dir artifacts/reranker
```

5. Inferencia y submission

```bash
python scripts/04_infer_and_submit.py \
  --test_csv data/bundles_product_match_test.csv \
  --bundles_csv data/bundles_dataset.csv \
  --bundle_paths_csv artifacts/paths/bundle_paths.csv \
  --products_manifest artifacts/manifests/products_manifest.jsonl \
  --retrieval_checkpoint artifacts/retrieval/pretrained_encoder.pt \
  --product_embeddings artifacts/retrieval/product_embeddings.npz \
  --index_dir artifacts/retrieval/index \
  --max_boxes 10 \
  --padding 0.15 \
  --topk_per_crop 200 \
  --submission_out artifacts/submission.csv
```

6. Evaluación

```bash
python scripts/05_eval.py \
  --val_manifest artifacts/manifests/val_manifest.jsonl \
  --products_manifest artifacts/manifests/products_manifest.jsonl \
  --retrieval_checkpoint artifacts/retrieval/pretrained_encoder.pt \
  --product_embeddings artifacts/retrieval/product_embeddings.npz \
  --index_dir artifacts/retrieval/index \
  --max_boxes 10 \
  --padding 0.15 \
  --topk_per_crop 200 \
  --report_out artifacts/eval_report.json
```

## Outputs

- `artifacts/paths/*.csv`: rutas locales de imágenes.
- `artifacts/manifests/*.jsonl`: manifests train/val/products.
- `artifacts/retrieval/product_embeddings.npz`: embeddings del catálogo.
- `artifacts/retrieval/index/`: índice retrieval persistido.
- `artifacts/retrieval/pretrained_encoder.pt`: snapshot del encoder preentrenado.
- `artifacts/retrieval/metrics.jsonl`: métricas de la fase zero-shot.
- `artifacts/retrieval/pretrained_report.json`: resumen de recall y tiempos.
- `artifacts/submission.csv`: submission final.

## Notas

- No se entrena encoder ni reranker.
- No se usa `product_description` para filtrar o clasificar.
- Si GroundingDINO devuelve pocas cajas, se activa fallback de multi-crops.

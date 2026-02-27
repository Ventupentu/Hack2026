# Hack2026

## Project Goal
Build a model that predicts product matches for fashion bundles.

You must train using the provided training CSV files and generate the final output by completing `product_asset_id` values for test bundles.

## Dataset Overview

### 1) `data/bundles_dataset.csv`
Bundle catalog (the query side).

Columns:
- `bundle_asset_id`: Unique bundle image identifier.
- `bundle_id_section`: Bundle section/category id.
- `bundle_image_url`: URL of the bundle image.

### 2) `data/product_dataset.csv`
Product catalog (the candidate side).

Columns:
- `product_asset_id`: Unique product image identifier.
- `product_image_url`: URL of the product image.
- `product_description`: Product text description.

### 3) `data/bundles_product_match_train.csv`
Supervised training pairs (ground truth matches).

Columns:
- `bundle_asset_id`
- `product_asset_id`

Each row is a valid bundle-product match used for training.

### 4) `data/bundles_product_match_test.csv`
Test template for final prediction.

Columns:
- `bundle_asset_id`: Bundle to evaluate.
- `product_asset_id`: Empty field to be filled by your model predictions.

## How the Files Connect
- Join train/test bundle ids with `bundles_dataset.csv` by `bundle_asset_id`.
- Join predicted or train product ids with `product_dataset.csv` by `product_asset_id`.
- Training signal comes from `bundles_product_match_train.csv`.
- Final submission is based on `bundles_product_match_test.csv`.

## Final Submission Rules (Important)

### Purpose
Use this dataset to deliver your final results.

### Required Columns
- `bundle_asset_id`: Identifier of the bundle image (group of products).
- `product_asset_id`: Identifier of the product image. You must complete this column.

### Format Rules
- Even if an example shows one row per bundle, the final result must include one row per recognized product inside each bundle.
- Multiple rows can share the same `bundle_asset_id` (one per predicted product).
- A maximum of 15 products per bundle will be evaluated.
- For each bundle, only the first 15 rows are considered during evaluation.

## Suggested Output Behavior
- Keep rows grouped by `bundle_asset_id`.
- Within each bundle, sort rows by model confidence (best prediction first).
- Cap predictions to at most 15 rows per bundle to match evaluation constraints.

## Baseline: Pretrained Retrieval

This repository now includes a simple baseline in `src/infer.py`:

- Encoder: pretrained `torchvision` model (`resnet50` by default).
- Method: extract normalized embeddings for all products, then retrieve top-K nearest products for each bundle.
- Validation: computes `hit@K` and `recall@K` on a random validation split by `bundle_asset_id`.
- Submission: writes one row per predicted product and caps at 15 products per bundle.

### Run

```bash
python -m src.infer \
  --model-name resnet50 \
  --batch-size 64 \
  --val-ratio 0.2 \
  --top-n-submit 15 \
  --submission-out outputs/test_submission.csv \
  --metrics-out outputs/val_metrics.json
```

### Main outputs

- `outputs/test_submission.csv`: file ready for upload (`bundle_asset_id,product_asset_id`).
- `outputs/val_metrics.json`: local validation metrics and run summary.

## Example Output (Format Only)

```csv
bundle_asset_id,product_asset_id
B_xxxxx,I_aaaaa
B_xxxxx,I_bbbbb
B_yyyyy,I_ccccc
```

In this example, bundle `B_xxxxx` has two recognized products, so it appears in two rows.

## Recommended AI Pipeline

This challenge is best solved as a **multi-object visual retrieval** problem, not as a closed-set classifier.
Also, each `product_asset_id` has only one image, so the pipeline must be robust to single-view product representations.

### 1) Data Preparation
- Validate IDs and joins across all CSVs.
- Build train/validation splits by `bundle_asset_id` (no leakage).
- Generate product metadata tables (`product_asset_id`, image path, `product_description`).

### 2) Product Embedding Index (Offline)
- Encode all product images into embeddings using strong vision backbones.
- Use strong test-time augmentation (TTA) on product images (multi-crop/flip/color jitter) and average embeddings to create a more robust single product vector.
- Store vectors and build an ANN index for fast nearest-neighbor search.
- Keep index persistent for inference reuse.

```bash
python preprocess_data.py --skip_download --out_dir data/preprocessed --val_ratio 0.1 --seed 42
```

### 3) Bundle Item Detection
- Detect item regions/crops from each bundle image.
- Keep a fallback full-image crop for robustness when detection misses small items.

### 4) Candidate Retrieval
- For each bundle crop, retrieve top-K product candidates from the ANN index.
- Use category priors from `product_description` and `bundle_id_section` to reduce false positives.
- Apply query-time augmentation on bundle crops and fuse scores to compensate for viewpoint/background differences against single-view product images.

### 5) Re-ranking
- Re-rank retrieved candidates with a pairwise scorer using:
- Visual similarity features.
- Detector confidence.
- Category/section compatibility.
- Emphasize metric-learning losses with hard negatives to improve discrimination when only one reference image exists per product.

### 6) Final Prediction & Submission
- Merge candidates from all crops in a bundle.
- Deduplicate `product_asset_id`.
- Sort by confidence and keep up to the first 15 predictions per bundle.
- Export submission CSV with one row per predicted product.

### Single-Image Product Constraint (Important)
- There is only one image per product ID, so do not rely on multi-view learning at product level.
- Prefer embedding robustness strategies: TTA averaging, strong image normalization, and feature fusion from two complementary encoders.
- Use retrieval + re-ranking instead of direct classification over all products, since class support is extremely sparse.

## Recommended Technologies

### Core Framework
- `Python 3.11+`
- `PyTorch`
- `torchvision`
- `pandas`, `numpy`

### Feature Extraction / Encoders
- `transformers` + `timm`
- `SigLIP` (image-text aligned embeddings)
- `DINOv2` (strong visual embeddings)

### Detection & Localization
- `GroundingDINO` (open-vocabulary detection), or `OWL-ViT` as alternative
- Optional: `segment-anything` for tighter crops if needed

### Retrieval
- `FAISS` for ANN index and similarity search
- Embedding-time and query-time TTA fusion to stabilize nearest-neighbor ranking under single-image-per-product conditions

### Re-ranking
- `LightGBM` ranker or a small `PyTorch` MLP scorer

### Training / Experimentation
- `Hydra` for configuration management
- `Weights & Biases` or `MLflow` for experiment tracking

### Inference & Output
- Batched GPU inference for embeddings and detection
- Deterministic post-processing to enforce top-15-per-bundle output constraint

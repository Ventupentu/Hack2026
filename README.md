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

## Example Output (Format Only)

```csv
bundle_asset_id,product_asset_id
B_xxxxx,I_aaaaa
B_xxxxx,I_bbbbb
B_yyyyy,I_ccccc
```

In this example, bundle `B_xxxxx` has two recognized products, so it appears in two rows.


### Structure


## Estructura del Proyecto


The project's directory and code structure is organized as follows:

```text
Hack2026/
|-- data/
|   |-- raw/                  Original downloaded data (CSV files, product and bundle images).
|   |-- processed/            Processed images ready for training (resized, crops).
|   |-- embeddings/           Pre-calculated feature vectors to speed up inference.
|-- notebooks/
|   |-- 01_EDA.ipynb          Notebook for statistical exploration and visual data cleaning.
|   |-- 02_Baseline.ipynb     Notebook for quick inference tests with base models (zero-shot).
|-- src/                      Main source code and model logic.
|   |-- config.py             Centralized file for hyperparameters, system paths, and general configurations.
|   |-- data/                 Data management module.
|   |   |-- dataset.py        Classes to structure the data (PyTorch Datasets for products and bundles).
|   |   |-- transforms.py     Logic associated with augmentation and transformation of input images.
|   |-- models/               Architectures and learning module.
|   |   |-- feature_extractors.py Wrappers to load feature extractor models (e.g., ViT, ResNet, CLIP).
|   |   |-- metric_learning.py    Implementation of loss functions and metric learning heads (Contrastive, etc.).
|   |-- utils/                General project utilities module.
|   |   |-- metrics.py        Functions for performance calculation based on required metrics (Recall@K, mAP).
|   |   |-- retrieval.py      Logic for fast indexation and search of similarity vectors.
|   |   |-- logger.py         Utilities for logging events and training metrics.
|   |-- train.py              Main script coordinating the training or fine-tuning cycle.
|   |-- infer.py              Inference script that evaluates the test set and generates the submission document.
|-- tests/                    Unit tests module to verify specific elements like transformations.
|-- submissions/              Destination folder to orderly save CSVs with generated predictions.
|-- requirements.txt          List of dependencies and Python versions to ensure reproducibility.
|-- README.md                 This central document.
```
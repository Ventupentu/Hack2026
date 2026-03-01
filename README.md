# Hack2026 - HackUDC x Inditex Tech Reto

**Team:** GoofyTex

## Model solution (first overview)
Our solution is a **bundle-to-product retrieval pipeline** for fashion images.

In short, this is what the model does:
1. Detect clothing regions in each bundle image with a YOLOv8 clothing detector.
2. Encode bundle crops and catalog products with OpenCLIP (SigLIP-based embeddings).
3. Retrieve the most similar products for each crop.
4. Apply post-processing and reranking (section/gender/category constraints, hubness penalty, optional MLP/heavy reranker).
5. Export a submission CSV with **exactly 15 products per bundle**.

## Objective
This repository contains our work for the **HackUDC hackathon**, in the **Inditex Tech reto**.

The objective is to predict which products from the catalog appear in each bundle/look image.  
Input: bundle image.  
Output: ranked product IDs per bundle (submission format for evaluation/leaderboard).

## How everything is done

### 1. Data preparation
- Put challenge CSVs inside `data/` (bundles, products, train/test relations).
- Download images:
  - `bash download.sh`
- Optional preprocessing and extra data utilities:
  - `python preprocess_data.py ...`
  - `python offline_augment.py ...`

### 2. Training
- Main training entrypoint:
  - `python -m src.train`
- Training config:
  - `config/config.yaml`
  - `config/files/file.yaml`
- Model checkpoints are saved under `outputs/.../retrieval_openclip/`.

### 3. Inference and submission
- Main improved inference pipeline:
  - `python -m src.infer_phase1 infer.checkpoint_path=<path_to_best.pt>`
- This generates:
  - `outputs/test_submission_phase1.csv`
- Other variants available:
  - `src/infer.py`
  - `src/new_infer.py`
  - `src/infer_top1.py`

### 4. Reranking and ensemble
- Optional reranking modules are integrated in phase1 inference (`src/reranker.py` + MLP options).
- You can ensemble multiple submission CSVs with:
  - `python src/ensemble_csv.py <csv1> <csv2> ... --output outputs/ensemble_submission.csv`

## Repository layout
- `src/`: training, inference, models, reranking, utilities.
- `config/`: Hydra configuration files.
- `data/`: datasets and images.
- `outputs/`: generated checkpoints, metrics, and submissions.
- `best_results/`: stored best submission versions.
- `notebooks/`: EDA and baseline notebooks.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run training
python -m src.train

# run inference with your trained checkpoint
python -m src.infer_phase1 infer.checkpoint_path=outputs/<date>/<time>/retrieval_openclip/best.pt
```

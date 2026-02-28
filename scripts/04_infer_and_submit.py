#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crops.cropper import Cropper
from src.detection.grounding_dino_detector import DEFAULT_PROMPT, GroundingDINODetector
from src.embeddings.encoder import FashionSigLIPEncoder
from src.pipeline.inference import (
    build_or_load_product_index,
    compute_product_embeddings,
    infer_bundle_topk,
    load_reranker,
)
from src.utils import read_jsonl, setup_logging

LOGGER = logging.getLogger("infer_and_submit")


def resolve_default_path(preferred: str, alternatives: list[str]) -> Path:
    p = ROOT / preferred
    if p.exists():
        return p
    for alt in alternatives:
        cand = ROOT / alt
        if cand.exists():
            return cand
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference + submission generation (top-15 per bundle)")
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=resolve_default_path("data/test.csv", ["data/bundles_product_match_test.csv"]),
    )
    parser.add_argument(
        "--bundles_csv",
        type=Path,
        default=resolve_default_path("data/bundles.csv", ["data/bundles_dataset.csv"]),
    )
    parser.add_argument("--bundle_paths_csv", type=Path, default=Path("artifacts/paths/bundle_paths.csv"))

    parser.add_argument("--products_manifest", type=Path, default=Path("artifacts/manifests/products_manifest.jsonl"))
    parser.add_argument("--retrieval_checkpoint", type=Path, default=Path("artifacts/retrieval/pretrained_encoder.pt"))

    parser.add_argument("--product_embeddings", type=Path, default=Path("artifacts/retrieval/product_embeddings.npz"))
    parser.add_argument("--index_dir", type=Path, default=Path("artifacts/retrieval/index"))
    parser.add_argument("--index_mode", type=str, default="brute", choices=["brute", "faiss"])
    parser.add_argument("--use_faiss_gpu", action="store_true")

    parser.add_argument("--model_name", type=str, default="hf-hub:Marqo/marqo-fashionSigLIP")
    parser.add_argument("--detector_model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--detector_prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    parser.add_argument("--max_boxes", type=int, default=10)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--topk_per_crop", type=int, default=200)

    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--reranker_checkpoint", type=Path, default=Path("artifacts/reranker/best_reranker.pt"))
    parser.add_argument("--reranker_alpha", type=float, default=0.35)
    parser.add_argument("--rerank_topn", type=int, default=500)

    parser.add_argument("--submission_out", type=Path, default=Path("artifacts/submission.csv"))
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = FashionSigLIPEncoder(model_name=args.model_name, device=device, trainable=False)
    if args.retrieval_checkpoint.exists():
        payload = torch.load(args.retrieval_checkpoint, map_location=device)
        state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
        encoder.model.load_state_dict(state_dict, strict=False)
        LOGGER.info("Loaded retrieval checkpoint: %s", args.retrieval_checkpoint)

    detector = GroundingDINODetector(
        model_id=args.detector_model,
        prompt=args.detector_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_boxes=args.max_boxes,
        device=device,
    )
    cropper = Cropper()

    products_manifest = read_jsonl(args.products_manifest)
    product_ids, product_embeddings = compute_product_embeddings(
        encoder=encoder,
        products_manifest=products_manifest,
        out_path=args.product_embeddings,
        batch_size=128,
    )

    index = build_or_load_product_index(
        index_dir=args.index_dir,
        mode=args.index_mode,
        use_faiss_gpu=args.use_faiss_gpu,
        product_ids=product_ids,
        product_embeddings=product_embeddings,
        device=device,
    )

    reranker = None
    if args.use_reranker and args.reranker_checkpoint.exists():
        reranker = load_reranker(
            checkpoint_path=args.reranker_checkpoint,
            embedding_dim=product_embeddings.shape[1],
            device=device,
        )
        LOGGER.info("Loaded reranker checkpoint: %s", args.reranker_checkpoint)

    bundles_df = pd.read_csv(args.bundles_csv)
    bundles_df["bundle_asset_id"] = bundles_df["bundle_asset_id"].astype(str)

    if args.bundle_paths_csv.exists():
        bundle_paths_df = pd.read_csv(args.bundle_paths_csv)
        bundle_paths_df["bundle_asset_id"] = bundle_paths_df["bundle_asset_id"].astype(str)
        bundle_paths_map = dict(zip(bundle_paths_df["bundle_asset_id"], bundle_paths_df["image_path"]))
    else:
        bundle_paths_map = {
            str(x): str(Path("data/bundle_images") / f"{x}.jpg")
            for x in bundles_df["bundle_asset_id"].tolist()
        }

    test_df = pd.read_csv(args.test_csv)
    test_df["bundle_asset_id"] = test_df["bundle_asset_id"].astype(str)
    bundle_ids = test_df["bundle_asset_id"].drop_duplicates().tolist()

    global_fallback_ids = index.product_ids[:500]

    submission_rows = []
    total_time = 0.0

    for bundle_id in tqdm(bundle_ids, desc="Inference"):
        image_path = bundle_paths_map.get(bundle_id)
        if not image_path:
            image_path = str(Path("data/bundle_images") / f"{bundle_id}.jpg")

        ranked_ids, _scores, elapsed = infer_bundle_topk(
            bundle_image_path=image_path,
            detector=detector,
            cropper=cropper,
            encoder=encoder,
            index=index,
            max_boxes=args.max_boxes,
            padding_ratio=args.padding,
            topk_per_crop=args.topk_per_crop,
            final_topk=15,
            use_reranker=(reranker is not None),
            reranker=reranker,
            reranker_alpha=args.reranker_alpha,
            rerank_topn=args.rerank_topn,
        )
        total_time += elapsed

        dedup = []
        seen = set()
        for pid in ranked_ids + global_fallback_ids:
            if pid in seen:
                continue
            seen.add(pid)
            dedup.append(pid)
            if len(dedup) >= 15:
                break

        for pid in dedup:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})

    submission_df = pd.DataFrame(submission_rows)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.submission_out, index=False)

    avg_time = total_time / max(1, len(bundle_ids))
    LOGGER.info("Saved submission: %s", args.submission_out)
    LOGGER.info("Generated rows: %d", len(submission_df))
    LOGGER.info("Average inference time per bundle: %.4fs", avg_time)


if __name__ == "__main__":
    main()

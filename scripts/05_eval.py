#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

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

LOGGER = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline on validation manifest")
    parser.add_argument("--val_manifest", type=Path, default=Path("artifacts/manifests/val_manifest.jsonl"))
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

    parser.add_argument("--report_out", type=Path, default=Path("artifacts/eval_report.json"))
    parser.add_argument("--failures_out", type=Path, default=Path("artifacts/eval_failures.json"))
    parser.add_argument("--max_failure_examples", type=int, default=100)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_rows = read_jsonl(args.val_manifest)
    products_manifest = read_jsonl(args.products_manifest)

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

    total_recall_5 = 0.0
    total_recall_10 = 0.0
    total_recall_15 = 0.0
    total_time = 0.0

    failures = []
    confusion_counter: Counter[str] = Counter()

    for row in tqdm(val_rows, desc="Evaluating"):
        bundle_id = str(row["bundle_asset_id"])
        positives = set(str(x) for x in row["positives"])
        if not positives:
            continue

        ranked_ids, scores, elapsed = infer_bundle_topk(
            bundle_image_path=row["image_path"],
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

        pred5 = set(ranked_ids[:5])
        pred10 = set(ranked_ids[:10])
        pred15 = set(ranked_ids[:15])

        total_recall_5 += len(positives & pred5) / len(positives)
        total_recall_10 += len(positives & pred10) / len(positives)
        total_recall_15 += len(positives & pred15) / len(positives)

        if not (positives & pred15):
            for pid in ranked_ids[:15]:
                confusion_counter[pid] += 1
            if len(failures) < args.max_failure_examples:
                failures.append(
                    {
                        "bundle_asset_id": bundle_id,
                        "image_path": row["image_path"],
                        "positives": sorted(list(positives)),
                        "predictions_top15": ranked_ids[:15],
                        "scores_top15": [float(scores.get(pid, 0.0)) for pid in ranked_ids[:15]],
                    }
                )

    denom = max(1, len(val_rows))
    report = {
        "num_val_bundles": len(val_rows),
        "recall@5": total_recall_5 / denom,
        "recall@10": total_recall_10 / denom,
        "recall@15": total_recall_15 / denom,
        "avg_inference_seconds_per_bundle": total_time / denom,
        "top_confusions": confusion_counter.most_common(50),
        "use_reranker": reranker is not None,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.failures_out.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    LOGGER.info("Saved report: %s", args.report_out)
    LOGGER.info("Saved failures: %s", args.failures_out)
    LOGGER.info("Recall@15=%.6f", report["recall@15"])


if __name__ == "__main__":
    main()

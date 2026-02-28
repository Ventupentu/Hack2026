#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
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
from src.pipeline.inference import build_or_load_product_index, compute_product_embeddings, infer_bundle_topk
from src.utils import ensure_dir, read_jsonl, set_global_seed, setup_logging

LOGGER = logging.getLogger("prepare_retrieval")


def evaluate_manifest(
    rows: list[dict],
    detector: GroundingDINODetector,
    cropper: Cropper,
    encoder: FashionSigLIPEncoder,
    topk_per_crop: int,
    max_boxes: int,
    padding: float,
    index,
) -> dict[str, float]:
    total_recall_5 = 0.0
    total_recall_10 = 0.0
    total_recall_15 = 0.0
    total_time = 0.0

    for row in tqdm(rows, desc="Zero-shot validation"):
        positives = set(str(x) for x in row.get("positives", []))
        if not positives:
            continue

        ranked_ids, _scores, elapsed = infer_bundle_topk(
            bundle_image_path=row["image_path"],
            detector=detector,
            cropper=cropper,
            encoder=encoder,
            index=index,
            max_boxes=max_boxes,
            padding_ratio=padding,
            topk_per_crop=topk_per_crop,
            final_topk=15,
            use_reranker=False,
        )
        total_time += elapsed

        pred5 = set(ranked_ids[:5])
        pred10 = set(ranked_ids[:10])
        pred15 = set(ranked_ids[:15])

        total_recall_5 += len(positives & pred5) / len(positives)
        total_recall_10 += len(positives & pred10) / len(positives)
        total_recall_15 += len(positives & pred15) / len(positives)

    denom = max(1, len(rows))
    return {
        "recall@5": total_recall_5 / denom,
        "recall@10": total_recall_10 / denom,
        "recall@15": total_recall_15 / denom,
        "avg_inference_seconds_per_bundle": total_time / denom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare retrieval artifacts with pretrained FashionSigLIP (no fine-tuning)",
    )
    parser.add_argument("--train_manifest", type=Path, default=Path("artifacts/manifests/train_manifest.jsonl"))
    parser.add_argument("--val_manifest", type=Path, default=Path("artifacts/manifests/val_manifest.jsonl"))
    parser.add_argument("--products_manifest", type=Path, default=Path("artifacts/manifests/products_manifest.jsonl"))

    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--product_embeddings", type=Path, default=Path("artifacts/retrieval/product_embeddings.npz"))
    parser.add_argument("--index_dir", type=Path, default=Path("artifacts/retrieval/index"))
    parser.add_argument("--index_mode", type=str, default="brute", choices=["brute", "faiss"])
    parser.add_argument("--use_faiss_gpu", action="store_true")

    parser.add_argument("--model_name", type=str, default="hf-hub:Marqo/marqo-fashionSigLIP")
    parser.add_argument("--save_pretrained_checkpoint", action="store_true", default=True)
    parser.add_argument("--no_save_pretrained_checkpoint", action="store_false", dest="save_pretrained_checkpoint")

    parser.add_argument("--run_val_eval", action="store_true", default=True)
    parser.add_argument("--no_run_val_eval", action="store_false", dest="run_val_eval")

    parser.add_argument("--detector_model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--detector_prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--max_boxes", type=int, default=10)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--topk_per_crop", type=int, default=200)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_global_seed(args.seed)

    ensure_dir(args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FashionSigLIPEncoder(model_name=args.model_name, device=device, trainable=False)

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

    checkpoint_path = args.output_dir / "pretrained_encoder.pt"
    if args.save_pretrained_checkpoint:
        torch.save(
            {
                "mode": "pretrained_only",
                "model_name": args.model_name,
                "model_state_dict": encoder.model.state_dict(),
            },
            checkpoint_path,
        )
        LOGGER.info("Saved pretrained encoder snapshot: %s", checkpoint_path)

    metrics = {
        "mode": "pretrained_only",
        "model_name": args.model_name,
        "num_products_embedded": len(product_ids),
        "embedding_dim": int(product_embeddings.shape[1]) if len(product_embeddings.shape) == 2 else 0,
        "index_mode": args.index_mode,
        "use_faiss_gpu": bool(args.use_faiss_gpu),
    }

    if args.run_val_eval and args.val_manifest.exists():
        val_rows = read_jsonl(args.val_manifest)
        detector = GroundingDINODetector(
            model_id=args.detector_model,
            prompt=args.detector_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_boxes=args.max_boxes,
            device=device,
        )
        cropper = Cropper()
        val_metrics = evaluate_manifest(
            rows=val_rows,
            detector=detector,
            cropper=cropper,
            encoder=encoder,
            topk_per_crop=args.topk_per_crop,
            max_boxes=args.max_boxes,
            padding=args.padding,
            index=index,
        )
        metrics.update(val_metrics)

    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    report_path = args.output_dir / "pretrained_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    LOGGER.info("Zero-shot retrieval preparation complete")
    LOGGER.info("Metrics: %s", json.dumps(metrics))


if __name__ == "__main__":
    main()

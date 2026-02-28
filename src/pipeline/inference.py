from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from src.crops.cropper import Cropper
from src.detection.grounding_dino_detector import GroundingDINODetector
from src.embeddings.encoder import FashionSigLIPEncoder
from src.rerank.model import MLPReRanker, build_pair_features
from src.retrieval.index import ProductIndex, load_product_embeddings, save_product_embeddings
from src.utils.image import load_image_rgb

LOGGER = logging.getLogger(__name__)


def compute_product_embeddings(
    encoder: FashionSigLIPEncoder,
    products_manifest: list[dict],
    out_path: str | Path,
    batch_size: int = 128,
) -> tuple[list[str], np.ndarray]:
    out_path = Path(out_path)
    if out_path.exists():
        LOGGER.info("Loading cached product embeddings from %s", out_path)
        return load_product_embeddings(out_path)

    product_ids: list[str] = []
    batch_images = []
    embeddings_chunks: list[np.ndarray] = []

    for row in tqdm(products_manifest, desc="Encoding products"):
        pid = str(row["product_asset_id"])
        image_path = row["image_path"]
        img = load_image_rgb(image_path)
        if img is None:
            continue
        product_ids.append(pid)
        batch_images.append(img)

        if len(batch_images) >= batch_size:
            embs = encoder.encode_pil_batch(batch_images, transform=encoder.preprocess_val, batch_size=batch_size)
            embeddings_chunks.append(embs.detach().cpu().numpy().astype(np.float32))
            batch_images.clear()

    if batch_images:
        embs = encoder.encode_pil_batch(batch_images, transform=encoder.preprocess_val, batch_size=batch_size)
        embeddings_chunks.append(embs.detach().cpu().numpy().astype(np.float32))

    if embeddings_chunks:
        embeddings = np.concatenate(embeddings_chunks, axis=0)
    else:
        embeddings = np.empty((0, encoder.embedding_dim), dtype=np.float32)

    save_product_embeddings(out_path, product_ids, embeddings)
    LOGGER.info("Saved product embeddings to %s", out_path)
    return product_ids, embeddings


def build_or_load_product_index(
    index_dir: str | Path,
    mode: str,
    use_faiss_gpu: bool,
    product_ids: list[str] | None = None,
    product_embeddings: np.ndarray | None = None,
    device: str | None = None,
) -> ProductIndex:
    index_dir = Path(index_dir)
    meta_path = index_dir / "index_meta.json"
    if meta_path.exists():
        LOGGER.info("Loading existing product index from %s", index_dir)
        return ProductIndex.load(index_dir, device=device)

    if product_ids is None or product_embeddings is None:
        raise ValueError("product_ids and product_embeddings are required when building a new index")

    index = ProductIndex(mode=mode, use_gpu=use_faiss_gpu, device=device)
    index.build(product_ids=product_ids, embeddings=product_embeddings)
    index.save(index_dir)
    LOGGER.info("Saved product index to %s", index_dir)
    return index


def load_reranker(
    checkpoint_path: str | Path,
    embedding_dim: int,
    device: str,
) -> MLPReRanker:
    reranker = MLPReRanker(embedding_dim=embedding_dim)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    reranker.load_state_dict(state_dict)
    reranker.to(device)
    reranker.eval()
    return reranker


def _rerank_fused_scores(
    query_embs: torch.Tensor,
    fused_scores: dict[int, float],
    index: ProductIndex,
    reranker: MLPReRanker,
    alpha: float,
    rerank_topn: int,
    device: str,
) -> dict[int, float]:
    if not fused_scores:
        return fused_scores

    sorted_base = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    head = sorted_base[:rerank_topn]
    tail = sorted_base[rerank_topn:]

    refined: dict[int, float] = {}
    prod_embs = torch.from_numpy(index.embeddings).to(device)  # type: ignore[arg-type]

    with torch.no_grad():
        for prod_idx, base_score in head:
            prod = prod_embs[prod_idx].unsqueeze(0).repeat(query_embs.shape[0], 1)
            feats = build_pair_features(query_embs, prod)
            logits = reranker(feats)
            prob = torch.sigmoid(logits).max().item()
            refined[prod_idx] = (1.0 - alpha) * float(base_score) + alpha * float(prob)

    for prod_idx, base_score in tail:
        refined[prod_idx] = float(base_score)

    return refined


def infer_bundle_topk(
    bundle_image_path: str,
    detector: GroundingDINODetector,
    cropper: Cropper,
    encoder: FashionSigLIPEncoder,
    index: ProductIndex,
    max_boxes: int,
    padding_ratio: float,
    topk_per_crop: int,
    final_topk: int = 15,
    use_reranker: bool = False,
    reranker: MLPReRanker | None = None,
    reranker_alpha: float = 0.35,
    rerank_topn: int = 500,
) -> tuple[list[str], dict[str, float], float]:
    """Run detector->crops->retrieval->fusion and return ranked product IDs."""
    start = perf_counter()
    image = load_image_rgb(bundle_image_path)
    if image is None:
        return [], {}, 0.0

    detections = detector.detect(
        image=image,
        max_boxes=max_boxes,
        padding_ratio=padding_ratio,
    )
    boxes = [det.bbox for det in detections]
    crop_items = cropper.build_query_crops(
        image=image,
        detector_boxes=boxes,
        max_boxes=max_boxes,
        min_detector_crops_for_no_fallback=2,
    )
    if not crop_items:
        return [], {}, 0.0

    crop_images = [item.image for item in crop_items]
    query_embs = encoder.encode_pil_batch(crop_images, transform=encoder.preprocess_val, batch_size=64)
    if query_embs.shape[0] == 0:
        return [], {}, 0.0

    scores, indices = index.search(query_embs, topk=topk_per_crop)

    fused_scores_by_idx: dict[int, float] = {}
    for row_idx in range(scores.shape[0]):
        for score, prod_idx in zip(scores[row_idx], indices[row_idx]):
            prod_idx = int(prod_idx)
            score = float(score)
            prev = fused_scores_by_idx.get(prod_idx)
            if prev is None or score > prev:
                fused_scores_by_idx[prod_idx] = score

    if use_reranker and reranker is not None and index.embeddings is not None:
        fused_scores_by_idx = _rerank_fused_scores(
            query_embs=query_embs,
            fused_scores=fused_scores_by_idx,
            index=index,
            reranker=reranker,
            alpha=reranker_alpha,
            rerank_topn=rerank_topn,
            device=encoder.device,
        )

    ranked = sorted(fused_scores_by_idx.items(), key=lambda x: x[1], reverse=True)[:final_topk]
    ranked_ids = [index.product_ids[idx] for idx, _ in ranked]
    ranked_scores = {index.product_ids[idx]: float(score) for idx, score in ranked}
    elapsed = perf_counter() - start
    return ranked_ids, ranked_scores, elapsed

"""Post-processing rerankers for Phase 1 predictions."""

from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm
import open_clip

def apply_hubness_penalty(
    predictions: Dict[str, List[Tuple[str, float]]],
    max_frequency_ratio: float = 0.015,
    penalty_weight: float = 0.1
) -> Dict[str, List[Tuple[str, float]]]:
    """Penalize products that appear in too many bundles."""
    total_bundles = len(predictions)
    if total_bundles == 0:
        return predictions
        
    product_counts = defaultdict(int)
    for preds in predictions.values():
        for pid, _ in preds:
            product_counts[pid] += 1
            
    reranked_predictions = {}
    for bundle_id, preds in predictions.items():
        new_preds = []
        for pid, score in preds:
            freq_ratio = product_counts[pid] / total_bundles
            if freq_ratio > max_frequency_ratio:
                excess = freq_ratio - max_frequency_ratio
                score -= (penalty_weight * excess)
            new_preds.append((pid, score))
        reranked_predictions[bundle_id] = sorted(new_preds, key=lambda x: x[1], reverse=True)
        
    return reranked_predictions


def heavy_model_rerank(
    predictions: Dict[str, List[Tuple[str, float]]],
    bundle_crops: Dict[str, List[Image.Image]],
    products_df: pd.DataFrame,
    product_images_dir: str,
    device: str,
    model_name: str,
    pretrained: str,
    batch_size: int = 32,
    heavy_weight: float = 0.4
) -> Dict[str, List[Tuple[str, float]]]:
    """Rerank candidates using a heavy vision model (late interaction)."""
    if not predictions:
        return predictions

    print(f"Loading heavy model {model_name} for reranking...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=torch.device(device)
    )
    clip_model.eval()

    # Get unique products to encode
    unique_pids = set()
    for preds in predictions.values():
        for pid, _ in preds:
            unique_pids.add(pid)
    unique_pids = list(unique_pids)
    print(f"  Encoding {len(unique_pids)} unique candidate products with heavy model...")

    product_image_urls = dict(zip(products_df["product_asset_id"].astype(str), products_df["product_image_url"].astype(str)))
    product_images_dir_path = Path(product_images_dir)
    
    # 1. Encode candidate products
    product_embeds = {}
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=(device=="cuda")):
            for i in tqdm(range(0, len(unique_pids), batch_size), desc="Heavy Prod Encode"):
                batch_pids = unique_pids[i:i+batch_size]
                batch_imgs = []
                valid_indices = []
                for j, pid in enumerate(batch_pids):
                    url = product_image_urls.get(pid, "")
                    if url:
                        fn = url.split("/")[-1]
                        img_path = product_images_dir_path / fn
                        if img_path.exists():
                            try:
                                batch_imgs.append(preprocess(Image.open(img_path).convert("RGB")))
                                valid_indices.append(j)
                            except Exception:
                                pass
                if batch_imgs:
                    img_tensor = torch.stack(batch_imgs).to(device)
                    emb = clip_model.encode_image(img_tensor)
                    emb = F.normalize(emb, p=2, dim=-1)
                    
                    for idx, emb_row in zip(valid_indices, emb):
                        product_embeds[batch_pids[idx]] = emb_row.cpu()
                        
    # 2. Encode bundle crops and score against candidates
    print("  Encoding bundle crops and scoring with heavy model...")
    reranked_predictions = {}
    fast_weight = 1.0 - heavy_weight
    
    # Group crops by bundle if passed as a list of CropItem tuples
    bundle_crops_dict = {}
    if isinstance(bundle_crops, list):
        for item in bundle_crops:
            # CropItem is (bundle_id, box_idx, pil_image)
            bundle_id = item[0]
            img = item[2]
            if bundle_id not in bundle_crops_dict:
                bundle_crops_dict[bundle_id] = []
            bundle_crops_dict[bundle_id].append(img)
    else:
        bundle_crops_dict = bundle_crops
        
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=(device=="cuda")):
            for bundle_id, preds in tqdm(predictions.items(), desc="Heavy Bundle Rerank"):
                crops = bundle_crops_dict.get(bundle_id, [])
                if not crops:
                    reranked_predictions[bundle_id] = preds
                    continue
                    
                # Encode crops
                proc_crops = []
                for c in crops:
                    try:
                        proc_crops.append(preprocess(c.convert("RGB")))
                    except Exception:
                        pass
                
                if not proc_crops:
                    reranked_predictions[bundle_id] = preds
                    continue
                    
                crop_tensor = torch.stack(proc_crops).to(device)
                crop_emb = clip_model.encode_image(crop_tensor)
                crop_emb = F.normalize(crop_emb, p=2, dim=-1) # (num_crops, dim)
                
                # Compare against candidate products
                new_preds = []
                for pid, fast_score in preds:
                    if pid in product_embeds:
                        p_emb = product_embeds[pid].to(device).unsqueeze(1) # (dim, 1)
                        # dot product of each crop with product
                        sims = torch.mm(crop_emb, p_emb).squeeze(1) # (num_crops,)
                        heavy_score = sims.max().item() # Take best crop match
                        
                        final_score = (fast_weight * fast_score) + (heavy_weight * heavy_score)
                    else:
                        final_score = fast_score
                        
                    new_preds.append((pid, final_score))
                
                reranked_predictions[bundle_id] = sorted(new_preds, key=lambda x: x[1], reverse=True)

    # Free memory
    del clip_model
    torch.cuda.empty_cache()

    return reranked_predictions

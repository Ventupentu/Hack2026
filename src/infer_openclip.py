"""Inference using our trained OpenCLIP model."""

import argparse
import random
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import open_clip

class AssetImageDataset(Dataset):
    def __init__(self, asset_ids, image_map, transform, text_map=None, tokenizer=None):
        self.asset_ids = list(asset_ids)
        self.image_map = image_map
        self.transform = transform
        self.text_map = text_map
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.asset_ids)

    def __getitem__(self, idx):
        asset_id = self.asset_ids[idx]
        image_path = self.image_map.get(asset_id)
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(image)
        except Exception:
            img_tensor = self.transform(Image.new("RGB", (224, 224), (0,0,0)))
            
        out = {"id": asset_id, "img": img_tensor}
        
        # Load text if provided
        if self.text_map is not None and self.tokenizer is not None:
            text = self.text_map.get(asset_id, "")
            if text:
                out["text"] = self.tokenizer(text).squeeze(0)
                
        return out

@torch.inference_mode()
def encode_assets(asset_ids, image_map, model, transform, device, batch_size, num_workers, use_amp, text_map=None, tokenizer=None):
    ids = [asset_id for asset_id in asset_ids if asset_id in image_map]
    if not ids: 
        return [], torch.zeros((0, 768), dtype=torch.float32)

    dataset = AssetImageDataset(ids, image_map=image_map, transform=transform, text_map=text_map, tokenizer=tokenizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    output_ids = []
    output_embeddings = []
    amp_enabled = use_amp and device.type == "cuda"
    
    for batch in tqdm(loader, desc=f"Encoding {len(dataset)} images", leave=False):
        batch_ids = batch["id"]
        batch_images = batch["img"].to(device, non_blocking=True)
        output_ids.extend(list(batch_ids))
        
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if "text" in batch and batch["text"] is not None:
                batch_texts = batch["text"].to(device, non_blocking=True)
                features = model(batch_images, text=batch_texts)
            else:
                features = model(batch_images)
                
        features = F.normalize(features.float(), p=2, dim=1)
        output_embeddings.append(features.cpu())

    if not output_embeddings:
        return [], torch.zeros((0, 768), dtype=torch.float32)

    return output_ids, torch.cat(output_embeddings, dim=0)

def retrieve_topk_product_ids(query_embeddings, product_embeddings, product_ids, k):
    M = query_embeddings.shape[0]
    preds = []
    for i in range(M):
        sims = query_embeddings[i:i+1] @ product_embeddings.T
        topk_idx = torch.topk(sims.squeeze(0), min(k, len(product_ids))).indices.tolist()
        preds.append([product_ids[idx] for idx in topk_idx])
    return preds

def build_image_map(image_dir):
    return {p.stem: p for p in image_dir.iterdir() if p.is_file()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles-csv", type=Path, default=Path("data/bundles_dataset.csv"))
    parser.add_argument("--products-csv", type=Path, default=Path("data/product_dataset.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/bundles_product_match_train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/bundles_product_match_test.csv"))
    parser.add_argument("--bundle-images-dir", type=Path, default=Path("data/bundle_images"))
    parser.add_argument("--product-images-dir", type=Path, default=Path("data/product_images"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/retrieval_openclip/submission.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/retrieval_openclip/best.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-n-submit", type=int, default=15)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    
    bundle_image_map = build_image_map(args.bundle_images_dir)
    product_image_map = build_image_map(args.product_images_dir)
    
    test_bundle_ids = test_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()
    
    products_df = pd.read_csv(args.products_csv)
    # create a text map for product descriptions
    product_text_map = {}
    if "product_description" in products_df.columns:
        product_text_map = dict(zip(products_df["product_asset_id"].astype(str), products_df["product_description"].fillna("").astype(str)))
        
    product_ids_all = products_df["product_asset_id"].astype(str).tolist()
    product_ids = [pid for pid in product_ids_all if pid in product_image_map]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading OpenCLIP model from {args.checkpoint}...")
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    clip_model.load_state_dict(checkpoint["model"])
    clip_model.eval().to(device)

    class ImageEncoder(torch.nn.Module):
        def __init__(self, clip): 
            super().__init__()
            self.clip = clip
        def forward(self, x, text=None): 
            if text is not None:
                image_features = self.clip.encode_image(x)
                text_features = self.clip.encode_text(text)
                return image_features + text_features
            return self.clip.encode_image(x)

    model = ImageEncoder(clip_model)

    print(f"Encoding {len(product_ids)} products...")
    encoded_product_ids, product_embeddings = encode_assets(
        asset_ids=product_ids, image_map=product_image_map, model=model, transform=preprocess_val,
        device=device, batch_size=args.batch_size, num_workers=args.num_workers, use_amp=args.amp,
        text_map=product_text_map, tokenizer=tokenizer
    )

    print(f"Encoding {len(test_bundle_ids)} test bundles...")
    test_ids_encoded, test_embeddings = encode_assets(
        asset_ids=test_bundle_ids, image_map=bundle_image_map, model=model, transform=preprocess_val,
        device=device, batch_size=args.batch_size, num_workers=args.num_workers, use_amp=args.amp
    )

    print("Retrieving top-15 products...")
    test_predictions = retrieve_topk_product_ids(
        query_embeddings=test_embeddings, product_embeddings=product_embeddings, 
        product_ids=encoded_product_ids, k=args.top_n_submit
    )

    pred_map = {b_id: preds for b_id, preds in zip(test_ids_encoded, test_predictions)}

    popular_products = train_df["product_asset_id"].astype(str).tolist()
    fallback_products = [pid for pid, _ in Counter(popular_products).most_common(args.top_n_submit)]

    submission_rows = []
    for bundle_id in test_bundle_ids:
        preds = pred_map.get(bundle_id, fallback_products)[:args.top_n_submit]
        # Pad with fallback if needed
        if len(preds) < args.top_n_submit:
            for fallback_pid in fallback_products:
                if fallback_pid not in preds:
                    preds.append(fallback_pid)
                if len(preds) == args.top_n_submit:
                    break
                    
        for product_id in preds:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": product_id})

    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(submission_rows).to_csv(args.submission_out, index=False)
    print(f"Saved submission to {args.submission_out}")

if __name__ == "__main__":
    main()

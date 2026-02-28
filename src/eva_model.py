import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from detection import ClothingYOLODetector

# --- PATH CONFIGURATION ---
BUNDLE_IMAGES_DIR = Path("/scratch/tesla8/sgrodriguez23/images/bundle_images")
PRODUCT_IMAGES_DIR = Path("/scratch/tesla8/sgrodriguez23/images/product_images")

CATALOG_CSV = "data/product_dataset.csv"
TEST_CSV = "data/bundles_product_match_test.csv"
EMBEDDINGS_FILE = "data/catalog_embeddings.npy" 

MAX_PER_CATEGORY = 3  
K_SEARCH = 60         

TRAIN_CSV = "data/bundles_product_match_train.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0, help="0 for the first half, 1 for the second")
    parser.add_argument("--total_parts", type=int, default=1, help="Number of parts to split the dataset into")
    return parser.parse_args()


class CatalogDataset(Dataset):
    def __init__(self, df, images_dir, preprocess):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{row['product_asset_id']}.jpg"
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img)
        except Exception:
            # SigLIP usa resolución 384 por defecto
            img_tensor = torch.zeros((3, 384, 384))
        return img_tensor


def encode_catalog(df_catalog, model, preprocess, device, batch_size=256):
    print(f"Encoding catalog with SigLIP... (batch_size={batch_size})")

    dataset = CatalogDataset(df_catalog, PRODUCT_IMAGES_DIR, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=False,
    )

    all_embeddings = []

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for images_tensor in tqdm(dataloader):
            images_tensor = images_tensor.to(device, non_blocking=True)
            
            # En OpenCLIP, el método es encode_image
            img_features = model.encode_image(images_tensor)
            img_features = F.normalize(img_features, p=2, dim=-1)

            all_embeddings.append(img_features.cpu().to(torch.float32).numpy())

    return np.vstack(all_embeddings)


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Starting part {args.part + 1}/{args.total_parts} on {device}")

    # 1. LOAD MODELS 
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli', device=device)
    model.eval()

    print("Loading YOLOv8...")
    yolo_detector = ClothingYOLODetector()

    # 2. PREPARE CATALOG DATA
    df_catalog = pd.read_csv(CATALOG_CSV)
    catalog_ids = df_catalog['product_asset_id'].tolist()
    id_to_category = dict(zip(df_catalog['product_asset_id'], df_catalog['product_description']))

    if Path(EMBEDDINGS_FILE).exists():
        print("Loading pre-computed Marqo embeddings...")
        catalog_embeddings = np.load(EMBEDDINGS_FILE)
    else:
        catalog_embeddings = encode_catalog(df_catalog, model, preprocess, device)
        np.save(EMBEDDINGS_FILE, catalog_embeddings)

    # Build FAISS index
    dimension = catalog_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(catalog_embeddings)

    # 3. SPLIT DATASET
    df_test = pd.read_csv(TEST_CSV)
    if args.total_parts > 1:
        chunk_size = len(df_test) // args.total_parts
        start_idx = args.part * chunk_size
        end_idx = len(df_test) if args.part == args.total_parts - 1 else start_idx + chunk_size
        df_test = df_test.iloc[start_idx:end_idx]

    # 4. INFERENCE
    results = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        bundle_id = row['bundle_asset_id']
        bundle_img_path = BUNDLE_IMAGES_DIR / f"{bundle_id}.jpg"

        if not bundle_img_path.exists():
            results.append({'bundle_asset_id': bundle_id, 'product_asset_id': ""})
            continue

        boxes = yolo_detector.detect_boxes_without_scores(bundle_img_path)
        original_img = Image.open(bundle_img_path).convert('RGB')
        width, height = original_img.size

        if not boxes: boxes = [(0, 0, width, height)]

        crop_tensors = []
        valid_boxes = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = original_img.crop((x1, y1, x2, y2))
            crop_tensors.append(preprocess(crop))
            valid_boxes.append((x1, y1, x2, y2))

        if not crop_tensors:
            results.append({'bundle_asset_id': bundle_id, 'product_asset_id': ""})
            continue

        crop_tensors = torch.stack(crop_tensors).to(device)
        with torch.no_grad():
            crop_features = model.encode_image(crop_tensors)
            crop_features = F.normalize(crop_features, p=2, dim=-1).cpu().numpy().astype(np.float32)

        distances, indices = index.search(crop_features, k=K_SEARCH)

        candidates = []
        for i in range(len(valid_boxes)):
            for j in range(K_SEARCH):
                candidates.append((distances[i][j], catalog_ids[indices[i][j]]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        
        top_15_products = []
        seen_products = set()
        category_counts = {}

        for score, product_id in candidates:
            if product_id not in seen_products:
                category = str(id_to_category.get(product_id, "UNKNOWN"))
                current_count = category_counts.get(category, 0)
                if current_count < MAX_PER_CATEGORY:
                    seen_products.add(product_id)
                    top_15_products.append(product_id)
                    category_counts[category] = current_count + 1
            if len(top_15_products) == 15: break

        if len(top_15_products) < 15:
            for score, product_id in candidates:
                if product_id not in seen_products:
                    seen_products.add(product_id)
                    top_15_products.append(product_id)
                if len(top_15_products) == 15: break

        results.append({'bundle_asset_id': bundle_id, 'product_asset_id': " ".join(top_15_products)})

    df_submission = pd.DataFrame(results)
    df_submission.to_csv(f"submission_part_{args.part}.csv", index=False)


def evaluate_on_train(num_samples=800):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n--- EVALUATION WITH SigLIP Shape-Optimized ({num_samples} samples) ---")

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli', device=device)
    yolo_detector = ClothingYOLODetector()

    df_catalog = pd.read_csv(CATALOG_CSV)
    catalog_ids = df_catalog['product_asset_id'].tolist()
    id_to_category = dict(zip(df_catalog['product_asset_id'], df_catalog['product_description']))

    if Path(EMBEDDINGS_FILE).exists():
        catalog_embeddings = np.load(EMBEDDINGS_FILE)
    else:
        catalog_embeddings = encode_catalog(df_catalog, model, preprocess, device)
        np.save(EMBEDDINGS_FILE, catalog_embeddings)

    index = faiss.IndexFlatIP(catalog_embeddings.shape[1])
    index.add(catalog_embeddings)

    df_train = pd.read_csv(TRAIN_CSV).sample(n=num_samples, random_state=42)
    hits_top1, hits_top15, total = 0, 0, 0

    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        bundle_id, true_id = row['bundle_asset_id'], row['product_asset_id']
        path = BUNDLE_IMAGES_DIR / f"{bundle_id}.jpg"
        if not path.exists(): continue
        
        total += 1
        boxes = yolo_detector.detect_boxes_without_scores(path)
        img = Image.open(path).convert('RGB')
        if not boxes: boxes = [(0, 0, img.size[0], img.size[1])]

        crops = [preprocess(img.crop(box)) for box in boxes]
        crops_t = torch.stack(crops).to(device)

        with torch.no_grad():
            feats = model.encode_image(crops_t)
            feats = F.normalize(feats, p=2, dim=-1).cpu().numpy().astype(np.float32)

        dist, idx = index.search(feats, k=K_SEARCH)
        
        cands = []
        for i in range(len(boxes)):
            for j in range(K_SEARCH):
                cands.append((dist[i][j], catalog_ids[idx[i][j]]))
        cands.sort(key=lambda x: x[0], reverse=True)

        res, seen, cats = [], set(), {}
        for _, pid in cands:
            cat = id_to_category.get(pid, "UNK")
            if pid not in seen and cats.get(cat, 0) < MAX_PER_CATEGORY:
                seen.add(pid); res.append(pid); cats[cat] = cats.get(cat, 0) + 1
            if len(res) == 15: break
        
        if res:
            if true_id == res[0]: hits_top1 += 1
            if true_id in res: hits_top15 += 1

    print(f"\nResultados SigLIP Shape-Optimized:\nTop-1: {hits_top1/total:.2%}\nTop-15: {hits_top15/total:.2%}")

if __name__ == "__main__":
    evaluate_on_train(800) # Cambia a main() para generar la submission final
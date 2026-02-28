import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from dataset import FashionMatchDataset, get_train_transforms
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fallback to older PyTorch imports if torch.amp is missing
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class FashionMatchingModel(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # remove classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        # Normalize embeddings using L2 normalization
        return nn.functional.normalize(embeddings, p=2, dim=1)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel processing!")

    # 1. Dataset & Transforms setup
    bundle_transform, product_transform = get_train_transforms(image_size=256)
    
    # We use the train CSV for pairs
    train_dataset = FashionMatchDataset(
        csv_file='data/bundles_product_match_train.csv',
        bundle_dir='data/bundle_images',
        product_dir='data/product_images',
        bundle_transform=bundle_transform,
        product_transform=product_transform
    )

    # Dataloader: num_workers is crucial! It runs Albumentations transformations on CPU threads
    # preparing batches while GPUs are busy training.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=8,    # Adjust based on CPU core count
        pin_memory=True   # Speeds up tensor transfer to GPU
    )

    # 2. Model, Optimizer, and Loss Definition
    model = FashionMatchingModel(embedding_dim=256)
    
    # 3. Apply Multi-GPU Support via DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  
    
    model.to(device)

    # CosineEmbeddingLoss tries to maximize cosine similarity for matches (target=1)
    # and minimize it for non-matches (target=-1) with a configured margin.
    criterion = nn.CosineEmbeddingLoss(margin=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler() # Automatic Mixed Precision (AMP) to maximize speed

    # 4. Training Loop
    num_epochs = 10
    model.train()
    
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        for batch in loop:
            # Transfer batch to GPU asynchronously
            bundles = batch["bundle"].to(device, non_blocking=True)
            products = batch["product"].to(device, non_blocking=True)
            
            # Create negative samples easily by shifting the product batch by 1
            # (assuming batch size > 1)
            neg_products = torch.roll(products, shifts=1, dims=0)
            
            # Combine positive pairs and generated negative pairs
            all_bundles = torch.cat([bundles, bundles], dim=0)
            all_products = torch.cat([products, neg_products], dim=0)
            
            # Targets: 1 for correct product array, -1 for the shifted array
            targets = torch.cat([
                torch.ones(bundles.size(0)), 
                -torch.ones(bundles.size(0))
            ], dim=0).to(device)

            optimizer.zero_grad()
            
            # Forward pass using Mixed Precision
            with autocast('cuda'):
                bundle_emb = model(all_bundles)
                product_emb = model(all_products)
                loss = criterion(bundle_emb, product_emb, targets)

            # Optimizing step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(train_loader):.4f}")

if __name__ == '__main__':
    train()

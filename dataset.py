import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FashionMatchDataset(Dataset):
    """
    Dataset for Fashion Product Matching.
    Loads a bundle image and a corresponding product image.
    """
    def __init__(self, csv_file, bundle_dir, product_dir, 
                 bundle_transform=None, product_transform=None):
        self.data_df = pd.read_csv(csv_file)
        self.bundle_dir = bundle_dir
        self.product_dir = product_dir
        self.bundle_transform = bundle_transform
        self.product_transform = product_transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bundle_id = self.data_df.iloc[idx, 0]
        product_id = self.data_df.iloc[idx, 1]

        bundle_path = os.path.join(self.bundle_dir, f"{bundle_id}.jpg")
        product_path = os.path.join(self.product_dir, f"{product_id}.jpg")

        bundle_img = self._load_image(bundle_path)
        product_img = self._load_image(product_path)

        if self.bundle_transform:
            bundle_img = self.bundle_transform(image=bundle_img)["image"]
        if self.product_transform:
            product_img = self.product_transform(image=product_img)["image"]

        # Note: If no transform to tensor is provided, it returns numpy arrays.
        return {
            "bundle": bundle_img,
            "product": product_img,
            "bundle_id": bundle_id,
            "product_id": product_id
        }

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            # Fallback if image not found, create a black image
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

def get_train_transforms(image_size=256):
    """
    Returns data augmentation pipelines for bundles and products.
    """
    # Bundle augmentation: Heavy augmentations for domain gap (wrinkles, light, occlusions)
    bundle_transform = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=1.0),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3), # Simulate wrinkles
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20, p=0.4), # NO hue shift!
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # A.CoarseDropout(max_holes=1, max_height=int(image_size*0.2), max_width=int(image_size*0.2), fill_value=0, p=0.3),
        # CoarseDropout has been updated to CoarseDropout directly or using Cutout. 
        # Using newer syntax for CoarseDropout usually fill_value is used.
        # It's better to omit it if it conflicts, or just use standard:
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Product augmentation: Lighter, mostly standardizations, maybe minor color/light changes
    product_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return bundle_transform, product_transform

def get_val_transforms(image_size=256):
    """
    Validation transforms (No augmentations, only resizing and normalization).
    """
    transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transform, transform

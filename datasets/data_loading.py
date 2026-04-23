import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

class ADE20KDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir,
            img_name.replace(".jpg", ".png")
        )

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask
        
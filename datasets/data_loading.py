import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]

class ADE20KDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform

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

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)

        if self.image_transform:
            image = self.image_transform(image)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask
        
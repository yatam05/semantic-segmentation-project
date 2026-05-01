import torch.nn as nn
import torch
import argparse
import yaml
import cv2
from models.model import initialize_model, save_model
from torchmetrics.segmentation import MeanIoU
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from datasets.data_loading import ADE20KDataset
import torch.nn.functional as fct
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

with open("config.yml") as f:
    config = yaml.safe_load(f)

num_classes = config["dataset"]["num_classes"]
image_size = config["dataset"]["image_size"]
learning_rate = args.lr if args.lr else config["training"]["lr"]
weight_decay = args.weight_decay if args.weight_decay else config["training"]["weight_decay"]
training_epochs = args.epochs if args.epochs else config["training"]["epochs"]
batch_size = args.batch_size if args.batch_size else config["training"]["batch_size"]
total_epochs = config["training"]["total_epochs"]

transform = A.Compose([
    A.RandomResizedCrop(
        (image_size, image_size),
        scale=(0.5, 1.0),
        ratio=(0.75, 1.33),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST
    ),

    A.HorizontalFlip(p=0.5),

    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=0.1,
        rotate=(-10, 10),
        shear=(-5, 5),
        p=0.2
    ),

    A.OneOf([
        A.GaussianBlur(5),
        A.MotionBlur(5),
    ], p=0.3),

    A.GaussNoise(p=0.3),

    A.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05,
        p=0.5
    ),

    A.CoarseDropout(
      num_holes_range=(1, 4),
      hole_height_range=(8, 24),
      hole_width_range=(8, 24),
      p=0.3
    ),

    A.Normalize(),
    ToTensorV2()
])

train_dataset = ADE20KDataset(
    image_dir="data/ADEChallengeData2016/images/training",
    mask_dir="data/ADEChallengeData2016/annotations/training",
    transform=transform
)

loader = DataLoader(train_dataset, batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, optimizer, start_epoch, scaler, scheduler = initialize_model('Training', num_classes, learning_rate, weight_decay, total_epochs, device)
print(f"Using device: {device}")
criterion = nn.CrossEntropyLoss(ignore_index=255)

for epoch in range(start_epoch, start_epoch+training_epochs):
    model.train()
    miou_metric = MeanIoU(num_classes)
    miou_metric = miou_metric.to(device)
    running_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with autocast():
          outputs = model(images)["out"]
          loss = criterion(outputs, masks)

        preds = torch.argmax(outputs, dim=1)
        masks_resized = fct.interpolate(
            masks.unsqueeze(1).float(),   
            size=preds.shape[1:],       
            mode='nearest'                
        ).squeeze(1).long()
        miou_metric.update(preds, masks_resized)   

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()


    avg_loss = running_loss / len(loader)
    scheduler.step()
    print(f"Epoch {epoch}, loss = {avg_loss}")
    miou = miou_metric.compute()
    print("Training Mean IoU:", str(miou))
    save_model(model, optimizer, epoch, scaler, scheduler)


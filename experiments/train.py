import torch.nn as nn
import torch
import argparse
import yaml
from models.model import initialize_model, save_model
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.data_loading import ADE20KDataset

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std)
])

train_dataset = ADE20KDataset(
    image_dir="data/ADEChallengeData2016/images/training",
    mask_dir="data/ADEChallengeData2016/annotations/training",
    image_transform=image_transform
)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

with open("checkpoints/config.yml") as f:
    config = yaml.safe_load(f)

num_classes = config["dataset"]["num_classes"]
learning_rate = args.lr if args.lr else config["training"]["lr"]
training_epochs = args.epochs if args.epochs else config["training"]["epochs"]
batch_size = args.batch_size if args.batch_size else config["training"]["batch_size"]

loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=True)
model, optimizer, start_epoch = initialize_model('Training', num_classes, learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(start_epoch, start_epoch+training_epochs):
    model.train()
    running_loss = 0

    for images, masks in loader:
        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch}, loss = {avg_loss}")
    save_model(model, optimizer, epoch)

















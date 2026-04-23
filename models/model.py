from torchvision.models.segmentation import deeplabv3_resnet50
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch
import os

CHECKPOINT_PATH = 'checkpoints/checkpoint.pth'

def build_deeplab(num_classes):

    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(
        256,
        num_classes,
        kernel_size=1
    )
    return model

def initialize_model(mode, num_classes, learning_rate, weight_decay, total_epochs, device):
    
    model = build_deeplab(num_classes)
    model = model.to(device)
    optimizer = None
    scaler = GradScaler()
    start_epoch = 0
    scheduler = None

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if mode == "Training":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if "optimizer_state_dict" in checkpoint:
              optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]

            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
           
            if "scaler_state_dict" in checkpoint:
              scaler.load_state_dict(checkpoint["scaler_state_dict"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
            if "scheduler_state_dict" in checkpoint:
              scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
              
            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        elif mode == "Testing":
            print(f"Checkpoint loaded. Evaluating on validation data.")
    else:
        print("Checkpoint not found. Using pretrained model.")

    if mode == 'Training':
        return model, optimizer, start_epoch, scaler, scheduler
    elif mode == 'Testing':
        return model

def save_model(model, optimizer, current_epoch, scaler, scheduler):
    checkpoint = {
        'epoch': current_epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {current_epoch} to {CHECKPOINT_PATH}")
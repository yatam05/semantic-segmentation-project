from torchvision.models.segmentation import deeplabv3_resnet50
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

def initialize_model(mode, num_classes, learning_rate):
    
    model = build_deeplab(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        if mode == "Training":
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        elif mode == "Testing":
            print(f"Checkpoint loaded. Evaluating on validation data.")
    else:
        print("Checkpoint not found. Using pretrained model.")

    if mode == 'Training':
        return model, optimizer, start_epoch
    elif mode == 'Testing':
        return model

def save_model(model, optimizer, current_epoch):
    checkpoint = {
        'epoch': current_epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {current_epoch} to {CHECKPOINT_PATH}")
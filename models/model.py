from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn

def build_deeplab(num_classes):

    model = deeplabv3_resnet50(weights="DEFAULT")

    model.classifier[4] = nn.Conv2d(
        256,
        num_classes,
        kernel_size=1
    )

    return model
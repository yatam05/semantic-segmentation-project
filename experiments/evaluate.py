import torch
import yaml
import argparse
import cv2
from datasets.data_loading import ADE20KDataset
from models.model import initialize_model
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as fct
import matplotlib.pyplot as plt
import numpy as np

def compare_prediction(image_tensor, pred_mask, gt_mask):
    img = image_tensor.cpu().permute(1,2,0).numpy() 
    img = img * norm_std + norm_mean            
    img = np.clip(img, 0, 1)                        
    pred_colors = plt.cm.get_cmap('tab20', num_classes)
    pred_colored_mask = pred_colors(pred_mask.cpu().numpy() / num_classes)
    gt_colors = plt.cm.get_cmap('tab20', num_classes)
    gt_colored_mask = gt_colors(gt_mask.cpu().numpy() / num_classes)
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.title("Prediction")
    plt.imshow(pred_colored_mask)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Ground Truth")
    plt.imshow(gt_colored_mask)
    plt.axis('off')
    
    plt.show()

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]

parser = argparse.ArgumentParser()
parser.add_argument("--visualize_results", type=bool)
args = parser.parse_args()
visualize_results = args.visualize_results if args.visualize_results else False

with open("checkpoints/config.yml") as f:
    config = yaml.safe_load(f)
num_classes = config["dataset"]["num_classes"]
image_size = config["dataset"]["image_size"]

transform = A.Compose([
    A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
    A.Normalize(),
    ToTensorV2()
])

validation_dataset = ADE20KDataset(
    image_dir="data/ADEChallengeData2016/images/validation",
    mask_dir="data/ADEChallengeData2016/annotations/validation",
    transform=transform
)

loader = DataLoader(validation_dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_model("Testing", num_classes, None, None, None, device)
model.eval()

miou_metric = MeanIoU(num_classes)
miou_metric = miou_metric.to(device)

with torch.no_grad():
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)["out"]
        preds = torch.argmax(outputs, dim=1)

        masks_resized = fct.interpolate(
            masks.unsqueeze(1).float(),   
            size=preds.shape[1:],       
            mode='nearest'                
        ).squeeze(1).long()               
        
        miou_metric.update(preds, masks_resized)
        print(f'Evaluating image ', i)
        if visualize_results:
          compare_prediction(images[0], preds[0], masks_resized[0])
        
miou = miou_metric.compute()
print("Validation Mean IoU:", str(miou))
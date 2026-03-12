import torch
from datasets.data_loading import ADE20KDataset
from models.model import build_deeplab
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.segmentation import MeanIoU
import torch.nn.functional as fct
import matplotlib.pyplot as plt
import numpy as np

def compare_prediction(image_tensor, pred_mask, gt_mask):
    img = image_tensor.permute(1,2,0).numpy() 
    img = img * norm_std + norm_mean            
    img = np.clip(img, 0, 1)                        
    pred_colors = plt.cm.get_cmap('tab20', num_classes)
    pred_colored_mask = pred_colors(pred_mask.numpy() / num_classes)
    gt_colors = plt.cm.get_cmap('tab20', num_classes)
    gt_colored_mask = gt_colors(gt_mask.numpy() / num_classes)
    
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
num_classes = 150

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=norm_mean,
        std=norm_std
    )
])

dataset = ADE20KDataset(
    image_dir="data/ADEChallengeData2016/images/validation",
    mask_dir="data/ADEChallengeData2016/annotations/validation",
    transform=transform
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = build_deeplab(num_classes)
model.eval()

miou_metric = MeanIoU(num_classes)

all_preds = []
all_masks = []

with torch.no_grad():
    for i, (images, masks) in enumerate(loader):
        outputs = model(images)["out"]
        preds = torch.argmax(outputs, dim=1)

        masks_resized = fct.interpolate(
            masks.unsqueeze(1).float(),   
            size=preds.shape[1:],       
            mode='nearest'                
        ).squeeze(1).long()               
        
        all_preds.append(preds)
        all_masks.append(masks_resized)
        miou_metric.update(preds, masks_resized)

        print("Evaluating image " + str(i))
        compare_prediction(images[0], preds[0], masks_resized[0])

miou = miou_metric.compute()
print("Mean IoU:", str(miou))
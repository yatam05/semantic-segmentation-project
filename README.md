# Computer Vision Project – Segmentation Pipeline

## Overview

This project implements an image segmentation pipeline using a deep learning model.
It supports training, evaluation, and deployment (including optional acceleration on NVIDIA Jetson devices).

---

## Features (WIP)

* Train segmentation model on custom datasets
* Evaluate using mIoU and qualitative outputs
* Run inference on images or live camera
* Deployment with TensorRT on Jetson

---

## Project Structure

```
.
├── data/
├── datasets/
│   ├── data_loading.py
├── models/
│   ├── model.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
├── checkpoints/
│   ├── config.yml
│   ├── checkpoint.pth
├── environment.yml
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/yatam05/semantic-segmentation-project
cd semantic-segmentation-project
```

### 2. Create environment (recommended)

```
conda create -n segmentation-project python=3.10
conda activate segmentation-project
```

### 3. Install dependencies

```
pip install -r environment.yml
```

---

## Dataset Setup

Organize your dataset like this:

```
data/
├── ADEChallengeData2016/
│   ├── annotations/
│   ├── images/
```

* Images and masks must have matching filenames
* Masks should contain class labels (not RGB unless specified)

---

## Training

Run:

```
python -n experiments.train
```

Common options (edit in script or config):

* `--epochs`
* `--batch_size`
* `--lr`
* `--weight_decay`

Checkpoints will be saved to:

```
checkpoints/
```

---

## Evaluation

Run:

```
python -n experiments.evaluate
```

Outputs:

* mIoU score
* Optional visualizations

---


## Real-World Data Workflow (WIP)

1. Collect images using camera
2. Generate masks using current model
3. (Optional) Correct masks using annotation tools
4. Add to dataset
5. Fine-tune model

---

## Jetson Deployment (WIP)

### Export model

```

```

### Optimize with TensorRT

Use NVIDIA TensorRT tools on the Jetson device.

### Run inference

```

```

---

## Requirements

* Python 3.10+
* PyTorch
* OpenCV
* NumPy

(See `requirements.txt` for full list)

---

## Notes

* Performance may vary between dataset and real-world camera input
* Fine-tuning on real data is recommended for best results

---

## Troubleshooting

**Low accuracy**

* Add more real-world data
* Check mask formatting

**Slow inference**

* Reduce image size
* Use TensorRT (Jetson)

---


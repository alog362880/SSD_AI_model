# SSD Object Detection on COCO Dataset

A Google Colab notebook that fine-tunes an **SSDLite320 with MobileNetV3-Large** backbone on a custom subset of the COCO 2017 dataset for real-world object detection across 8 traffic-relevant categories.

---

## Overview

This project trains a lightweight Single Shot Detector (SSD) to identify common objects in traffic and urban scenes. It uses a pretrained MobileNetV3-Large backbone and fine-tunes the detection head on a balanced sample from COCO 2017.

**Target classes:**
- Person
- Car
- Truck
- Motorcycle
- Bus
- Bicycle
- Traffic Light
- Stop Sign

---

## Model Architecture

| Component | Details |
|---|---|
| Model | SSDLite320 |
| Backbone | MobileNet V3 Large (pretrained on ImageNet) |
| Input resolution | 320 × 320 |
| Output classes | 9 (8 categories + background) |
| Framework | PyTorch / TorchVision |

---

## Dataset

Uses the [COCO 2017](https://cocodataset.org/) dataset, automatically downloaded and stored in Google Drive.

- **Training:** 150 images per class → ~1,200 total
- **Validation:** 30 images per class → ~240 total
- Annotations are parsed from the official COCO JSON format and converted to `[x1, y1, x2, y2]` bounding boxes

---

## Requirements

- Google Colab (with GPU runtime — T4 recommended)
- Google Drive (for persistent dataset and model storage)
- Python packages (pre-installed on Colab):

```
torch
torchvision
numpy
Pillow
matplotlib
opencv-python (cv2)
tqdm
requests
```

---

## Getting Started

1. **Open the notebook in Google Colab** and connect to a GPU runtime (`Runtime > Change runtime type > T4 GPU`).

2. **Mount Google Drive** — the first cell handles this automatically.

3. **Download COCO 2017** — the notebook downloads train/val images and annotations directly from [cocodataset.org](https://cocodataset.org/) and stores them permanently in `MyDrive/COCO/`. This only runs once; subsequent runs skip the download if files already exist.

4. **Run all cells** from top to bottom.

---

## Training

- **Optimizer:** SGD (momentum=0.9, weight decay=5e-4)
- **Learning rate:** 0.01 → 0.001 (StepLR scheduler, step size 5, γ=0.1)
- **Epochs:** 50
- **Batch size:** 10
- **Random seed:** 42 (reproducible splits)

The best model checkpoint (lowest validation loss) is saved automatically to Google Drive during training. A `latest_checkpoint.pth` is also saved each epoch to allow resuming.

Checkpoints are saved to:
```
/content/drive/MyDrive/COCO/checkpoints/best_detection_model.pth
/content/drive/MyDrive/COCO/ssd_detection_model.pth
```

---

## Inference

After training, the notebook includes an inference cell that:
1. Loads the saved model weights
2. Prompts you to upload an image
3. Runs detection and draws bounding boxes with class labels and confidence scores

---

## Results

The model was trained for 50 epochs. Sample training progress:

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 10.23 | 9.48 |
| 10 | 7.68 | 8.21 |
| 25 | 6.99 | 8.17 |
| 37 | 6.47 | 7.99 ✓ best |
| 50 | 6.32 | 8.02 |

---

## Project Structure

```
SSD_code.ipynb        # Main training & inference notebook
```

Google Drive layout after running:
```
MyDrive/COCO/
├── train2017/                    # ~118k training images
├── val2017/                      # ~5k validation images
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── checkpoints/
│   ├── best_detection_model.pth
│   └── latest_checkpoint.pth
└── ssd_detection_model.pth
```

---

## Notes

- The SSDLite320 model expects a fixed 320×320 input — random crop/resize augmentations are disabled for this reason.
- The COCO download (~20 GB total) only runs once; subsequent runs detect existing folders and skip.
- Training uses `torch.no_grad()` during validation to compute val loss efficiently without updating weights.

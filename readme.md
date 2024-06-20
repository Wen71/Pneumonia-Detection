# Pneumonia Detection in Chest X-ray Images
This project is to depict if a patient has pneumonia, one of the lung disease, by detecting chest x-rays using DL model. 

## Installation
Use the package manager [pip]((https://pip.pypa.io/en/stable/)) to install the following packages for this project.
```bash
pip install torch
pip install torchvision
pip install transformers
pip install matplotlib
pip install pillow
pip install numpy
```

## Dataset
The dataset is from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which has 5,863 x-ray images (JPEG) and 2 categories (Pneumonia/normal).

## Model
I used the ViT (Vision Transformer), that was pretrained on ImageNet and ImageNet-21k datasets. 

```python
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2, ignore_mismatched_sizes=True).to(device)
```

## Result
The accuracy of the model in the test dataset is 93%, which is pretty good. 

## Visualization
Post-training results were used to visualize and localize areas in x-ray images indicating pneumonia, using heatmaps generated from the last convolution layer of each network.
<p align="center">
  <img src="heatmap_normal.png" alt="Normal" width="200"/>
  <img src="heatmap_pne.png" alt="Pneumonia" width="200"/>
</p>

## GUI
![A Gui to see how the model prediction looks like](gui.gif)
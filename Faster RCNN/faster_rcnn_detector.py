# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import torchvision

from PIL import Image
import torchvision.transforms as T


def faster_rcnn_pretrained_model(num_classes):
    # load an instance detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


if torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"

# load fasterrcnn_resnet50_fpn
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)


image1 = T.ToTensor()(Image.open('C:/Users/xwen2/Downloads/14. Deep Learning with PyTorch/opencv-dl-pytorch-week8/faster_rcnn_fine_tuning/FudanPed00066.png'))
bboxes1 = torch.tensor([[248.0, 50.0, 329.0, 351.0]])
labels1 = torch.tensor([1])

image2 = T.ToTensor()(Image.open('C:/Users/xwen2/Downloads/14. Deep Learning with PyTorch/opencv-dl-pytorch-week8/faster_rcnn_fine_tuning/PennPed00011.png'))
bboxes2 = torch.tensor([[92.0, 62.0, 236.0, 344.0], [242.0, 52.0, 301.0, 355.0]])
labels2 = torch.tensor([1, 1])

input_image1 = image1.clone()
input_image2 = image2.clone()

# input its image list
inputs = [input_image1.to(device), input_image2.to(device)]

model.eval()
output = model(inputs)

#print(output)

input_image1 = image1.clone()
target1 = {
    'boxes': bboxes1.clone().to(device),
    'labels' : labels1.clone().to(device)
} 

input_image2 = image2.clone()
target2 = {
    'boxes': bboxes2.clone().to(device),
    'labels' : labels2.clone().to(device)
} 

inputs = [input_image1.to(device), input_image2.to(device)]
targets = [target1, target2]

# change to train mode
model.train()
model(inputs, targets)


model.eval()
print(model)
'''
    This is a script for inference on the fine-tuned efficientnet_v2 model
    This script I have used arguments from cli to load the image
'''

import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

# preparing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model weights(i.e.,: state_dicts) location/path", type=str, required=True)
parser.add_argument("--source", help="Source image location/path", type=str, required=True)
parser.add_argument("--show", help="Boolean value to view classified image", type=bool, default=False)
argv = parser.parse_args()
# extract info from arg values
model_path = argv.model
source_path = argv.source
show_bool = argv.show
# check is path exists
if (not os.path.isfile(model_path)):
    print(f'''"{model_path}" does not exists''')
    exit(0)
if (not os.path.isfile(source_path)):
    print(f'''"{source_path}" does not exists''')
    exit(0)
# print(argv.source)

# create device
device = "cuda" if torch.cuda.is_available() else "cpu"


# load the model from torchvision module
num_classes = 10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
model = models.vit_b_16()
# model.classifier = torch.nn.Linear(in_features=1024, out_features=10, bias=True)
model.heads = torch.nn.Sequential(
    torch.nn.Linear(in_features=768, out_features=100, bias=True),
    torch.nn.Linear(in_features=100, out_features=10, bias=True),
)

# load models state_dict
model.load_state_dict(torch.load(model_path))
model.to(device)


# read the image from source file
source_image = cv2.imread(source_path, flags=1)     # flags = 1 for color image 
# source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
# source_image.resize([32,32,3])

# transform input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224],antialias=True)
])
image = transform(source_image).unsqueeze(dim=1).permute(1,0,2,3)
image = image.to(device)

# model inference
model.eval()
if torch.inference_mode():
    logits = model(image)
    pred_prob = torch.softmax(logits,dim=1)
    conf = torch.max(pred_prob).cpu().item()
    pred = torch.argmax(pred_prob,dim=1)
    pred_class = labels[pred.item()]

print(f"Predicted label: {pred_class}")
# show image
if (show_bool):
    plt.imshow(source_image)
    plt.title(f"{pred_class} -> Conf: {round(conf,2) * 100}")
    plt.show()

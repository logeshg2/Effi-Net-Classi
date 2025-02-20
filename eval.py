'''
    This the script for evaluation of the fine-tuned model on CIFAR-10 dataset
    Model fine-tuned is EfficientNet_v2 -> check 'train.py' for more details
'''

import cv2
import torch 
import torchmetrics
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")

# load the model
num_classes = 10 
model = models.efficientnet_b1()
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4,inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes, bias=True)
)
model.load_state_dict(torch.load(r"D:\Effi-Net-Classi\checkpoints\model1_18_02_25.pt"))
model.to(device)
print("Model device:", next(model.parameters()).device)

# load test dataset for evaluation
# manual transforms like resizing
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) 
])
test_data = datasets.CIFAR10(r"D:\CV-Datasets", 
                    train=False,
                    transform=manual_transforms,
                    download=False,
                    target_transform=None 
                )
batch_size = 32
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# data - visualization 
img = test_data[1][0].permute(1,2,0).numpy()
label = test_data.classes[test_data[1][1]]
# model.eval()
# if (torch.inference_mode()):
#     ig = (test_data[1][0].unsqueeze(dim=1))
#     test_pred = model((ig.permute(1,0,2,3).to(device)))
#     test_pred_classes = torch.softmax(test_pred,dim=1).argmax(dim=1)
# plt.imshow(img); plt.title(test_data.classes[test_pred_classes.item()])
# plt.show()

# accuracy over test data
total_test_acc = 0
model.eval()
if torch.inference_mode():
    for batch, (X,y) in tqdm(enumerate(test_dataloader)):
        X, y = X.to(device), y.to(device)
        test_pred = model(X)
        test_pred_classes = torch.softmax(test_pred,dim=1).argmax(dim=1)
        acc = (test_pred_classes == y).sum().item() / len(test_pred)
        total_test_acc += acc

test_acc = round(total_test_acc / len(test_dataloader),4)
print("------ Evaluation Metrics ------")
print(f"Accuracy: {test_acc*100} %")

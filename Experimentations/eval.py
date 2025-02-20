'''
    This the script for evaluation of the fine-tuned model on CIFAR-10 dataset
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
model = models.densenet121()
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=1024, out_features=100, bias=True),
    torch.nn.Linear(in_features=100, out_features=10, bias=True),
)
model.load_state_dict(torch.load(r".\checkpoints\test3.pt"))
model.to(device)
print("Model device:", next(model.parameters()).device)

# load test dataset for evaluation
# manual transforms like resizing
manual_transforms = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Resize((32, 32), antialias=True)
])
test_data = datasets.CIFAR10(r"D:\CV-Datasets", 
                    train=False,
                    transform=manual_transforms,
                    download=False,
                    target_transform=None 
                )
batch_size = 256
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

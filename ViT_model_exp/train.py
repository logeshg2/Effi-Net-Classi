'''
    Experimenting with ViT models for classification problem 
'''

import torch
import argparse
import torchinfo
from tqdm.auto import tqdm
from engine import train_model, eval_model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

# get arguments from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="Number of epochs to train the model", type=int, default=1)
parser.add_argument("--model_name", help="Model name to save the parameters on", type=str, required=True)
argv = parser.parse_args()
epochs = argv.epochs
model_name = argv.model_name

# datasets and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224, 224], antialias=True)
])
train_data = datasets.CIFAR10(
    root=r"D:\CV-Datasets",
    train=True,
    download=False,
    transform=transform,
    target_transform=None
)
test_data = datasets.CIFAR10(
    root=r"D:\CV-Datasets",
    train=False,
    download=False,
    transform=transform,
    target_transform=None
)
# dataloader
train_dataloader = DataLoader(train_data, batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32,shuffle=True)

# device 
device = "cuda" if torch.cuda.is_available() else "cpu"


### MODELS:
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# freeze the layers of the model
# for param in model.parameters():
#     param.requires_grad = False
model.requires_grad_(False)             # another way of freezing the layers
# modify the model's classifier
model.heads = torch.nn.Sequential(
    torch.nn.Linear(in_features=768, out_features=100, bias=True),
    torch.nn.Linear(in_features=100, out_features=10, bias=True),
)
# unfreeze 'heads' layer
model.heads.requires_grad_(True)
# now put the model parameter into gpu
model.to(device)
# print(next(model.parameters()).device)

### loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

### model training (function is in 'engine.py')
train_model(
    model=model,
    loss_fn=loss_function,
    optimizer=optimizer,
    epochs=epochs,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    device=device
)

### saving the state_dicts of the trainined model
torch.save(model.state_dict(), f"D:\\Effi-Net-Classi\\ViT_model_exp\\checkpoints\\{model_name}.pt")


### eval script
eval_model(
    model=model,
    loss_fn=loss_function,
    test_dataloader=test_dataloader,
    device=device
)

### End of training 
print("####### END #######")
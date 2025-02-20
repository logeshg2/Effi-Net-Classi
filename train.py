'''
    This script uses the model (EfficientNet_v2) from the torchvision.models package.
    The model is fine tuned on CIFAR 10 dataset
'''
import torch
import torchinfo
import torchmetrics
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

# Create a transforms pipeline manually (required for torchvision < 0.13)
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])

# import download CIFAR 10 dataset 
train_data = datasets.CIFAR10(r"D:\CV-Datasets", 
                    train=True,
                    transform=manual_transforms,
                    download=False,
                    target_transform=None 
                )
# test data
test_data = datasets.CIFAR10(r"D:\CV-Datasets", 
                    train=False,
                    transform=manual_transforms,
                    download=False,
                    target_transform=None 
                )
print(f"Total no. of training data: {len(train_data)}\nTotal no. of test data: {len(test_data)}")
img = (train_data[1][0])#.permute(1,2,0)
label = train_data.classes[train_data[1][1]]
# plt.imshow(img); plt.title(f"{train_data.classes[train_data[1][1]]}")
# plt.show()

# declare the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device Used: {device}")

# preparation of dataloader
batch_size = 128
train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# EfficientNet_V2 model
model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
# print(model)

### fine tuning 'model' for CIFAR - 10 dataset
for param in model.features.parameters():
    param.requires_grad = False

# lets create a classifier for out dataset (lets modify the models classifier)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
out_shape = len(train_data.classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features=1280, out_features=out_shape, bias=True)
)
model.to(device)

summary = torchinfo.summary(model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

train_model = False  
### training the model
# loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)          # using Adam optimizer here

# epochs
epochs = 5
# train function
def train_function():
    # train loop
    for epoch in tqdm(range(epochs)):
        print(f"------ Epoch {epoch} ------")
        model.train()
        total_train_loss = 0                                            # of current batch
        total_train_acc = 0
        total_test_acc = 0
        total_test_loss = 0
        for batch, (X, y) in tqdm(enumerate(train_dataloader)):
            # pred_logits = model(X)                                    # do not require this -> model itsel does these
            # y_pred = torch.softmax(pred_logits,dim=1).argmax(dim=1)
            X, y = X.to(device), y.to(device)
            # forward pass
            y_pred = model(X)
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
            # compute loss
            loss = loss_fn(y_pred, y)
            total_train_loss += loss
            # compute accuacy
            total_train_acc += (y_pred_class == y).sum().item() / len(y_pred)
            # optimizer
            optimizer.zero_grad()   
            # backprop
            loss.backward()
            # update parameters
            optimizer.step()
        
        total_train_loss = total_train_loss / len(train_dataloader)
        total_train_acc = total_train_acc / len(train_dataloader)

        # testing model
        model.eval() 
        if torch.inference_mode():                       
            for batch, (X, y) in tqdm(enumerate(test_dataloader)):
                # put data into the device
                X, y = X.to(device), y.to(device)
                # forward
                y_pred_test = model(X)
                y_pred_class = torch.softmax(y_pred_test, dim=1).argmax(dim=1)
                # compute loss and accuracy
                loss = loss_fn(y_pred_test, y)
                total_test_loss += loss
                total_test_acc += (y_pred_class == y).sum().item() / len(y_pred_test)
        
        total_test_acc = total_test_acc / len(test_dataloader)
        total_test_loss = total_test_loss / len(test_dataloader)

        print(f"Train loss: {total_train_loss:.2} | Train Accuracy {total_train_acc:.2}")
        print(f"Test loss: {total_test_loss:.2} | Test Accuracy {total_test_acc:.2}")


if (train_model):
    train_function()

# save check points
torch.save(model.state_dict(), r"D:\Effi-Net-Classi\checkpoints\model1_18_02_25.pt")

# trying inference (just testing)
test_img = (img.unsqueeze(dim=1)).permute(1,0,2,3)
model.eval()
if torch.inference_mode():
    pred = model(test_img.to(device))
    pred_class = torch.softmax(pred,dim=1).argmax(dim=1)
plt.imshow(img.permute(1,2,0)); plt.title(f"{test_data.classes[pred_class.cpu().item()]}")
plt.show()
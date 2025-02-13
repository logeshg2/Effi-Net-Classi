# this script is used to download CIFAR-10 dataset from torchvision.datasets package
from torchvision import datasets
from torchvision.transforms import ToTensor

# train data
datasets.CIFAR10(r"D:\CV-Datasets", 
                    train=True,
                    transform=ToTensor(),
                    download=False,
                    target_transform=None 
                )
# test data
datasets.CIFAR10(r"D:\CV-Datasets", 
                    train=False,
                    transform=ToTensor(),
                    download=False,
                    target_transform=None 
                )
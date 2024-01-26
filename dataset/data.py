from torch.utils.data import DataLoader
from torchvision import datasets
import torch, ssl
from config import CONFIG

BATCH_SIZE, IMG_SIZE = CONFIG["batch_size"], CONFIG["img_size"]

def data_loader():
    ssl._create_default_https_context = ssl._create_unverified_context
    trainset = datasets.CIFAR10(root='./data', train=True,
                                          download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False,
                                         download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader
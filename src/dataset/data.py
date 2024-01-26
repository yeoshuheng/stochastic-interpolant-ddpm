from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch, ssl
from src.config.config import CONFIG
import matplotlib.pyplot as plt
from src.DDPM import sample

BATCH_SIZE, IMG_SIZE, T = CONFIG["batch_size"], CONFIG["img_size"], CONFIG["img_size"]

def data_loader():
    transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
    ssl._create_default_https_context = ssl._create_unverified_context
    trainset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.ToPILImage(),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def sample_generated_image(model):
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size))
    plt.figure(figsize=(50,50))
    plt.axis("off")
    num_images = 10
    step_size = int(T / num_images)
    for i in range(0, T)[::-1]:
        t = torch.full((1, ), i, dtype=torch.long)
        img = sample(img, t, model)
        img = torch.clamp(img, -1.0, 1.0)
        if i % step_size == 0:
            plt.subplot(1, num_images, int(i / step_size) + 1)
            show_tensor_image()
    plt.show()
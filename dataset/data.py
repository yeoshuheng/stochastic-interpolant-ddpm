from torchvision import transforms
from torch.utils.data import DataLoader
from config import CONFIG

BATCH_SIZE, IMG_SIZE = CONFIG["batch_size"], CONFIG["img_size"]


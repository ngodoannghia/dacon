import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image

class SMILESDataset(Dataset):
    def __init__(self, dataframe, img_size=(224, 224), mode='train'):
        self.data = dataframe
        self.img_size = img_size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.Path[idx]
        
        img = Image.open(path)
        img = self.transform(img)

        if self.mode == 'test':
            return img
        else:
            target = self.data.pIC50[idx]
            return img, torch.tensor(target, dtype=torch.float32)
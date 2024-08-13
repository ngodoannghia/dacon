from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.utils import ic50_to_pic50

class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, img_size=(224, 224), mode='train'):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.img_size = img_size
        self.mode = mode
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = self.data.Smiles[index]
        path = self.data.Path[index]
        
        inputs = self.tokenizer.encode_plus(
            smiles,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )
        
        img = Image.open(path)
        img = self.transform(img)

        if self.mode == 'test':
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'image': img
            }
        else:
            targets = self.data.IC50_nM[index]
            targets = ic50_to_pic50(targets)
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'image': img,
                'targets': torch.tensor(targets, dtype=torch.float32)
            }
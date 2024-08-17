import torch
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128, mode='train'):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.Smiles[idx]
        encoding = self.tokenizer(smiles, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        if self.mode == 'test':
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            target = self.data.pIC50[idx]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'targets': torch.tensor(target, dtype=torch.float)
            }

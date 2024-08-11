from torch.utils.data import Dataset, DataLoader
import torch

class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, mode='train'):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = self.data.Smiles[index]
        
        inputs = self.tokenizer.encode_plus(
            smiles,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )

        if self.mode == 'test':
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            }
        else:
            targets = self.data.pIC50[index]
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.float),
            }
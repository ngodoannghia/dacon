from torch.utils.data import Dataset, DataLoader
import torch
from src.utils import ic50_to_pic50, sigmoid, create_feature

class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, scaler, max_len, mode='train'):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.mode = mode
        self.scaler = scaler

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

        feature = self.scaler.transform([create_feature(smiles)])[0]
        # feature = create_feature(smiles)

        if self.mode == 'test':
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'feature': torch.tensor(feature, dtype=torch.float)
            }
        else:
            # targets = self.data.pIC50[index]
            targets = self.data.IC50_nM[index]
            targets = ic50_to_pic50(targets)
            return {
                'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'feature': torch.tensor(feature, dtype=torch.float),
                'targets': torch.tensor(targets, dtype=torch.float),
            }
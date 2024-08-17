from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
from src.utils import *
import torch

class RMSELoss(nn.Module):     
    def __init__(self, eps=1e-6):         
        super().__init__()         
        self.mse = nn.MSELoss()         
        self.eps = eps  

    def forward(self,yhat,y):         
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)         
        return loss


class RobertaForRegression(nn.Module):
    def __init__(self, model_path):
        super(RobertaForRegression, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.dropout = nn.Dropout(p=0.2)

        self.downstream = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.criterion_pic50 = nn.L1Loss()
        self.criterion_ic50 = RMSELoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.downstream(pooled_output)
        
        return output
    
    def loss(self, output, target):
        ic50_pred = pic50_to_ic50(output)
        ic50_true = pic50_to_ic50(target)
    
        loss_pic50 = self.criterion_pic50(output, target)
        loss_ic50 = self.criterion_ic50(ic50_pred, ic50_true)

        loss = 0.8 * loss_pic50 + 0.2 * loss_ic50
        
        return loss
    
    def score(self, output, target):
        ic50_pred = pic50_to_ic50(output)
        ic50_true = pic50_to_ic50(target)
        
        norm_rmse = normalized_rmse(ic50_true, ic50_pred)
        correct_ratio_value = correct_ratio(target, output)
        
        score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * correct_ratio_value
        
        return score
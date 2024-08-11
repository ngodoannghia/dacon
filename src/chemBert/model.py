import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from src.utils import *

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, outputs):
        weights = torch.softmax(self.attention(outputs), dim=1)
        context_vector = torch.sum(weights * outputs, dim=1)
        return context_vector

class IC50_Prediction_Model(nn.Module):
    def __init__(self, model_path, num_feature):
        super(IC50_Prediction_Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chemBert = AutoModel.from_pretrained(model_path)
        
        self.downstream = nn.Sequential(
            AttentionPooling(num_feature),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(num_feature, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        )
        
        self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask):
        feat_bert = self.chemBert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.downstream(feat_bert[0])
        
        return output
    
    def loss(self, output, target):
        loss = self.criterion(output, target)
        
        return loss
    
    def score(self, output, target):
        ic50_pred = pic50_to_ic50(output)
        ic50_true = pic50_to_ic50(target)
        
        norm_rmse = normalized_rmse(ic50_true, ic50_pred)
        correct_ratio_value = correct_ratio(target, output)
        
        score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * correct_ratio_value
        
        return score
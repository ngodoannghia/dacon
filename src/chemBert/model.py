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

class RMSELoss(nn.Module):     
    def __init__(self, eps=1e-6):         
        super().__init__()         
        self.mse = nn.MSELoss()         
        self.eps = eps  

    def forward(self,yhat,y):         
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)         
        return loss

class IC50_Prediction_Model(nn.Module):
    # def __init__(self, model_path, num_feature, add_feature=True):
    #     super(IC50_Prediction_Model, self).__init__()
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     self.chemBert = AutoModel.from_pretrained(model_path)
    #     self.add_feature = add_feature

    #     for param in self.chemBert.parameters():
    #         param.requires_grad = True

    #     self.attention = AttentionPooling(num_feature)

    #     self.linear_1 = nn.Linear(num_feature, 128)

    #     if self.add_feature:
    #         self.linear_2 = nn.Linear(128 + 13, 1)
    #     else:
    #         self.linear_2 = nn.Linear(128, 1)

    #     self.batchnorm_1 = nn.BatchNorm1d(128)

    #     self.dropout = nn.Dropout(0.3)

    #     self.criterion_pic50 = nn.L1Loss()
    #     self.criterion_ic50 = RMSELoss()

    # def forward(self, input_ids, attention_mask, features):
    #     feat_bert = self.chemBert(input_ids=input_ids, attention_mask=attention_mask)
    #     output = self.attention(feat_bert[0])
    #     output = self.linear_1(output)
    #     output = self.batchnorm_1(output)
    #     output = self.dropout(output)
    #     if self.add_feature:
    #         output = torch.cat((output, features), 1)
        
    #     output = self.linear_2(output)
        
    #     return output
    def __init__(self, model_path, num_feature, add_feature=True):
        super(IC50_Prediction_Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chemBert = AutoModel.from_pretrained(model_path)
        
        self.downstream = nn.Sequential(
            AttentionPooling(num_feature),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(num_feature, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 1)
        )
        
        self.criterion_pic50 = nn.MSELoss()
        self.criterion_ic50 = RMSELoss()

    def forward(self, input_ids, attention_mask, features):
        feat_bert = self.chemBert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.downstream(feat_bert[0])
        
        return output
    
    def extract_feature(self, input_ids, attention_mask):
        feat_bert = self.chemBert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.attention(feat_bert[0])

        return output
    
    def loss(self, output, target):
        ic50_pred = pic50_to_ic50(output)
        ic50_true = pic50_to_ic50(target)
    
        loss_pic50 = self.criterion_pic50(output, target)
        loss_ic50 = self.criterion_ic50(ic50_pred, ic50_true)

        loss = 1 * loss_pic50 + 0 * loss_ic50
        
        return loss
    
    def score(self, output, target):
        ic50_pred = pic50_to_ic50(output)
        ic50_true = pic50_to_ic50(target)
        
        norm_rmse = normalized_rmse(ic50_true, ic50_pred)
        correct_ratio_value = correct_ratio(target, output)
        
        score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * correct_ratio_value
        
        return score

    def score_ic50(self, output, target):
        ic50_pred = output
        ic50_true = target

        pic50_pred = ic50_to_pic50(ic50_pred)
        pic50_true = ic50_to_pic50(ic50_true)
        
        norm_rmse = normalized_rmse(ic50_true, ic50_pred)
        correct_ratio_value = correct_ratio(pic50_true, pic50_pred)
        
        score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * correct_ratio_value
        
        return score
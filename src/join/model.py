import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from efficientnet_pytorch import EfficientNet
from src.utils import *


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, outputs):
        weights = torch.softmax(self.attention(outputs), dim=1)
        context_vector = torch.sum(weights * outputs, dim=1)
        return context_vector


class ChemBertIC50(nn.Module):
    def __init__(self, model_path, num_feature):
        super(ChemBertIC50, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chemBert = AutoModel.from_pretrained(model_path)
        
        self.downstream = nn.Sequential(
            AttentionPooling(num_feature),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Flatten(),
            nn.Linear(num_feature, 128),
            nn.ReLU(),
            nn.Dropout(0.01)
        )

    def forward(self, input_ids, attention_mask):
        feat_bert = self.chemBert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.downstream(feat_bert[0])
        
        return output

class EfficientNetIC50(nn.Module):
    def __init__(self):
        super(EfficientNetIC50, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7', weights_path="Efficientnet/efficientnet-b7-dcc49843.pth")
                
        self.downstream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.efficientnet._fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.01)
        )
        
    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = self.downstream(x)
        return x

class Join_ChemBert_EfficientNet(nn.Module):
    def __init__(self):
        super(Join_ChemBert_EfficientNet, self).__init__()
        
        self.chembert = ChemBertIC50(model_path="ChemBERTa-77M-MTR", num_feature=384)
        self.efficientnet = EfficientNetIC50()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.01)
        self.linear = nn.Linear(256, 1)
        
        self.criterion = nn.MSELoss()
        
    def forward(self, input_ids, attention_mask, img):
        feat_chem = self.chembert(input_ids, attention_mask)
        feat_efficient = self.efficientnet(img)
        
        feat_all = torch.cat((feat_chem, feat_efficient), -1)       
        
        output = self.relu(feat_all)
        output = self.dropout(output)
        
        output = self.linear(output)
        
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
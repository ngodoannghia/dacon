import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from src.utils import *

class EfficientNetForIC50(nn.Module):
    def __init__(self):
        super(EfficientNetForIC50, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7', weights_path="Efficientnet/efficientnet-b7-dcc49843.pth")
                
        self.downstream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.efficientnet._fc.in_features, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        )
        
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = self.downstream(x)
        return x

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
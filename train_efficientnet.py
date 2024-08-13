import torch
import torch.nn as nn
from src.efficientNet.model import EfficientNetForIC50
from src.efficientNet.dataset import SMILESDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

def train():
    # Paramater
    epochs = 2
    lr = 1e-4
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init mode
    model = EfficientNetForIC50()
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Init dataset    
    df_train = pd.read_csv("data/train_10_image.csv")
    df_test = pd.read_csv("data/test_image.csv")
    
    X_train, X_val = train_test_split(df_train, test_size=0.1, random_state=42)
    
    X_train.reset_index(inplace=True)
    X_val.reset_index(inplace=True)

    train_dataset = SMILESDataset(X_train)
    val_dataset = SMILESDataset(X_val)
    test_dataset = SMILESDataset(df_test, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    max_score = -1
    for epoch in range(epochs):
        model.train()
        
        for imgs, targets in tqdm(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = model.loss(outputs.squeeze(), targets)
            loss.backward()
            
            optimizer.step()
        
        model.eval()
        total_loss = 0
        true_values = []
        pred_values = []
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                
                outputs = model(imgs)
                
                loss = model.loss(outputs.squeeze(), targets)
                total_loss += loss.item()
                
                true_values.extend(targets.detach().cpu().numpy())
                pred_values.extend(outputs.squeeze().detach().cpu().numpy())
            
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        print(true_values)
        print(pred_values)
        
        score = model.score(pred_values, true_values)
        
        if max_score < score:
            torch.save(model.state_dict(), f"models/images/best_epoch.pth")
            max_score = score
    
        print("Total loss: ", total_loss)
        print("Score: ", score)
        print("Max score: ", max_score)
    
    model.load_state_dict(torch.load('models/images/best_epoch.pth'))
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            
            outputs = model(imgs)
            predictions.extend(outputs.detach().cpu().numpy()) 
            
    
    result_test = []
    predictions = np.array(predictions)
    for id, pred in zip(df_test['ID'].values, predictions):
        result_test.append([id, 10**(-pred[0])*1e9]) 
    
    submission = pd.DataFrame(columns=['ID', 'IC50_nM'], data=result_test)
    
    submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    train()
    
import torch
import torch.nn as nn
from src.join.dataset import SMILESDataset
from src.join.model import Join_ChemBert_EfficientNet
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def read_data(path):
    train = pd.read_csv(path)
    
    return train  

def train():
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    # Paramater
    epochs = 20
    path_train = "data/train_10_image.csv"
    path_test = "data/test.csv"
    
    # Init model
    model = Join_ChemBert_EfficientNet()
    model.to(device)
    
    # Read dataset
    df_train = read_data(path_train)
    df_test = read_data(path_test)
    
    X_train, X_val = train_test_split(df_train, test_size=0.1, random_state=42)
    
    X_train.reset_index(inplace=True)
    X_val.reset_index(inplace=True)
    
    train_dataset = SMILESDataset(X_train, model.chembert.tokenizer, max_len=128)
    val_dataset = SMILESDataset(X_val, model.chembert.tokenizer, max_len=128)
    test_dataset = SMILESDataset(df_test, model.chembert.tokenizer, max_len=128, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    
    # Training
    max_score = -1
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img = batch['image'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, img=img)
            loss = model.criterion(outputs.squeeze(), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        total_loss = 0
        true_values = []
        pred_values = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                img = batch['image'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, img=img)
                loss = model.criterion(outputs.squeeze(), targets)

                total_loss += loss.item()

                true_values.extend(targets.detach().cpu().numpy())
                pred_values.extend(outputs.squeeze().detach().cpu().numpy())
        
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        print("True value: ", true_values)
        print("Predict value: ", pred_values)
        
        score = model.score(pred_values, true_values)
        
        if max_score < score:
            torch.save(model.state_dict(), f"models/best_epoch.pth")
            max_score = score
    
        print("Total loss: ", total_loss)
        print("Score: ", score)
        print("Max score: ", max_score)
    
    
    model.load_state_dict(torch.load('models/best_epoch.pth'))
    model.eval()
    predictions = []
    
    with torch.no_grad():  # Tắt gradient trong quá trình dự đoán
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img = batch['image'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, img=img)
            predictions.extend(outputs.detach().cpu().numpy())
    
    result_test = []
    predictions = np.array(predictions)
    for id, pred in zip(df_test['ID'].values, predictions):
        result_test.append([id, 10**(-pred[0])*1e9]) 
    
    submission = pd.DataFrame(columns=['ID', 'IC50_nM'], data=result_test)
    
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    train()
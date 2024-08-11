import torch
import torch.nn as nn
from src.chemBert.dataset import SMILESDataset
from src.chemBert.model import IC50_Prediction_Model
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def read_data_train(path):
    train = pd.read_csv(path)
    data_train = pd.concat([train['Smiles'], train['IC50_nM'], train['pIC50']], axis=1)
    
    return data_train

def read_data_test(path):
    test = pd.read_csv(path)
    
    return test   

def train():
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    # Paramater
    epochs = 100
    path_model = "E://competition//atoms//ChemBERTa-77M-MTR"
    path_train = "E://competition//atoms//data//train_all.csv"
    path_test = "E://competition//atoms//data//test.csv"
    
    # Init model
    model = IC50_Prediction_Model(model_path=path_model,
                                  num_feature=384)
    model.to(device)
    
    # Read dataset
    df_train = read_data_train(path_train)
    df_test = read_data_test(path_test)
    
    X_train, X_val = train_test_split(df_train, test_size=0.1, random_state=42)
    
    X_train.reset_index(inplace=True)
    X_val.reset_index(inplace=True)
    
    train_dataset = SMILESDataset(X_train, model.tokenizer, max_len=128)
    val_dataset = SMILESDataset(X_val, model.tokenizer, max_len=128)
    test_dataset = SMILESDataset(df_test, model.tokenizer, max_len=128, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training
    max_score = -1
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
                targets = batch['targets'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = model.criterion(outputs.squeeze(), targets)

                total_loss += loss.item()

                true_values.extend(targets.detach().cpu().numpy())
                pred_values.extend(outputs.squeeze().detach().cpu().numpy())
        
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        score = model.score(pred_values, true_values)
        
        if max_score < score:
            torch.save(model.state_dict(), f"E://competition//atoms//models//best_epoch.pth")
            max_score = score
    
        print("Total loss: ", total_loss)
        print("Score: ", score)
        print("Max score: ", max_score)
    
    
    model.load_state_dict(torch.load('E://competition//atoms//models//best_epoch.pth'))
    model.eval()
    predictions = []
    
    with torch.no_grad():  # Tắt gradient trong quá trình dự đoán
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.detach().cpu().numpy())
    
    result_test = []
    predictions = np.array(predictions)
    for id, pred in zip(df_test['ID'].values, predictions):
        result_test.append([id, 10**(-pred[0])*1e9]) 
    
    submission = pd.DataFrame(columns=['ID', 'IC50_nM'], data=result_test)
    
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    train()
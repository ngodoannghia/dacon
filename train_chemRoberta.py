import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from src.chemRoberta.dataset import SMILESDataset
from src.chemRoberta.model import RobertaForRegression

def read_data(path):
    data = pd.read_csv(path)
    
    return data

def lr_lambda(epoch):
    if epoch == 1:
        return 0.1
    elif epoch == 2:
        return 0.25
    elif epoch == 3:
        return 0.5
    elif epoch ==  4:
        return 0.75
    return 1.0

def train():
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    # Paramater
    epochs = 100
    path_model = "roberta-large"
    path_train = "data/balance/train_10.csv"
    path_val = "data/balance/val_10.csv"
    path_test = "data/test.csv"
    
    # Init model
    model = RobertaForRegression(model_path=path_model)
    model.to(device)
    
    # Read dataset
    df_train = read_data(path_train)
    df_val = read_data(path_val)
    df_test = read_data(path_test)
    
    # X_train, X_val = train_test_split(df_train, test_size=0.1, random_state=42)
    # X_train.reset_index(inplace=True, drop=True)
    # X_val.reset_index(inplace=True, drop=True)

    train_dataset = SMILESDataset(df_train, model.tokenizer, max_len=128)
    val_dataset = SMILESDataset(df_val, model.tokenizer, max_len=128)
    test_dataset = SMILESDataset(df_test, model.tokenizer, max_len=128, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Training
    max_score = -1
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = model.loss(outputs.squeeze(), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        # scheduler.step()
        for param_group in optimizer.param_groups:
            print(f"Learning rate: {param_group['lr']}")
        
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
                loss = model.loss(outputs.squeeze(), targets)

                total_loss += loss.item()

                true_values.extend(targets.detach().cpu().numpy())
                pred_values.extend(outputs.squeeze().detach().cpu().numpy())
        
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        score = model.score(pred_values, true_values)
        
        if max_score < score:
            torch.save(model.state_dict(), f"models/smiles/best_model_chemRoberta.pth")
            max_score = score
    
        print("Total loss: ", total_loss)
        print("Score: ", score)
        print("Max score: ", max_score)
    
    
    model.load_state_dict(torch.load('models/smiles/best_model_chemRoberta.pth'))
    model.eval()
    predictions = []

    total_loss = 0
    true_values = []
    pred_values = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = model.loss(outputs.squeeze(), targets)

            total_loss += loss.item()

            true_values.extend(targets.detach().cpu().numpy())
            pred_values.extend(outputs.squeeze().detach().cpu().numpy())
    
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)
    
    score = model.score(pred_values, true_values)

    print("Total loss best model: ", total_loss)
    print("Score best model: ", score)
    
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
        # result_test.append([id, pred * 1e6])
    
    submission = pd.DataFrame(columns=['ID', 'IC50_nM'], data=result_test)
    
    submission.to_csv("submission_roberta.csv", index=False)

if __name__ == '__main__':
    train()
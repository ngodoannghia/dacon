import torch
import torch.nn as nn
from src.join.dataset import SMILESDataset
from src.join.model import Join_ChemBert_EfficientNet
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.utils import create_feature
from sklearn.preprocessing import StandardScaler, RobustScaler

def read_data(path):
    train = pd.read_csv(path)
    
    return train  

def train():
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    # Paramater
    epochs = 50
    path_train = "data/data_split_val/train_image.csv"
    path_val = "data/data_split_val/val_image.csv"
    path_test = "data/test_image.csv"
    
    # Init model
    model = Join_ChemBert_EfficientNet()
    model.to(device)
    
    # Read dataset
    df_train = read_data(path_train)
    df_val = read_data(path_val)
    df_test = read_data(path_test)
    
    list_feature = []
    for s in tqdm(df_train.Smiles):
        list_feature.append(create_feature(s))
    
    scaler = RobustScaler()
    scaler.fit(list_feature)
    
    train_dataset = SMILESDataset(df_train, model.chembert.tokenizer, scaler=scaler, max_len=128)
    val_dataset = SMILESDataset(df_val, model.chembert.tokenizer, scaler=scaler, max_len=128)
    test_dataset = SMILESDataset(df_test, model.chembert.tokenizer, scaler=scaler, max_len=128, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
    
    # Training
    max_score = -1
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img = batch['image'].to(device)
            targets = batch['targets'].to(device)
            features = batch['feature'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, img=img, feature=features)
            loss = model.criterion(outputs.reshape(-1), targets)
            
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
                features = batch['feature'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, img=img, feature=features)

                loss = model.criterion(outputs.reshape(-1), targets)

                assert outputs.reshape(-1).shape == targets.shape

                total_loss += loss.item()

                true_values += targets.detach().cpu().tolist()
                pred_values += outputs.reshape(-1).detach().cpu().tolist()
        

        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        

        print("True value: ", true_values)
        print("Predict value: ", pred_values)
        
        score = model.score(pred_values, true_values)
        
        if max_score < score:
            torch.save(model.state_dict(), f"models/join/balance_data/best_epoch_tmp.pth")
            max_score = score
    
        print("Total loss: ", total_loss)
        print("Score: ", score)
        print("Max score: ", max_score)
    
    
    model.load_state_dict(torch.load('models/join/balance_data/best_epoch_tmp.pth'))
    model.eval()
    predictions = []
    
    with torch.no_grad():  # Tắt gradient trong quá trình dự đoán
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img = batch['image'].to(device)
            features = batch['feature'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, img=img, feature=features)
            predictions += outputs.reshape(-1).detach().cpu().tolist()
    
    result_test = []
    predictions = np.array(predictions)
    for id, pred in zip(df_test['ID'].values, predictions):
        result_test.append([id, 10**(-pred)*1e9]) 
    
    submission = pd.DataFrame(columns=['ID', 'IC50_nM'], data=result_test)
    
    submission.to_csv("submission/join/balance_data/submission.csv", index=False)

if __name__ == '__main__':
    train()
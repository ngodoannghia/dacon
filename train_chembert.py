import torch
import torch.nn as nn
from src.chemBert.dataset import SMILESDataset
from src.chemBert.model import IC50_Prediction_Model
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.utils import create_feature

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
    epochs = 50
    path_model = "ChemBERTa-77M-MTR"
    path_train = "data/balance_v2/train.csv"
    path_val = "data/balance_v2/val.csv"
    path_test = "data/test.csv"
    
    # Init model
    model = IC50_Prediction_Model(model_path=path_model,
                                  num_feature=384, add_feature=False)
    model.load_state_dict(torch.load('models/smiles/pretrain_model.pth'))
    model.to(device)
    
    # Read dataset
    df_train = read_data_train(path_train)
    df_val = read_data_train(path_val)
    df_test = read_data_test(path_test)
    
    # X_train, X_val = train_test_split(df_train, test_size=0.1, random_state=42)
    # X_train.reset_index(inplace=True, drop=True)
    # X_val.reset_index(inplace=True, drop=True)

    list_feature = []
    for s in tqdm(df_train.Smiles[:10]):
        list_feature.append(create_feature(s))
    
    scaler = StandardScaler()
    scaler.fit(list_feature)


    train_dataset = SMILESDataset(df_train, model.tokenizer, scaler, max_len=128)
    val_dataset = SMILESDataset(df_val, model.tokenizer, scaler, max_len=128)
    test_dataset = SMILESDataset(df_test, model.tokenizer, scaler, max_len=128, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-10)

    # # Training
    # max_score = -1
    # for epoch in range(epochs):
    #     print("Epoch: ", epoch)
    #     model.train()
    #     for batch in tqdm(train_loader):
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         targets = batch['targets'].to(device)
    #         features = batch['feature'].to(device)

    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
    #         loss = model.loss(outputs.reshape(-1), targets)
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
    #     for param_group in optimizer.param_groups:
    #         print(f"Learning rate: {param_group['lr']}")
        
    #     model.eval()
    #     total_loss = 0
    #     true_values = []
    #     pred_values = []
        
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             input_ids = batch['input_ids'].to(device)
    #             attention_mask = batch['attention_mask'].to(device)
    #             targets = batch['targets'].to(device)
    #             features = batch['feature'].to(device)

    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
    #             loss = model.loss(outputs.reshape(-1), targets)

    #             total_loss += loss.item()

    #             true_values += targets.detach().cpu().tolist()
    #             pred_values += outputs.reshape(-1).detach().cpu().tolist()
        
        
        
    #     true_values = np.array(true_values)
    #     pred_values = np.array(pred_values)

    #     print("True value: ", true_values)
    #     print("Pre value: ", pred_values)
        
    #     score = model.score(pred_values, true_values)
        
    #     if max_score < score:
    #         torch.save(model.state_dict(), f"models/smiles/finetune_model.pth")
    #         max_score = score
    
    #     scheduler.step(total_loss)

    #     print("Total loss: ", total_loss)
    #     print("Score: ", score)
    #     print("Max score: ", max_score)
    
    
    model.load_state_dict(torch.load('models/smiles/pretrain_model.pth'))
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
            features = batch['feature'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = model.loss(outputs.squeeze(), targets)

            total_loss += loss.item()

            true_values += targets.detach().cpu().tolist()
            pred_values += outputs.reshape(-1).detach().cpu().tolist()
    
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)
    
    score = model.score(pred_values, true_values)

    print("Total loss best model: ", total_loss)
    print("Score best model: ", score)
    
    with torch.no_grad():  # Tắt gradient trong quá trình dự đoán
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['feature'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            predictions += outputs.detach().cpu().tolist()
   
    result_test = []
    predictions = np.array(predictions)
    for id, pred in zip(df_test['ID'].values, predictions):
        result_test.append([id, 10**(-pred[0])*1e9])
        # result_test.append([id, pred * 1e6])
    
    submission = pd.DataFrame(columns=['ID', 'IC50_nM'], data=result_test)
    
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    train()
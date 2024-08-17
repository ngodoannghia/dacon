from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdqueries
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcPMI1, CalcPMI2, CalcPMI3
from rdkit.Chem.rdMolDescriptors import CalcInertialShapeFactor
from rdkit.Chem.rdMolDescriptors import CalcPBF
from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
from PIL import Image

from xgboost import XGBRegressor

from src.chemBert.model import IC50_Prediction_Model
from src.efficientNet.model import EfficientNetForIC50

import torchvision.transforms as transforms


import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_chemBert = IC50_Prediction_Model(model_path="ChemBERTa-77M-MTR", num_feature=384)
model_chemBert.load_state_dict(torch.load('models/smiles/best_model_gen_500_val_100.pth'))
model_chemBert.to("cuda")
model_chemBert.eval()
# model_chem_bert = AutoModelForMaskedLM.from_pretrained("ChemBERTa-zinc-base-v1")

model_efficientnet = EfficientNetForIC50()
model_efficientnet.load_state_dict(torch.load('models/images/best_model_gen_500_val_100.pth'))
model_efficientnet.to("cuda")
model_efficientnet.eval()

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_feature(sample, path):
    mol = Chem.MolFromSmiles(sample)

    if mol:
        pass
    else:
        print("Faile")
        return None    
    # tổng khối lượng của tất cả các nguyên tử trong phân tử.
    mol_weight = Descriptors.MolWt(mol)
    # print("Mol weight: ", mol_weight)

    # Tính toán hệ số phân bố dầu-nước của phân tử, đại diện cho tính ưa dầu hoặc ưa nước
    logp = Descriptors.MolLogP(mol)
    # print("Log P: ", logp)

    # số liên kết đơn không thuộc vòng trong phân tử, biểu thị cho tính linh hoạt của phân tử.
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    # print("Num rotatable bonds: ", num_rotatable_bonds)

    #  diện tích bề mặt phân tử có phân cực, liên quan đến khả năng tương tác với môi trường nước
    tpsa = Descriptors.TPSA(mol)
    # print("Tpsa: ", tpsa)

    # số lượng nhóm chức trong phân tử có khả năng cho hoặc nhận liên kết hydro
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    # print("Num h donors: ", num_h_donors)
    # print("Num h acceptor: ", num_h_acceptors)

    # Đo lường độ phân cực của các liên kết trong phân tử
    mol_refractivity = Descriptors.MolMR(mol)
    # print("Mol refractivity: ", mol_refractivity)

    # số vòng trong phân tử
    num_rings = Descriptors.RingCount(mol)
    # print("Num rings: ", num_rings)

    # Tính tỷ lệ các nguyên tử carbon trong phân tử có cấu trúc lai Csp3
    fraction_csp3 = Descriptors.FractionCSP3(mol)
    # print("Fraction csp3: ", fraction_csp3)

    # Tính số lượng vòng thơm trong phân tử.
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    # print("Num aromatic rings: ", num_aromatic_rings)

    # Đánh giá khả năng của một phân tử có thể trở thành một loại thuốc dựa trên các đặc trưng hóa học.
    qed_value = QED.qed(mol)
    # print("Qed value: ", qed_value)

    # Đánh giá độ phức tạp của phân tử, tức là mức độ khó khăn khi tổng hợp phân tử trong phòng thí nghiệm.
    # sa_score = rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
    # print("Sa score: ", sa_score)

    # Đo lường độ kết nối của phân tử dựa trên cấu trúc đồ thị của nó.
    balaban_j = Descriptors.BalabanJ(mol)
    # print("Balaban j: ", balaban_j)

    # Các chỉ số chi khác nhau (Chi0, Chi1, Chi2,...) đo lường các khía cạnh khác nhau của sự kết nối trong phân tử.
    chi0 = Descriptors.Chi0(mol)
    # print("Chi0: ", chi0)

    # 3d vitualize
    # AllChem.EmbedMolecule(mol)
    # AllChem.UFFOptimizeMolecule(mol)

    # # Tính toán các mômen quán tính chính
    # pmi1 = CalcPMI1(mol)  # PMI theo trục 1
    # pmi2 = CalcPMI2(mol)  # PMI theo trục 2
    # pmi3 = CalcPMI3(mol)  # PMI theo trục 3
    # # print(f"pmi1: {pmi1}\npmi2: {pmi2}\npmi3: {pmi3}")

    # # Tính toán chỉ số hình dạng
    # inertial_shape_factor = CalcInertialShapeFactor(mol)
    # # print("inertial shape factor", inertial_shape_factor)

    # # Tính toán tỷ lệ mômen quán tính
    # pmi_ratio = CalcPBF(mol)  # Principal moments balance factor (PBF)
    # # print("Pmi ratio: ", pmi_ratio)

    # # Tính bán kính quán tính
    # radius_of_gyration = CalcRadiusOfGyration(mol)
    # print("Radius of gyration: ", radius_of_gyration)


    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024).ToList()

    features_chem = create_feature_chembert(sample).tolist()

    features_efficient = create_feature_efficentnet(path).tolist()

    print(f"Smile: {sample}: {len(features_chem)}")
    
    # return [inertial_shape_factor, fraction_csp3, qed_value, balaban_j], [mol_weight, logp, tpsa, mol_refractivity, chi0, pmi1, pmi2, pmi3, pmi_ratio, radius_of_gyration, num_aromatic_rings, num_h_donors, num_h_acceptors, num_rings, num_rotatable_bonds]
    # return [inertial_shape_factor, fraction_csp3, qed_value, balaban_j, mol_weight, logp, tpsa, mol_refractivity, chi0, pmi1, pmi2, pmi3, pmi_ratio, radius_of_gyration, num_aromatic_rings, num_h_donors, num_h_acceptors, num_rings, num_rotatable_bonds] + features_chem
    # return [fraction_csp3, qed_value, balaban_j, mol_weight, logp, tpsa, mol_refractivity, chi0, num_aromatic_rings, num_h_donors, num_h_acceptors, num_rings, num_rotatable_bonds]
    return [fraction_csp3, qed_value, balaban_j, mol_weight, logp, tpsa, mol_refractivity, chi0, num_aromatic_rings, num_h_donors, num_h_acceptors, num_rings, num_rotatable_bonds] + fingerprint + features_chem + features_efficient 

def create_feature_chembert(sample):
    inputs = model_chemBert.tokenizer.encode_plus(
            sample,
            None,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )
    inputs_data = {
        'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to('cuda'),
        'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to('cuda'),
    }
    with torch.no_grad():
        features = model_chemBert.extract_feature(inputs_data['input_ids'], inputs_data['attention_mask']).squeeze()
    
    return features.cpu()

def create_feature_efficentnet(path):
    img = Image.open(path)
    img = transform(img).unsqueeze(0).to('cuda')

    with torch.no_grad():
        features = model_efficientnet.extract_feature(img).squeeze()

    return features.cpu()

def read_data(path):
    data = pd.read_csv(path)
    
    return data


def create_data_train(data_train):
    
    feature = []
    lables = []
    
    for i in tqdm(range(len(data_train))): 
        row = data_train.iloc[i].values
        path, smile, lb = row[0], row[1], row[3]
        
        feat_1 = create_feature(smile, path)
    
        
        feature.append(feat_1)

        lables.append(lb)
        
    
    return feature, lables

def create_data_test(data_test):
    
    feature = []
    
    for i in range(len(data_test)): 
        row = data_test.iloc[i].values
        path, smile = row[0], row[2]
        
        feat_1 = create_feature(smile, path)
        
        feature.append(feat_1)
    
    return feature
    
    
def create_model():
    # Các mô hình thành phần
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('svr', SVR()),
        ('nn', MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=20000)),
        ('xgb1', XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)),
    ]
    # Mô hình meta
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

    return stacking_model

def create_model_xgboots():
    base_models = [
        ('xgb1', XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)),
        ('xgb2', XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)),
        ('xgb3', XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.01))
    ]
    
    # Define the final estimator (e.g., Linear Regression)
    final_estimator = LinearRegression()

    # Create the stacking regressor
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=5  # Use cross-validation to train the stacking model
    )
    
    return stacking_model

def create_df_feat(features):
    columns = []
    for i in range(len(features[0])):
        columns.append(f"columns_{i}")
    
    df = pd.DataFrame(features, columns=columns)
    
    return df

if __name__ == '__main__':
    path_train = "data/train_10_image.csv"
    path_val = "data/balance/val_image_500.csv"
    path_test = "data/test_image.csv"
    
    data_train = read_data(path_train)
    data_val = read_data(path_val)
    data_test = read_data(path_test)

    data_train_all = pd.concat([data_train, data_val])
    
    model = create_model()
    # model = create_model_xgboots()
    # model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1)
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    # scaler = StandardScaler()
    scaler = RobustScaler()
    # scaler = MinMaxScaler()
    
    features, labels = create_data_train(data_train_all)
    
    df = create_df_feat(features)
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    X_train, X_valid, y_train, y_valid = train_test_split(df_scaled.values, labels, test_size=0.1, random_state=42)
    
    # Huấn luyện mô hình ensemble
    model.fit(X_train, y_train)
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_valid)
    
    # Inference test
    features_test = create_data_test(data_test)
    
    df_test = create_df_feat(features_test)
    
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)
    
    y_pred_test = model.predict(df_test_scaled.values)
    
    result_test = []
    
    for id, pred in zip(data_test['ID'].values, y_pred_test):
        result_test.append([id, 10**(-pred)*1e9]) 
    
    submission = pd.DataFrame(columns=['ID', 'IC50_nM'], data=result_test)
    
    submission.to_csv("ml_submission.csv", index=False)
    
    # Tính toán điểm số của mô hình
    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid, y_pred)
    
    print(f"Result: \nMSE: {mse}\nRMSE: {rmse}\nR2: {r2}")
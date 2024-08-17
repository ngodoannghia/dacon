import pandas as pd

from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcPMI1, CalcPMI2, CalcPMI3
from rdkit.Chem.rdMolDescriptors import CalcInertialShapeFactor
from rdkit.Chem.rdMolDescriptors import CalcPBF
from rdkit.Chem.rdMolDescriptors import CalcRadiusOfGyration

import numpy as np

def read_data_train(path):
    train = pd.read_csv(path)
    data_train = pd.concat([train['Smiles'], train['IC50_nM'], train['pIC50']], axis=1)
    
    return data_train

def read_data_test(path):
    test = pd.read_csv(path)
    
    return test

def create_feature(sample):
    mol = Chem.MolFromSmiles(sample)

    if mol:
        pass
    else:
        print("Faile")
        return None    
    # tổng khối lượng của tất cả các nguyên tử trong phân tử.
    mol_weight = Descriptors.MolWt(mol)

    # Tính toán hệ số phân bố dầu-nước của phân tử, đại diện cho tính ưa dầu hoặc ưa nước
    logp = Descriptors.MolLogP(mol)

    # số liên kết đơn không thuộc vòng trong phân tử, biểu thị cho tính linh hoạt của phân tử.
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    #  diện tích bề mặt phân tử có phân cực, liên quan đến khả năng tương tác với môi trường nước
    tpsa = Descriptors.TPSA(mol)

    # số lượng nhóm chức trong phân tử có khả năng cho hoặc nhận liên kết hydro
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)

    # Đo lường độ phân cực của các liên kết trong phân tử
    mol_refractivity = Descriptors.MolMR(mol)
    
    # số vòng trong phân tử
    num_rings = Descriptors.RingCount(mol)

    # Tính tỷ lệ các nguyên tử carbon trong phân tử có cấu trúc lai Csp3
    fraction_csp3 = Descriptors.FractionCSP3(mol)

    # Tính số lượng vòng thơm trong phân tử.
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)

    # Đánh giá khả năng của một phân tử có thể trở thành một loại thuốc dựa trên các đặc trưng hóa học.
    qed_value = QED.qed(mol)

    # Đo lường độ kết nối của phân tử dựa trên cấu trúc đồ thị của nó.
    balaban_j = Descriptors.BalabanJ(mol)
    
    # Các chỉ số chi khác nhau (Chi0, Chi1, Chi2,...) đo lường các khía cạnh khác nhau của sự kết nối trong phân tử.
    chi0 = Descriptors.Chi0(mol)

    # # 3d vitualize
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
    
    # # Tạo fingerprints từ SMILES
    # fingerprint = Chem.RDKFingerprint(mol, fpSize=128).ToList()
    
    # return [inertial_shape_factor, fraction_csp3, qed_value, balaban_j], [mol_weight, logp, tpsa, mol_refractivity, chi0, pmi1, pmi2, pmi3, pmi_ratio, radius_of_gyration, num_aromatic_rings, num_h_donors, num_h_acceptors, num_rings, num_rotatable_bonds]
    # return [inertial_shape_factor, fraction_csp3, qed_value, balaban_j, mol_weight, logp, tpsa, mol_refractivity, chi0, pmi1, pmi2, pmi3, pmi_ratio, radius_of_gyration, num_aromatic_rings, num_h_donors, num_h_acceptors, num_rings, num_rotatable_bonds]
    return [fraction_csp3, qed_value, balaban_j, mol_weight, logp, tpsa, mol_refractivity, chi0, num_aromatic_rings, num_h_donors, num_h_acceptors, num_rings, num_rotatable_bonds]


def ic50_to_pic50(ic50):
    return -np.log10(ic50*1e-9)

def pic50_to_ic50(pic50):
    return 10 ** (-pic50) * 1e9

def normalized_rmse(ic50_true, ic50_pred):
    rmse = np.sqrt(np.mean((ic50_true - ic50_pred) ** 2))
    norm_rmse = rmse / (np.max(ic50_true) - np.min(ic50_true))
    return norm_rmse

def correct_ratio(pic50_true, pic50_pred):
    absolute_errors = np.abs(pic50_true - pic50_pred)
    correct_count = np.sum(absolute_errors <= 0.5)
    return correct_count / len(pic50_true)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_inverse(y):
    return np.log(y / (1 - y))
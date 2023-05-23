import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import h5py
import os

def depict_pic(pdb_file='c1ccccc1',emb_file=''):
    file=open(pdb_file,'r')
    lines = file.readlines()

    mol = AllChem.MolFromPDBFile(pdb_file,proximityBonding=False)  # 将pdb表示分子转换为mol对象
    atoms=mol.GetAtoms()

    w_file = open(emb_file, 'a')
    for i in range(len(atoms)):
        atom_feat=[]
        a=lines[i].split()
        #print(lines[i])
        #print(a[6])
        if a[0]=='ATOM':
            atom_feat.append(float(lines[i][30:38]))  # x
            atom_feat.append(float(lines[i][38:46]))  #y
            atom_feat.append(float(lines[i][46:54]))  #z
            #atom_feat.append(a[6])#x
            #atom_feat.append(a[7])#y
            #atom_feat.append(a[8])#z
            #atom_feat.append(atoms[i].GetSymbol())#获得原子的元素符号 C
            atom_feat.append(atoms[i].GetAtomicNum())#获得原子对应元素的编号 6
            atom_feat.append(atoms[i].GetFormalCharge())#获得原子的电荷信息 0
            atom_feat.append(1 if atoms[i].GetIsAromatic() else 0 )#判断原子是否是芳香性原子 True 1  False 0
            atom_feat.append(1 if atoms[i].IsInRing() else 0 )#判断原子是否在环上 True 1  False 0
            #atom_feat.append(atoms[i].GetHybridization())#获取原子杂化方式
            hybridization=atoms[i].GetHybridization()
            #print(str(hybridization))
            if str(hybridization)=='SP3':
                atom_feat.append('3')
            elif str(hybridization)=='SP2':
                atom_feat.append('2')
            elif str(hybridization)=='SP1':
                atom_feat.append('1')
            atom_feat.append(atoms[i].GetExplicitValence())  # 获取原子显式化合价
            atom_feat.append(atoms[i].GetImplicitValence())  # 获取原子隐式化合价
            atom_feat.append(atoms[i].GetTotalValence())  # 获取原子总化合价
            #feat.append(atom_feat)
            feat_str = "\t".join('%s' % feat for feat in atom_feat)
            feat_str=feat_str+'\n'

            w_file.write(feat_str)
    print(feat_str)
    print('________________________________')

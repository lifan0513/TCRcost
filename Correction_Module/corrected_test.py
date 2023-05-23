import os
import sys
from numpy import *
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from corrected_model import cmodel
import RMSD
import argparse
torch.set_printoptions(profile="full")


def all_data_read(file_path):
    files = os.listdir(file_path)
    file_xyz=[]
    file_others = []
    files_ca = []
    files_ca1 = []
    for file in files:
        with open (file_path + '/' + file,'r') as f :
            pr_xyz=[]
            pr_others = []
            atom_x = []
            atom_y = []
            atom_z = []

            lines = f.readlines()
            ca = []
            ca1 = []
            i=-1
            for line in lines:
                #a = line.split()
                i=i+1
                a=[line[0:4],line[4:11],line[11:17],line[17:20],line[20:22],line[22:30],line[30:38],line[38:46],line[46:54],line[54:60],line[60:66],line[66:78]]
                # a = ['ATOM', '3068', 'N', 'LEU', 'C', '1', 82.338, 9.558, 9.331, '1.00', '24.78', 'N']
                if a[2] == '  CA  ':
                        ca.append(1)
                        ca1.append(1)
                elif a[2] == '  N   ':
                    ca1.append(1)
                    if a[3] == 'PRO':
                        ca.append(22)
                    else:
                        ca.append(2)
                elif a[2] == '  C   ':
                    ca.append(3)
                    ca1.append(1)
                else:
                    ca.append(0)
                    ca1.append(0)
                atom_x.append(float(a[6]))
                atom_y.append(float(a[7]))
                atom_z.append(float(a[8]))

                atom_others = []
                atom_others.append(a[0])#line[0:4]
                atom_others.append(a[1])#line[4:11]
                atom_others.append(a[2])#line[11:17]
                atom_others.append(a[3])#line[17:20]
                atom_others.append(a[4])#line[20:22]
                atom_others.append(a[5])#line[22:30]
                atom_others.append(a[6])#x   line[30:38]
                atom_others.append(a[7])#y   line[38:46]
                atom_others.append(a[8])#z   line[46:54]
                atom_others.append(a[9])#line[54:60]
                atom_others.append(a[10])#line[60:66]
                atom_others.append(a[11])#line[66:78]
                if a[11][0]!=' ':
                    pass#print(file)

                pr_others.append(atom_others)
            for i in range(400-len(atom_z)):
                #zero_xyz=[0,0,0]
                ca.append(0)
                ca1.append(0)
                zero_others = ['#','#','#','#','#','#','#','#','#','#','#','#']
                atom_x.append(0)
                atom_y.append(0)
                atom_z.append(0)
                pr_others.append(zero_others)

            pr_xyz.append(atom_x)
            pr_xyz.append(atom_y)
            pr_xyz.append(atom_z)
            file_xyz.append(pr_xyz)
            file_others.append(pr_others)
            files_ca.append(ca)
            files_ca1.append(ca1)

    data = torch.tensor(file_xyz)
    data = data.reshape([len(data),3,400])
    others = file_others
    data_ca = torch.tensor(files_ca)
    data_ca1 = torch.tensor(files_ca1)
    #print(files_ca)
    return data,others,files,data_ca,data_ca1


def test(model_path="C:/Users/lifan/Desktop/result/model2/500.pth",
         batch_size=32,
         pdb_path="C:/Users/lifan/Desktop/result/model2/all_files_500",
         test_path="E:/paper_data/5-fold-structure-data/0",
         ref_path = "E:/paper_data/5-fold-structure-data/0"):
    alphafold_data, alphafold_others, alphafold_files,alphafold_data_ca,alphafold_data_ca1 = all_data_read(test_path)#(test_path+'/test_af')#"C:/Users/lifan/Desktop/model/change_alphafold_test"
    true_data, true_others, true_files,true_data_ca,true_data_ca1 = all_data_read(test_path)#(test_path+'/test_true')
    data = TensorDataset(alphafold_data, true_data,alphafold_data_ca,true_data_ca,alphafold_data_ca1,true_data_ca1)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model = cmodel(verbose=0)
    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint.pop("model_state_dict")
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    for batch_ind, batch_data in enumerate(data_loader):
        alphafold_input, true_output, alphafold_ca, true_ca, alphafold_ca1, true_ca1 = batch_data
        alphafold_input = alphafold_input.to(torch.float32)

        batch_ca_main = []
        batch_ca_side = []
        for i in range(len(true_ca1)):
            k1, k2 = 0, 0
            pr_ca_main = np.zeros((150, 400))
            pr_ca_side = np.zeros((400, 400))
            for j in range(len(true_ca1[i])):
                if true_ca1[i][j] == 1:
                    main_ca = np.zeros(400)
                    main_ca[j] = 1
                    pr_ca_main[k1] = main_ca
                    k1 = k1 + 1
                else:
                    side_ca = np.zeros(400)
                    side_ca[j] = 1
                    pr_ca_side[k2] = side_ca
                    k2 = k2 + 1
            batch_ca_main.append(np.array(pr_ca_main).T)
            batch_ca_side.append(np.array(pr_ca_side).T)
        batch_main = torch.tensor(batch_ca_main, dtype=float)#main chain 原子矩阵
        batch_side = torch.tensor(batch_ca_side, dtype=float)#side chain 原子矩阵

        true_main = torch.matmul(true_output.float(), batch_main.float())  # true_output * ca  main chain 真实的原子坐标
        true_side = torch.matmul(true_output.float(), batch_side.float())  # true_output * (1-ca)  side chain 真实的原子坐标
        true_all = true_output

        pre_main, pre_side, pre_all = model(alphafold_input, batch_main, batch_side)

        is_exists = os.path.exists(pdb_path+'/all')
        if not is_exists:
            os.makedirs(pdb_path)
            os.makedirs(pdb_path+'/all')
            os.makedirs(pdb_path+'/main')
            os.makedirs(pdb_path+'/side')
        for i in range(len(true_main)):
            filename=true_files[batch_ind*batch_size+i]
            k1,k2=0,0
            for j in range(400):
                # all
                str_x = str(format(pre_all[i][0][j].item(), '.4f'))
                str_y = str(format(pre_all[i][1][j].item(), '.4f'))
                str_z = str(format(pre_all[i][2][j].item(), '.4f'))
                for x in range(8 - len(str_x)):
                    str_x = ' ' + str_x
                for y in range(8 - len(str_y)):
                    str_y = ' ' + str_y
                for z in range(8 - len(str_z)):
                    str_z = ' ' + str_z
                true_others[batch_ind * batch_size + i][j][6] = str_x
                true_others[batch_ind * batch_size + i][j][7] = str_y
                true_others[batch_ind * batch_size + i][j][8] = str_z
                f = open(pdb_path + "/all/" + filename, "a")
                if true_others[batch_ind * batch_size + i][j][0] != '#':
                    f.write(''.join(true_others[batch_ind * batch_size + i][j]))
                    f.write('\n')
                f.close()
                if true_ca1[i][j] != 1:
                    # side
                    str_x = str(format(pre_side[i][0][k1].item(), '.4f'))
                    str_y = str(format(pre_side[i][1][k1].item(), '.4f'))
                    str_z = str(format(pre_side[i][2][k1].item(), '.4f'))
                    k1 = k1 + 1
                    for x in range(8-len(str_x)):
                        str_x = ' ' + str_x
                    for y in range(8 - len(str_y)):
                        str_y = ' ' + str_y
                    for z in range(8-len(str_z)):
                        str_z = ' ' + str_z
                    true_others[batch_ind*batch_size+i][j][6] = str_x
                    true_others[batch_ind*batch_size+i][j][7] = str_y
                    true_others[batch_ind*batch_size+i][j][8] = str_z
                    f = open(pdb_path+"/side/"+filename, "a")
                    if true_others[batch_ind*batch_size+i][j][0] !='#':
                        f.write(''.join(true_others[batch_ind*batch_size+i][j]))
                        f.write('\n')
                    f.close()
                elif true_ca1[i][j] == 1:
                    # main
                    str_x = str(format(pre_main[i][0][k2].item(), '.4f'))
                    str_y = str(format(pre_main[i][1][k2].item(), '.4f'))
                    str_z = str(format(pre_main[i][2][k2].item(), '.4f'))
                    k2 = k2 + 1
                    for x in range(8-len(str_x)):
                        str_x = ' ' + str_x
                    for y in range(8 - len(str_y)):
                        str_y = ' ' + str_y
                    for z in range(8-len(str_z)):
                        str_z = ' ' + str_z
                    true_others[batch_ind*batch_size+i][j][6] = str_x
                    true_others[batch_ind*batch_size+i][j][7] = str_y
                    true_others[batch_ind*batch_size+i][j][8] = str_z
                    f = open(pdb_path+"/main/"+filename, "a")
                    if true_others[batch_ind*batch_size+i][j][0] !='#':
                        f.write(''.join(true_others[batch_ind*batch_size+i][j]))
                        f.write('\n')
                    f.close()

    # 计算RMSD
    all_rmsd = RMSD.print_rmsd(ref_path=ref_path+"/true_all", probe_path=pdb_path+'/all')
    main_rmsd = RMSD.print_rmsd(ref_path=ref_path+"/true_main", probe_path=pdb_path+'/main')
    side_rmsd = RMSD.print_rmsd(ref_path=ref_path+"/true_side", probe_path=pdb_path+'/side')
    print(" all_RMSD: ", all_rmsd,'\n',"main_RMSD: ", main_rmsd,'\n',"side_RMSD: ", side_rmsd)


def main():
    # python ./Correction_Module/corrected_test.py --test_file_path ./Correction_Module/data/test_af --test_save_path ./Correction_Module/output --test_model ./Correction_Module/model.pth --test_ref_path data/ref_data
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", default="data/test_af")
    parser.add_argument("--test_save_path", default="output")
    parser.add_argument("--test_model", default="corrected_model.pth")
    parser.add_argument("--test_ref_path", default="data/ref_data")
    args = parser.parse_args()
    test(model_path=args.test_model,#"E:/paper_data/5-fold-binding-data/after_correction/structure_model/870.pth",
         batch_size=32,
         pdb_path=args.test_save_path,#"E:/paper_data/5-fold-binding-data/after_correction/true/all_alphafold_ac",
         test_path=args.test_file_path,
         ref_path = args.test_ref_path)#"E:/paper_data/5-fold-binding-data/after_correction/true/all_alphafold")


if __name__ == "__main__":
    main()

import csv
import os
import random
import sys
import numpy as np
import pandas as pd
import h5py
import shutil
import str_feat
import argparse


def real_tcr_pep_part(pdb_files="C:/Users/lifan/Desktop/alphafold/alphafold_structure_pdb",
                      chain_files='C:/Users/lifan/Desktop/FAPE'):
    #将各链分开
    #true样本：C pep;  D  TCRA;  E  TCRB
    #af2样本 ：D pep;  B  TCRA;  C  TCRB
    filenames=os.listdir(pdb_files)
    print(filenames)
    os.makedirs(chain_files + '/D_true_pdb')
    os.makedirs(chain_files + '/E_true_pdb')
    os.makedirs(chain_files + '/C_true_pdb')
    for filename in filenames:
        E_file = open(chain_files+'/E_true_pdb/E_'+filename, 'a')  # TCRB
        C_file = open(chain_files+'/C_true_pdb/C_'+filename, 'a')  # peptides
        D_file = open(chain_files+'/D_true_pdb/D_'+filename, 'a')  # TCRA

        list1 = []
        try:
            file = open(pdb_files+'/'+filename, 'r')
        except FileNotFoundError:
            print('File is not found')
        else:
            lines = file.readlines()
            for line in lines:
                a = line.split()
                if len(a)>5:
                    x = a[4]
                    if x=="E":
                        E_file.write(line)
                    elif x=="C":
                        C_file.write(line)
                    elif x=="D":
                        D_file.write(line)
                    list1.append(x)
        file.close()


def af2_tcr_pep_part(pdb_files = "C:/Users/lifan/Desktop/alphafold/alphafold_structure_pdb",
                     chain_files = 'C:/Users/lifan/Desktop/FAPE'):
    # 将各链分开
    # true样本：C pep;  D  TCRA;  E  TCRB
    # af2样本 ：D pep;  B  TCRA;  C  TCRB
    # af2样本 ：C pep;  A  TCRA;  B  TCRB
    filenames=os.listdir(pdb_files)
    print(filenames)
    is_exists = os.path.exists(chain_files)
    if not is_exists:
        os.makedirs(chain_files)
    os.makedirs(chain_files + '/D_PEP_pdb')
    os.makedirs(chain_files + '/C_TCRB_pdb')
    os.makedirs(chain_files + '/B_TCRA_pdb')
    os.makedirs(chain_files + '/TCRA_TCRB_pdb')
    for filename in filenames:
        C_file = open(chain_files+'/C_TCRB_pdb/C_'+filename, 'a')  # TCRB
        D_file = open(chain_files+'/D_PEP_pdb/D_'+filename, 'a')  # peptides
        B_file = open(chain_files+'/B_TCRA_pdb/B_'+filename, 'a')  # TCRA
        BC_file = open(chain_files + '/TCRA_TCRB_pdb/BC_' + filename, 'a')  # TCRA

        list1 = []
        try:
            file = open(pdb_files+'/'+filename, 'r')
        except FileNotFoundError:
            print('File is not found')
        else:
            lines = file.readlines()
            for line in lines:
                #print(line)
                a = line.split()
                if len(a)>5:
                    x = a[4]
                    if x=="A":
                        B_file.write(line)
                        BC_file.write(line)
                    elif x=="B":
                        C_file.write(line)
                        BC_file.write(line)
                    elif x=="C":
                        D_file.write(line)
                    list1.append(x)
        file.close()


def real_neg_file(chain_files='C:/Users/lifan/Desktop/FAPE',
             pdb_files="C:/Users/lifan/Desktop/loss_fape",
             neg_files="C:/Users/lifan/Desktop/loss_fape/neg_pdb"):

    filenames=os.listdir(pdb_files)
    os.makedirs(neg_files)

    for filename in filenames:
        D_file = open(chain_files+'/D_true_pdb/D_'+filename, 'r')  # TCRA
        E_file = open(chain_files+'/E_true_pdb/E_'+filename, 'r')  # TCRB
        C_true_file = open(chain_files+'/C_true_pdb/C_'+filename, 'r') #peptide
        C_true_line = C_true_file.readline().rsplit()

        peptide_filename = random.sample(filenames, 1)
        while (1):
            C_file = open(chain_files + '/C_true_pdb/C_' + peptide_filename[0], 'r')  # peptides随机选择
            C_line = C_file.readline().rsplit()
            if C_line[3]==C_true_line[3]:
                peptide_filename = random.sample(filenames, 1)
            else:
                C_file.close()
                break
        list1 = []

        file = open(neg_files+'/{}_{}'.format(filename, peptide_filename[0]), 'a')
        C_file = open(chain_files + '/C_true_pdb/C_' + peptide_filename[0], 'r')

        C_lines = C_file.readlines()
        for line in C_lines:
            file.write(line)
        D_lines = D_file.readlines()
        for line in D_lines:
            file.write(line)
        E_lines = E_file.readlines()
        for line in E_lines:
            file.write(line)

        file.close()


def af2_neg_file(chain_files='C:/Users/lifan/Desktop/FAPE',
             pdb_files="C:/Users/lifan/Desktop/loss_fape",
             neg_files="C:/Users/lifan/Desktop/loss_fape/neg_pdb"):

    filenames = os.listdir(pdb_files)

    for filename in filenames:
        B_file = open(chain_files+'/B_true_pdb/B_'+filename, 'r')  # TCRA'/B_true_pdb/B_'
        C_file = open(chain_files+'/C_true_pdb/C_'+filename, 'r')  # TCRB
        D_true_file = open(chain_files+'/D_true_pdb/D_'+filename, 'r') #peptide
        D_true_line = D_true_file.readline().rsplit()

        peptide_filename = random.sample(filenames, 1)
        while (1):
            D_file = open(chain_files + '/D_true_pdb/D_' + peptide_filename[0], 'r')  # peptides随机选择
            D_line = D_file.readline().rsplit()
            if D_line[3]==D_true_line[3]:
                peptide_filename = random.sample(filenames, 1)
            else:
                D_file.close()
                break
        list1 = []
        D_file = open(chain_files + '/D_true_pdb/D_' + peptide_filename[0], 'r')
        file = open(neg_files+'/{}_{}'.format(filename, peptide_filename[0]), 'a')

        B_lines = B_file.readlines()
        # print(B_lines)
        for line in B_lines:
            file.write(line)
        C_lines = C_file.readlines()
        for line in C_lines:
            file.write(line)
        D_lines = D_file.readlines()
        print(D_lines)
        for line in D_lines:
            #print(line)
            file.write(line)
        file.close()


def structure_to_seq(csv_file='C:/Users/lifan/Desktop/seq.csv',
                     pdb_files="E:/alphafold2/alphafold2_2_1/train&test/test",
                     label='1'):

    filenames=os.listdir(pdb_files)
    aa_dic={'GLY':'G',
            'ALA':'A',
            'VAL':'V',
            'LEU':'L',
            'ILE':'I',
            'PRO':'P',
            'PHE':'F',
            'TYR':'Y',
            'TRP':'W',
            'SER':'S',
            'THR':'T',
            'CYS':'C',
            'MET':'M',
            'ASN':'N',
            'GLN':'Q',
            'ASP':'D',
            'GLU':'E',
            'LYS':'K',
            'ARG':'R',
            'HIS':'H',
            'AARG':'R',
            'BARG':'R',
            'AHIS':'H'
            }
    TCRA_list = []
    TCRB_list = []
    pep_list = []
    name_list=[]
    for filename in filenames:
        print(filename)

        TCRA_seq=''
        TCRB_seq=''
        pep_seq=''

        try:
            file = open(pdb_files+'/{}'.format(filename), 'r')  #负样本
        except FileNotFoundError:
            print('File is not found')
        else:
            lines = file.readlines()
            seq_num = 0
            x = ' A'
            for line in lines:
                a = line.split()
                if a[0]=='ATOM' and  int(line[22:26])!=seq_num :
                    if line[20:22] == x :
                        print(aa_dic[line[17:20]],end='')
                    else:
                        print(':',end='')
                    seq_num=int(line[22:26])
                    x = line[20:22]
        file.close()
        print('\n')


def binding_random_train_test(all_files="E:/alphafold2/alphafold2_2_5/true_pdb",
                              train_path="E:/alphafold2/alphafold2_2_5/train&test/test_true",
                              test_path="E:/alphafold2/alphafold2_2_5/train&test/train_true",
                              train_num = 96):
    files = os.listdir(all_files)
    os.makedirs(train_path)
    os.makedirs(test_path)
    train_files = random.sample(files,train_num)
    for file in files:
        if file in train_files:
            shutil.copyfile(all_files+"/{}".format(file),train_path+"/{}".format(file))
        else:
            shutil.copyfile(all_files+"/{}".format(file),test_path+"/{}".format(file))


def hdf_file(emb_p_files="E:/data/structure_data_hdf/CDR3_3alphafold/test_true",
             emb_n_files="E:/data/structure_data_hdf/CDR3_3alphafold/test_neg",
             hdf_path='E:/data/structure_data_hdf/CDR3_3alphafold/Test_CDR3_3alphafold.hdf'):

    hdf = h5py.File(hdf_path,'a')#训练集
    filenames=os.listdir(emb_p_files)
    filenames_neg=os.listdir(emb_n_files)
    for key in filenames:
        print(key)
        file = open(emb_p_files+'/{}'.format(key), 'r')
        lines = file.readlines()
        lines_list = []
        for line in lines:
            a = line.split()
            float_a = list(map(float, a))
            lines_list.append(float_a)
        hdf.create_group(key)
        hdf[key]["data"] = lines_list
        hdf[key].attrs["binding"] = 1
    for key in filenames_neg:
        print(key)
        file = open(emb_n_files+'/{}'.format(key), 'r')
        lines = file.readlines()
        lines_list = []
        for line in lines:
            a = line.split()
            float_a = list(map(float, a))
            lines_list.append(float_a)
        hdf.create_group(key)
        hdf[key]["data"] = lines_list
        hdf[key].attrs["binding"] = 0


if __name__ == "__main__":
    # python binding_sample_process.py --pdb_files data/real_structures
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_files", default='data/real_structures')
    args = parser.parse_args()

    positive_path = args.pdb_files  # 'data/CDR3_true_no_mhc'
    positive_emb_path = positive_path + '_emb'
    negative_path = positive_path + '_neg'
    negative_emb_path = negative_path + '_emb'

    real_tcr_pep_part(pdb_files=positive_path,
                     chain_files=positive_path + '_chains')

    real_neg_file(chain_files=positive_path + '_chains',
                  pdb_files=positive_path,
                  neg_files=negative_path)

    os.makedirs(positive_emb_path)
    for file in os.listdir(positive_path):
        str_feat.depict_pic(
            pdb_file=positive_path + '/' + file,
            emb_file=positive_emb_path + '/emb_' + file)

    os.makedirs(negative_emb_path)
    for file in os.listdir(negative_path):
        str_feat.depict_pic(
            pdb_file=negative_path + '/' + file,
            emb_file=negative_emb_path + '/emb_' + file)

    binding_random_train_test(all_files=positive_emb_path,
                              train_path="data/p_train",
                              test_path="data/p_test",
                              train_num=100)
    binding_random_train_test(all_files=negative_emb_path,
                              train_path="data/n_train",
                              test_path="data/n_test",
                              train_num=100)
    hdf_file(emb_p_files="data/p_train",
             emb_n_files="data/n_train",
             hdf_path="data/Train.hdf")
    hdf_file(emb_p_files="data/p_test",
             emb_n_files="data/n_test",
             hdf_path="data/Test.hdf")

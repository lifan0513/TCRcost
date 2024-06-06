import os
import sys
from numpy import *
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import alphafold_loss
import LDDT_loss
import structure_loss
from corrected_model
import argparse


def all_data_read(file_path):
    files = os.listdir(file_path)
    file_xyz=[]
    file_others = []
    files_ca = []
    files_ca1 = []
    files_chain1_D, files_chain2_B, files_chain3_C = [], [], []  # D,B,C
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
            chain1_D,chain2_B,chain3_C=[],[],[] # D,B,C
            i=-1
            for line in lines:

                a=[line[0:4],line[4:11],line[11:17],line[17:20],line[20:22],line[22:30],line[30:38],line[38:46],line[46:54],line[54:60],line[60:66],line[66:78]]
                # a = ['ATOM', '3068', 'N', 'LEU', 'C', '1', 82.338, 9.558, 9.331, '1.00', '24.78', 'N']
                if a[0] == 'ATOM':
                    i=i+1
                    if a[2] == '  CA  ':
                            ca.append(1)
                            ca1.append(1)
                            if a[4] == ' D':
                                chain1_D.append(1)
                                chain1_D.append(1)
                                chain1_D.append(1)
                                chain2_B.append(0)
                                chain2_B.append(0)
                                chain2_B.append(0)
                                chain3_C.append(0)
                                chain3_C.append(0)
                                chain3_C.append(0)
                            elif a[4] == ' B':
                                chain1_D.append(0)
                                chain1_D.append(0)
                                chain1_D.append(0)
                                chain2_B.append(1)
                                chain2_B.append(1)
                                chain2_B.append(1)
                                chain3_C.append(0)
                                chain3_C.append(0)
                                chain3_C.append(0)
                            elif a[4] == ' C':
                                chain1_D.append(0)
                                chain1_D.append(0)
                                chain1_D.append(0)
                                chain2_B.append(0)
                                chain2_B.append(0)
                                chain2_B.append(0)
                                chain3_C.append(1)
                                chain3_C.append(1)
                                chain3_C.append(1)
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
                    #if a[11][0]!=' ':
                    #    print(file)

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
            for j in range(150-len(chain3_C)):
                chain3_C.append(0)
                chain2_B.append(0)
                chain1_D.append(0)

            pr_xyz.append(atom_x[:400])
            pr_xyz.append(atom_y[:400])
            pr_xyz.append(atom_z[:400])
            file_xyz.append(pr_xyz)
            file_others.append(pr_others)
            files_ca.append(ca[:400])
            files_ca1.append(ca1[:400])
            files_chain3_C.append(chain3_C[:150])
            files_chain1_D.append(chain1_D[:150])
            files_chain2_B.append(chain2_B[:150])

    data = torch.tensor(file_xyz)
    data = data.reshape([len(data),3,400])
    others = file_others
    data_ca = torch.tensor(files_ca)
    data_ca1 = torch.tensor(files_ca1)
    data_files_chain3_C = torch.tensor(files_chain3_C)
    data_files_chain1_D = torch.tensor(files_chain1_D)
    data_files_chain2_B = torch.tensor(files_chain2_B)
    return data,others,files,data_ca,data_ca1,data_files_chain1_D,data_files_chain2_B,data_files_chain3_C


def train(batch_size=16,
          epochs=5000,
          learning_rate=0.001,
          train_test_path="C:/Users/lifan/Desktop/model",
          train_test_real_path="E:/paper_data/5-fold-structure-data/4",
          model_path="C:/Users/lifan/Desktop/model"):

    model = cmodel(verbose=0)

    side_lstm = list(map(id,model.side_lstm.parameters()))
    side_ln = list(map(id,model.side_ln.parameters()))
    base_params = filter(lambda p: id(p) not in side_lstm + side_ln,
                     model.parameters())
    optimizer = optim.Adam([{"params":base_params},
        {"params":model.side_lstm.parameters(),"lr":1e-5},
        {"params":model.side_ln.parameters(),"lr":1e-5},], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    
    """
    # 1DCNN model          
    model = multi_task_3_1dcnn(verbose=0)
    main_conv1 = list(map(id, model.main_conv1.parameters()))
    main_bn1 = list(map(id, model.main_bn1.parameters()))
    main_conv2 = list(map(id, model.main_conv2.parameters()))
    main_bn2 = list(map(id, model.main_bn2.parameters()))
    main_conv3 = list(map(id, model.main_conv3.parameters()))
    main_bn3 = list(map(id, model.main_bn3.parameters()))
    base_params = filter(lambda p: id(p) not in main_conv1 + main_conv2 + main_conv3 + main_bn1 + main_bn2 + main_bn3,
                         model.parameters())
    optimizer = optim.Adam([
        {"params": base_params},
        {"params": model.main_conv1.parameters(), "lr": 0.001},
        {"params": model.main_bn1.parameters(), "lr": 0.001},
        {"params": model.main_conv2.parameters(), "lr": 0.001},
        {"params": model.main_bn2.parameters(), "lr": 0.001},
        {"params": model.main_conv3.parameters(), "lr": 0.001},
        {"params": model.main_bn3.parameters(), "lr": 0.001}
        # {"params":model.side_lstm.parameters(),"lr":1e-5},
        # {"params":model.side_ln.parameters(),"lr":1e-5},
    ]
        , lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
        
    # 2DCNN model
    model = multi_task_3_2dcnn(verbose=0)
    main_conv1 = list(map(id, model.main_conv1.parameters()))
    main_bn1 = list(map(id, model.main_bn1.parameters()))
    main_conv2 = list(map(id, model.main_conv2.parameters()))
    main_bn2 = list(map(id, model.main_bn2.parameters()))
    main_conv3 = list(map(id, model.main_conv3.parameters()))
    main_bn3 = list(map(id, model.main_bn3.parameters()))
    base_params = filter(lambda p: id(p) not in main_conv1 + main_conv2 + main_conv3 + main_bn1 + main_bn2 + main_bn3,
                         model.parameters())
    optimizer = optim.Adam([
        {"params": base_params},
        {"params": model.main_conv1.parameters(), "lr": 0.001},
        {"params": model.main_bn1.parameters(), "lr": 0.001},
        {"params": model.main_conv2.parameters(), "lr": 0.001},
        {"params": model.main_bn2.parameters(), "lr": 0.001},
        {"params": model.main_conv3.parameters(), "lr": 0.001},
        {"params": model.main_bn3.parameters(), "lr": 0.001}
        # {"params":model.side_lstm.parameters(),"lr":1e-5},
        # {"params":model.side_ln.parameters(),"lr":1e-5},
    ]
        , lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

        
    # LSTM_ALL model
    model = all_all_all(verbose=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # LSTM_SIDE model
    model = all_side_all(verbose=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    
    # LSTM_MAIN model
    model = all_main_all(verbose=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    
    """


    alphafold_data, alphafold_others, alphafold_files,alphafold_data_ca, alphafold_data_ca1,data_files_chain1_D,data_files_chain2_B,data_files_chain3_C = all_data_read(train_test_path)#+"/train_af")#"C:/Users/lifan/Desktop/model/change_alphafold_train"
    true_data, true_others, true_files,true_data_ca,true_data_ca1,_,_,_ = all_data_read(train_test_real_path)#(train_test_path+"/train_true")

    data = TensorDataset(alphafold_data,true_data,alphafold_data_ca,true_data_ca,alphafold_data_ca1,true_data_ca1,alphafold_data_ca1,data_files_chain1_D,data_files_chain2_B,data_files_chain3_C)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    start_epoch = 0
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(model_path+"/model.pth"):
        checkpoint = torch.load(model_path+"/model.pth")#(model_path + '/final.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch,epochs):
        epoch_loss,epoch_loss1,epoch_loss2,epoch_loss3 =[],[],[],[]
        for batch_ind,batch_data in enumerate(data_loader):
            alphafold_input, true_output,alphafold_ca,true_ca,alphafold_ca1,true_ca1,alphafold_data_ca1,data_files_chain1_D,data_files_chain2_B,data_files_chain3_C = batch_data
            alphafold_input=alphafold_input.to(torch.float32)
            batch_ca_main = []
            batch_ca_side = []
            random_ca1 = torch.zeros(len(true_ca1),400)
            for i in range(len(true_ca1)):
                for j in range(torch.sum(true_ca1[i])):
                    random_ca1[i][j] = 1

            for i in range(len(true_ca1)):#range(len(random_ca1)):

                k1,k2 = 0,0
                pr_ca_main = np.zeros((150,400))
                pr_ca_side = np.zeros((400,400))
                for j in range(len(true_ca1[i])):
                    if true_ca1[i][j] == 1:
                        if k1>=150:
                            break
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
            batch_main = torch.tensor(batch_ca_main,dtype=float)
            batch_side = torch.tensor(batch_ca_side,dtype=float)

            true_main = torch.matmul(true_output.float(), batch_main.float())  # true_output * ca
            true_side = torch.matmul(true_output.float(), batch_side.float())  # true_output * (1-ca)
            true_all = true_output

            pre_main,pre_side,pre_all = model(alphafold_input,batch_main,batch_side)

            optimizer.zero_grad()

            """
            #mse
            loss_mse = nn.MSELoss().float()
            loss_main = loss_mse(pre_main, true_main)/len(pre_main)
            loss_side = loss_mse(pre_side, true_side)/len(pre_main)
            loss_all = loss_mse(pre_all, true_all)/len(pre_main)
            
            # lddt
            loss_side = 1 - LDDT_loss.lddt_loss(pre_side, true_side, None)
            loss_main = 1 - LDDT_loss.lddt_loss(pre_main, true_main, None)
            loss_all = 1 - LDDT_loss.lddt_loss(pre_all, true_all, None)
            
            # fape
            loss_main = alphafold_loss.fape_loss(len(pre_main), None, pre_main, true_main, train=True)
            loss_side = alphafold_loss.fape_loss(len(pre_side), None, pre_side, true_side, train=True)
            loss_all = alphafold_loss.fape_loss(len(pre_all), true_ca, pre_all, true_all, train=True)
            """
            # TCRcost
            loss_main = alphafold_loss.fape_loss(len(pre_main), None, pre_main, true_main, train=True)
            loss_side = 1-LDDT_loss.lddt_loss(pre_side, true_side, None)
            loss_all = alphafold_loss.fape_loss(len(pre_all), true_ca, pre_all, true_all, train=True) + \
                       structure_loss.loss4(pre_all.float(), true_ca)
            loss1 = loss_main
            loss2 = loss_side
            loss3 = loss_all

            loss = loss3 + loss2 + loss1

            loss.backward()
            """for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                ' -->grad_value:', parms.grad)"""

            optimizer.step()
            epoch_loss.append(loss.item())
            epoch_loss1.append(loss1.item())
            epoch_loss2.append(loss2.item())
            epoch_loss3.append(loss3.item())

        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "epoch": epoch
        }
        #torch.save(checkpoint_dict, model_path + '/final.pth')  # .format(epoch))

        if epoch%10==0:
            torch.save(checkpoint_dict, model_path + '/{}.pth'.format(epoch+1))

        print('Epoch[{}/{}],main_Loss:{},side_Loss:{},all_Loss:{},Loss: {}'.format(epoch + 1, epochs, mean(epoch_loss1), mean(epoch_loss2), mean(epoch_loss3), mean(epoch_loss)))


def main():
    # python corrected_train.py --epoch 1000 --lr 0.05 --train_file_path data/train_af --train_real_file_path data/train_true --model_path model
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int,default=1000)
    parser.add_argument("--lr", type=float,default=0.05)
    parser.add_argument("--train_file_path", default="E:/paper_data/5-fold-structure-data/4/train_af")
    parser.add_argument("--train_real_file_path", default="E:/paper_data/5-fold-structure-data/4/train_true")
    parser.add_argument("--model_path", default="E:/paper_data/5-fold-structure-data/4/model_mainCNN_sideLSTM")
    args = parser.parse_args()

    train(batch_size=32, epochs=args.epoch, learning_rate=args.lr,
          train_test_path=args.train_file_path,  #"E:/paper_data/5-fold-structure-data/4",
          train_test_real_path=args.train_real_file_path,  # "E:/paper_data/5-fold-structure-data/4",
          model_path=args.model_path)#"E:/paper_data/5-fold-structure-data/4/model_mainCNN_sideLSTM")


if __name__ == "__main__":
    main()

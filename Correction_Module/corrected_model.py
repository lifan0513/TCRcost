import os
import sys
import math
import numbers
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import r3

class cmodel(nn.Module):
    def __init__(self, verbose=1):

        super(cmodel, self).__init__()
        self.verbose = verbose
        self.all_conv1 = nn.Conv1d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.all_bn1 = nn.BatchNorm1d(3)

        # lstm
        self.main_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.main_ln = nn.LayerNorm((150, 3))
        self.side_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.side_ln = nn.LayerNorm((400, 3))
        self.all_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.all_ln = nn.LayerNorm((400, 3))

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()


    def forward(self, x,main,side):
        # all
        x_all = self.all_conv1(x)
        x_all = self.all_bn1(x_all)
        x_all = self.lrelu(x_all)

        # lstm
        x_main = torch.matmul(x_all, main.float())  # x * ca
        x_main = x_main.transpose(1, 2)
        x_main, (ht, ct) = self.main_lstm(x_main)
        x_main = self.main_ln(x_main)
        x_main = x_main.transpose(1, 2)

        x_side = torch.matmul(x_all, side.float())
        x_side = x_side.transpose(1, 2)
        x_side, (ht, ct) = self.side_lstm(x_side)
        x_side = self.side_ln(x_side)
        x_side = x_side.transpose(1, 2)

        xmain = torch.matmul(x_main, main.transpose(1, 2).float())
        xside = torch.matmul(x_side, side.transpose(1, 2).float())  # x * (1 - ca)
        x_all2 = xmain + xside
        x_all2 = x_all2.transpose(1, 2)
        x_all2, (ht, ct) = self.all_lstm(x_all2)
        x_all2 = self.all_ln(x_all2)
        x_all2 = x_all2.transpose(1, 2)

        y_main = x_main
        y_side = x_side
        y_all = x_all2

        return y_main, y_side, y_all


class all_main_all(nn.Module):
    def __init__(self, verbose=1):

        super(all_main_all, self).__init__()
        self.verbose = verbose
        self.all_conv1 = nn.Conv1d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.all_bn1 = nn.BatchNorm1d(3)
        
        # lstm
        self.main_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.main_ln = nn.LayerNorm((150, 3))
        self.side_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.side_ln = nn.LayerNorm((400, 3))
        self.all_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.all_ln = nn.LayerNorm((400, 3))

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()


    def forward(self, x,main,side):
        # main   main+side   lstm
        # xside = torch.matmul(torch.matmul(x, side.float()), side.transpose(1, 2).float())  # x * (1 - ca)
        x = self.lrelu(self.all_bn1(self.all_conv1(x)))  # all
        xside = torch.matmul(torch.matmul(x, side.float()), side.transpose(1, 2).float())  # x * (1 - ca)
        x_main = torch.matmul(x, main.float())  # x * ca
        x_main = x_main.transpose(1, 2)
        x_main, (ht, ct) = self.main_lstm(x_main)
        x_main = self.main_ln(x_main)
        x_main = x_main.transpose(1, 2)

        xmain = torch.matmul(x_main, main.transpose(1, 2).float())
        x2 = xmain + xside
        x2 = x2.transpose(1, 2)
        x2, (ht, ct) = self.all_lstm(x2)
        x2 = self.all_ln(x2)
        x2 = x2.transpose(1, 2)

        y_main = x_main
        y_all = x2

        return y_main, y_all


class all_side_all(nn.Module):
    def __init__(self, verbose=1):
        super(all_side_all, self).__init__()
        self.verbose = verbose
        self.all_conv1 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.all_bn1 = nn.BatchNorm1d(3)

        # lstm
        self.main_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.main_ln = nn.LayerNorm((150, 3))
        self.side_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.side_ln = nn.LayerNorm((400, 3))
        self.all_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.all_ln = nn.LayerNorm((400, 3))

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, x, main, side):
        # side   main+side   lstm
        x = self.relu(self.all_bn1(self.all_conv1(x)))  # all
        xmain = torch.matmul(torch.matmul(x, main.float()), main.transpose(1, 2).float())  # x * (1 - ca)
        x_side = torch.matmul(x, side.float())  # x * ca
        x_side = x_side.transpose(1, 2)
        x_side, (ht, ct) = self.side_lstm(x_side)
        x_side = self.side_ln(x_side)
        x_side = x_side.transpose(1, 2)

        xside = torch.matmul(x_side, side.transpose(1, 2).float())  # torch.matmul(x1, main.transpose(1, 2).float())
        x2 = xmain + xside
        x2 = x2.transpose(1, 2)
        x2, (ht, ct) = self.all_lstm(x2)
        x2 = self.all_ln(x2)
        x2 = x2.transpose(1, 2)

        y_side = x_side
        y_all = x2

        return y_side, y_all


class all_all_all(nn.Module):
    def __init__(self, verbose=1):
        super(all_all_all, self).__init__()
        self.verbose = verbose
        self.all_conv1 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.all_bn1 = nn.BatchNorm1d(3)

        # lstm
        #self.main_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        #self.main_ln = nn.LayerNorm((150, 3))
        #self.side_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        #self.side_ln = nn.LayerNorm((400, 3))
        self.all0_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.all0_ln = nn.LayerNorm((400, 3))

        self.all_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        self.all_ln = nn.LayerNorm((400, 3))

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, x, main, side):
        # side   main+side   lstm
        x = self.relu(self.all_bn1(self.all_conv1(x)))  # all

        x_all0 = x.transpose(1, 2)
        x_all0, (ht, ct) = self.all0_lstm(x_all0)
        x_all0 = self.all0_ln(x_all0)
        #xmain = torch.matmul(torch.matmul(x, main.float()), main.transpose(1, 2).float())  # x * (1 - ca)
        #x_side = torch.matmul(x, side.float())  # x * ca
        #x_side = x_side.transpose(1, 2)
        #x_side, (ht, ct) = self.side_lstm(x_side)
        #x_side = self.side_ln(x_side)
        #x_side = x_side.transpose(1, 2)

        #xside = torch.matmul(x_side, side.transpose(1, 2).float())  # torch.matmul(x1, main.transpose(1, 2).float())
        #x2 = xmain + xside
        #x2 = x_all0.transpose(1, 2)
        x2, (ht, ct) = self.all_lstm(x_all0)
        x2 = self.all_ln(x2)
        x2 = x2.transpose(1, 2)

        #y_side = x_side
        y_all = x2

        return y_all


class multi_task_3_1dcnn(nn.Module):
    def __init__(self, verbose=1):
        super(multi_task_3_1dcnn, self).__init__()
        self.verbose = verbose
        self.all_conv1 = nn.Conv1d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.all_bn1 = nn.BatchNorm1d(3)

        self.main_conv1 = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.main_bn1 = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.3)
        self.main_conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.main_bn2 = nn.BatchNorm1d(8)
        self.main_conv3 = nn.Conv1d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.main_bn3 = nn.BatchNorm1d(3)

        self.side_conv1 = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.side_bn1 = nn.BatchNorm1d(4)
        self.side_conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.side_bn2 = nn.BatchNorm1d(8)
        self.side_conv3 = nn.Conv1d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.side_bn3 = nn.BatchNorm1d(3)

        self.all_conv2 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.all_bn2 = nn.BatchNorm1d(8)
        self.all_conv3 = nn.Conv1d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.all_bn3 = nn.BatchNorm1d(3)
        
        # lstm
        #self.main_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        #self.main_ln = nn.LayerNorm((150, 3))
        #self.side_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        #self.side_ln = nn.LayerNorm((400, 3))
        #self.all_lstm = nn.LSTM(3, 3, num_layers=2, bidirectional=False, batch_first=True, dropout=0.3)
        #self.all_ln = nn.LayerNorm((400, 3))

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()


    def forward(self, x,main,side):
        # all
        #print("all1:",torch.var(x).item(),torch.var(x,0).sum().item(),end='\t')
        x_all = self.all_conv1(x)
        x_all = self.all_bn1(x_all)
        x_all = self.lrelu(x_all)


        # 1dcnn
        x_main = torch.matmul(x_all, main.float())  # x * ca
        x_main = self.lrelu(self.main_bn1(self.main_conv1(x_main)))   # main
        x_main = self.lrelu(self.main_bn2(self.main_conv2(x_main)))   # main
        x_main = self.lrelu(self.main_bn3(self.main_conv3(x_main)))     # main

        x_side = torch.matmul(x_all, side.float())
        x_side = self.lrelu(self.side_bn1(self.side_conv1(x_side)))  # side
        x_side = self.lrelu(self.side_bn2(self.side_conv2(x_side)))  # side
        x_side = self.lrelu(self.side_bn3(self.side_conv3(x_side)))  # side

        xmain = torch.matmul(x_main, main.transpose(1,2).float())
        xside = torch.matmul(x_side, side.transpose(1,2).float())   # x * (1 - ca)
        x_all2 = xmain + xside
        x_all2 = self.lrelu(self.all_bn2(self.all_conv2(x_all2)))
        x_all2 = self.all_bn3(self.all_conv3(x_all2))

        y_main = x_main
        y_side = x_side
        y_all = x_all2

        return y_main, y_side, y_all


class multi_task_3_2dcnn(nn.Module):
    def __init__(self, verbose=1):
        super(multi_task_3_2dcnn, self).__init__()
        self.verbose = verbose
        #self.all_conv1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1,padding_mode='circular')
        #self.all_bn1 = nn.BatchNorm2d(4)
        self.all_conv1 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.all_bn1 = nn.BatchNorm1d(3)

        self.main_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1,padding_mode='circular')
        self.main_bn1 = nn.BatchNorm2d(8)
        self.main_conv2 = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3,stride=1,padding=1,padding_mode='circular')
        self.main_bn2 = nn.BatchNorm2d(4)
        self.main_conv3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.main_bn3 = nn.BatchNorm2d(1)

        self.side_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1,padding_mode='circular')
        self.side_bn1 = nn.BatchNorm2d(8)
        self.side_conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1,padding_mode='circular')
        self.side_bn2 = nn.BatchNorm2d(4)
        self.side_conv3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1,padding=1,padding_mode='circular')
        self.side_bn3 = nn.BatchNorm2d(1)

        self.all_conv2 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.all_bn2 = nn.BatchNorm2d(4)
        self.all_conv3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.all_bn3 = nn.BatchNorm2d(1)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x, main, side):
        #x = x.unsqueeze(1)  # torch.Size([32, 1, 3, 400])
        x_all = self.lrelu(self.all_bn1(self.all_conv1(x)))  # torch.Size([32, 4, 3, 400])
        #print(x_all.shape,main.shape)
        #x_all = x_all.unsqueeze(1)
        x_main = torch.matmul(x_all, main.float())  # x * ca
        x_main = x_main.unsqueeze(1)
        #print(x_main.shape)
        x_main = self.lrelu(self.main_bn1(self.main_conv1(x_main)))  # main
        x_main = self.lrelu(self.main_bn2(self.main_conv2(x_main)))  # main
        x_main = self.lrelu(self.main_bn3(self.main_conv3(x_main)))  # main

        x_side = torch.matmul(x_all, side.float())
        x_side = x_side.unsqueeze(1)
        x_side = self.lrelu(self.side_bn1(self.side_conv1(x_side)))  # side
        x_side = self.lrelu(self.side_bn2(self.side_conv2(x_side)))  # side
        x_side = self.lrelu(self.side_bn3(self.side_conv3(x_side)))  # side

        xmain = torch.matmul(x_main.squeeze(1), main.transpose(1, 2).float())
        xside = torch.matmul(x_side.squeeze(1), side.transpose(1, 2).float())  # x * (1 - ca)
        #print(x_main.shape)
        x_all2 = xmain + xside
        x_all2 = x_all2.unsqueeze(1)
        x_all2 = self.lrelu(self.all_bn2(self.all_conv2(x_all2)))
        x_all2 = self.all_bn3(self.all_conv3(x_all2))
        #print(x_all2.shape)

        y_main = x_main.squeeze(1)
        y_side = x_side.squeeze(1)
        y_all = x_all2.squeeze(1)
        #print(y_all.shape)

        return y_main, y_side, y_all




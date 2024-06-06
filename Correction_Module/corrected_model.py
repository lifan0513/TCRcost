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



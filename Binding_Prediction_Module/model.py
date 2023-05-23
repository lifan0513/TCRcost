
import os
import sys
import math
import numbers
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from numpy.linalg import norm, svd

def strip_prefix_if_present(state_dict, prefix):
	keys = sorted(state_dict.keys())
	if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
		return

	for key in keys:
		newkey = key[len(prefix) :]
		state_dict[newkey] = state_dict.pop(key)

	try:
		metadata = state_dict._metadata
	except AttributeError:
		pass
	else:
		for key in list(metadata.keys()):
			if len(key) == 0:
				continue
			newkey = key[len(prefix) :]
			metadata[newkey] = metadata.pop(key)
def inexact_augmented_lagrange_multiplier(X, lmbda=.01, tol=1e-3,maxiter=100, verbose=True):
    #(X, lmbda=.01, tol=1e-3,maxiter=100, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two # mu = 1.25 / norm_two
    rho = 1.5 # rho = 1.5
    sv = 10. # sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        #print(A.shape,X.shape,E.shape)
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break
    if verbose:
        print("Finished at iteration %d" % (itr))
    return A, E
def inexact_augmented_lagrange_multiplier2(ten, lmbda=.00001, tol=1e-3, maxiter=100, verbose=True):
	# (X, lmbda=.01, tol=1e-3,maxiter=100, verbose=True):
	"""
    Inexact Augmented Lagrange Multiplier
    """
	Y = ten
	X = ten
	norm_two = torch.norm(Y.ravel(), 2)
	norm_inf = torch.norm(Y.ravel(), np.inf) / lmbda
	dual_norm = torch.max(norm_two, norm_inf)
	Y = Y / dual_norm
	A = torch.zeros(Y.shape)
	E = torch.zeros(Y.shape)
	dnorm = torch.norm(X, 'fro')
	mu = 1.25 / norm_two  # mu = 1.25 / norm_two
	rho = 1.5  # rho = 1.5
	sv = 10.  # sv = 10.
	n = Y.shape[0]
	itr = 0
	while True:
		Eraw = X - A + (1 / mu) * Y
		Eupdate = torch.maximum(Eraw - lmbda / mu,torch.tensor(0)) + torch.minimum(Eraw + lmbda / mu, torch.tensor(0))
		U, S, V = torch.linalg.svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)# full_matrices=False)
		svp = (S > 1 / mu).shape[0]
		if svp < sv:
			sv = min(svp + 1, n)
		else:
			sv = min(svp + round(.05 * n), n)
		Aupdate = torch.mm(torch.mm(U[:, :svp], torch.diag(S[:svp] - 1 / mu)), V[:svp, :])
		A = Aupdate
		E = Eupdate
		Z = X - A - E
		Y = Y + mu * Z
		mu = min(mu * rho, mu * 1e7)
		itr += 1
		if ((torch.norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
			break
	if verbose:
		print("Finished at iteration %d" % (itr))
	return A

class Model_3DCNN(nn.Module):
	def __init__(self, feat_dim=8, output_dim=1, num_filters=[32, 128, 256], use_cuda=True, verbose=1):
		super(Model_3DCNN, self).__init__()
        
		self.feat_dim = feat_dim
		self.output_dim = output_dim
		self.num_filters = num_filters
		self.use_cuda = use_cuda
		self.verbose = verbose

		self.conv_block1 = self.__conv_layer_set__(self.feat_dim, self.num_filters[0], 7, 2, 3)#(self.feat_dim, self.num_filters[0], 7, 2, 3)
		self.res_block1 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)#(self.num_filters[0], self.num_filters[0], 7, 1, 3)
		self.res_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)#(self.num_filters[0], self.num_filters[0], 7, 1, 3)

		self.conv_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[1], 7, 3, 3)#(self.num_filters[0], self.num_filters[1], 7, 3, 3)
		self.max_pool2 = nn.MaxPool3d(2)


		self.fc1 = nn.Linear(8192, 10)#nn.Linear(2048, 100)
		torch.nn.init.normal_(self.fc1.weight, 0, 1)
		self.fc1_bn = nn.BatchNorm1d(num_features=10, affine=True, momentum=0.1).train()
		self.fc2 = nn.Linear(10, 2)
		torch.nn.init.normal_(self.fc2.weight, 0, 1)
		self.fc2_bn = nn.BatchNorm1d(num_features=1, affine=True, momentum=0.1).train()
		self.relu = nn.ReLU()
		self.sigmoid=nn.Sigmoid()
		self.fc=nn.Sequential(
            nn.Linear(8192, 64),
            nn.Dropout(0.1),
			#nn.BatchNorm1d(64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 10))


	def __conv_layer_set__(self, in_c, out_c, k_size, stride, padding):
		conv_layer = nn.Sequential(
			nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(out_c))
		return conv_layer

	def forward(self, x):

		if x.dim() == 1:
			x = x.unsqueeze(-1)
		conv1_h = self.conv_block1(x)
		if self.verbose != 0:
			print(conv1_h.shape)

		conv1_res1_h = self.res_block1(conv1_h)
		if self.verbose != 0:
			print(conv1_res1_h.shape)

		conv1_res1_h2 = conv1_res1_h + conv1_h
		if self.verbose != 0:
			print(conv1_res1_h2.shape)

		conv1_res2_h = self.res_block2(conv1_res1_h2)
		if self.verbose != 0:
			print(conv1_res2_h.shape)

		conv2_h = self.conv_block2(conv1_res2_h)
		if self.verbose != 0:
			print(conv2_h.shape)

		pool2_h = self.max_pool2(conv2_h)
		if self.verbose != 0:
			print(pool2_h.shape)

		flatten_h = pool2_h.view(pool2_h.size(0), -1)
		if self.verbose != 0:
			print(flatten_h.shape)

		fc1=self.fc(flatten_h)
		fc1=self.sigmoid(fc1)

		fc2=self.fc2(fc1)
		return fc2,fc1


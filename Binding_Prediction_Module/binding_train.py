
import os
import sys
import pandas as pd
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset
from model import Model_3DCNN, strip_prefix_if_present,inexact_augmented_lagrange_multiplier
from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D
from file_util import *
from sklearn.metrics import *
import trpca
sys.stdout.flush()
sys.path.insert(0, "../common")
torch.set_printoptions(profile="full")

def worker_init_fn():
	np.random.seed(int(0))

def train(args,device,use_cuda):

	# load dataset
	is_crystal = False
	dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type, is_crystal=is_crystal, rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold)

	num_workers = 0

	# initialize data loader
	batch_count = len(dataset) // args.batch_size
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None)

	# define voxelizer, gaussian_filter
	voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=0)
	gaussian_filter = GaussianFilter(dim=3, channels=8, kernel_size=11, sigma=1, use_cuda=use_cuda)

	model = Model_3DCNN(use_cuda=use_cuda, verbose=0)
	model.to(device)
	
	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)

	# load model
	epoch_start = 0
	if not os.path.exists(os.path.dirname(args.model_path)):
		os.makedirs(os.path.dirname(args.model_path))
	output_dir = os.path.dirname(args.model_path)

	step = 0

	if os.path.exists(args.model_path + '/ac_0.005_5_model.pth'):
		checkpoint = torch.load(args.model_path + '/ac_0.005_5_model.pth')#(model_path + '/final.pth')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch_start = checkpoint['epoch']


	for epoch_ind in range(epoch_start, args.epoch_count):
		print("______________",epoch_ind,"___________________")
		vol_batch = torch.zeros((args.batch_size, 8, 48, 48, 48)).float().to(device)
		losses = []
		model.train()

		loss_fn = nn.CrossEntropyLoss()
		y_prob,y_true=[],[]
		for batch_ind, batch in enumerate(dataloader):

			x_batch_cpu, y_batch_cpu, name = batch
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)

			# voxelize into 3d volume
			feat_np=x_batch[:x_batch.shape[0], :, 3:].numpy()
			L, S, obj, err, i = trpca.trpca(feat_np, 1 / 48)
			#print(L)
			for i in range(x_batch.shape[0]):
				xyz, feat = x_batch[i, :, :3],torch.tensor(L[i,:,:]).float()
				vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
			
			# forward training
			ytrue_tensor = y_batch_cpu.squeeze()
			ytrue_tensor = ytrue_tensor.long()
			ypred_batch, ypred_batch_fc = model(vol_batch[:x_batch.shape[0]])

			loss = loss_fn(ypred_batch + 1e-8, ytrue_tensor)

			y_true.extend(ytrue_tensor.numpy())
			y_prob.extend(torch.softmax(ypred_batch, dim=1)[:, 1].detach().numpy())
			losses.append(loss.cpu().data.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			print("[%d/%d-%d/%d] training, loss: %f, lr: %.7f" % (epoch_ind+1, args.epoch_count, batch_ind+1, batch_count, loss.cpu().data.item(), optimizer.param_groups[0]['lr']))

			if epoch_ind % 5 == 4:
				checkpoint_dict = {
					"model_state_dict": model_to_save.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": loss,
					"step": step,
					"epoch": epoch_ind
				}
				torch.save(checkpoint_dict, args.data_dir+"/{}_model.pth".format(epoch_ind+1))
				print("checkpoint saved: "+args.data_dir+"/{}_model.pth".format(epoch_ind+1))
			step += 1

		print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, args.epoch_count, np.mean(losses)))

		y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]
		ACC = accuracy_score(y_true, y_pred)
		AUC = roc_auc_score(y_true, y_prob)
		Recall = recall_score(y_true, y_pred)
		Precision = precision_score(y_true, y_pred)
		F1 = f1_score(y_true, y_pred)
		print("ACC: %.3f,AUC: %.3f,Recall: %.3f,Precision: %.3f,F1: %.3f" % (ACC, AUC, Recall, Precision, F1))


def main():
	#python binding_train.py --data-dir --mlhdf-fn --model-path --epoch-count --learning-rate
	parser = argparse.ArgumentParser()
	parser.add_argument("--device-name", default="cpu", help="use cpu or cuda:0, cuda:1 ...")
	parser.add_argument("--data-dir", default="data", help="dataset directory")
	parser.add_argument("--dataset-type", type=float, default=1,
							help="ml-hdf version, (1: for fusion, 1.5: for cfusion 2: ml-hdf v2)")
	parser.add_argument("--mlhdf-fn", default="Train.hdf", help="training ml-hdf path")

	parser.add_argument("--model-path", default="models", help="model checkpoint file path")

	parser.add_argument("--rmsd-weight", action='store_false', default=0,
							help="whether rmsd-based weighted loss is used or not")
	parser.add_argument("--rmsd-threshold", type=float, default=2,
							help="rmsd cut-off threshold in case of docking data and/or --rmsd-weight is true")
	parser.add_argument("--epoch-count", type=int, default=50, help="number of training epochs")
	parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
	parser.add_argument("--learning-rate", type=float, default=0.005, help="initial learning rate")  # 0.0007
	parser.add_argument("--decay-rate", type=float, default=0.95, help="learning rate decay")
	parser.add_argument("--decay-iter", type=int, default=100, help="learning rate decay")
	parser.add_argument("--checkpoint-iter", type=int, default=50, help="checkpoint save rate")
	args = parser.parse_args()

	# set CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	cuda_count = torch.cuda.device_count()
	if use_cuda:
		device = torch.device(args.device_name)
		torch.cuda.set_device(int(args.device_name.split(':')[1]))
	else:
		device = torch.device("cpu")

	train(args, device, use_cuda)


if __name__ == "__main__":
	main()

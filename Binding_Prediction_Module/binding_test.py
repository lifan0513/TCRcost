
import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import *
from scipy.stats import *
from model import Model_3DCNN, strip_prefix_if_present
from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D
from file_util import *
import trpca


def eval(args,device,cuda_count,use_cuda):

	# load dataset
	if args.complex_type == 1:
		is_crystal = 1
	else:
		is_crystal = 0
	dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type, is_crystal=is_crystal, rmsd_weight=False, rmsd_thres=args.rmsd_threshold)

	# check multi-gpus
	num_workers = 0
	if args.multi_gpus and cuda_count > 1:
		num_workers = cuda_count

	# initialize data loader
	batch_size = args.batch_size
	batch_count = len(dataset) // batch_size
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

	# define voxelizer, gaussian_filter
	voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=0)

	# define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose=0)

	if args.multi_gpus and cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)

	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	# load model
	if not valid_file(args.model_path):
		print("checkpoint not found! %s" % args.model_path)
		return
	checkpoint = torch.load(args.model_path, map_location=device)
	model_state_dict = checkpoint.pop("model_state_dict")
	strip_prefix_if_present(model_state_dict, "module.")
	model_to_save.load_state_dict(model_state_dict, strict=False)
	output_dir = os.path.dirname(args.model_path)

	vol_batch = torch.zeros((batch_size, 8, 48, 48, 48)).float().to(device)
	ytrue_arr = np.zeros((len(dataset),), dtype=np.float32)
	ypred_arr = np.zeros((len(dataset),), dtype=np.float32)
	pred_list = []

	model.eval()
	with torch.no_grad():
		y_prob=[]
		for bind, batch in enumerate(dataloader):
		
			# transfer to GPU
			x_batch_cpu, y_batch_cpu ,name= batch
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
			# voxelize into 3d volume
			bsize = x_batch.shape[0]

			feat_np = x_batch[:x_batch.shape[0], :, 3:].numpy()
			L, S, obj, err, i = trpca.trpca(feat_np, 1 / 48)

			for i in range(bsize):
				xyz, feat = x_batch[i, :, :3], torch.tensor(L[i, :, :]).float()
				vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
			# forward training
			ypred_batch, zfeat_batch = model(vol_batch[:x_batch.shape[0]])
			y_prob.extend(torch.softmax(ypred_batch, dim=1)[:, 1].detach().numpy())
			ytrue = y_batch_cpu.float().data.numpy()[:,0]
			ypred = ypred_batch.cpu().float().data.numpy()[:,0]

			ytrue_arr[bind*batch_size:bind*batch_size+bsize] = ytrue
			ypred_arr[bind*batch_size:bind*batch_size+bsize] = ypred

			if args.save_pred:
				for i in range(bsize):
					pred_list.append([bind + i, ytrue[i], ypred[i]])

			print("[%d/%d] evaluating" % (bind+1, batch_count))
	y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]

	ACC=accuracy_score(ytrue_arr,y_pred)
	AUC= roc_auc_score(ytrue_arr,y_prob)
	Recall= recall_score(ytrue_arr,y_pred)
	Precision= precision_score(ytrue_arr,y_pred)
	F1= f1_score(ytrue_arr,y_pred)
	print("Evaluation Summary:")
	print("ACC: %.3f,AUC: %.3f,Recall: %.3f,Precision: %.3f,F1: %.3f" % (ACC,AUC,Recall,Precision,F1))


def main():
	# python binding_test.py --data-dir data --mlhdf-fn Test.hdf --model-path model.pth
	parser = argparse.ArgumentParser()
	parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
	parser.add_argument("--data-dir", default='data', help="dataset directory")
	parser.add_argument("--dataset-type", type=float, default=1, help="1: ml-hdf, 2: ml-hdf v2")
	parser.add_argument("--csv-fn", default="", help="csv file path")
	parser.add_argument("--mlhdf-fn", default='Test.hdf', help="ml-hdf path")
	parser.add_argument("--model-path", default='model.pth', help="model checkpoint file path")
	parser.add_argument("--complex-type", type=int, default=2, help="1: crystal, 2: docking")
	parser.add_argument("--rmsd-threshold", type=float, default=2,
							help="rmsd cut-off threshold in case of docking data and/or --rmsd-weight is true")
	parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
	parser.add_argument("--multi-gpus", default=False, action="store_true", help="whether to use multi-gpus")
	parser.add_argument("--save-pred", default=True, action="store_true",
						help="whether to save prediction results in csv")
	parser.add_argument("--save-feat", default=True, action="store_true",
						help="whether to save fully connected features in npy")
	args = parser.parse_args()

	# set CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	cuda_count = torch.cuda.device_count()
	if use_cuda:
		device = torch.device(args.device_name)
		torch.cuda.set_device(int(args.device_name.split(':')[1]))
	else:
		device = torch.device("cpu")

	eval(args, device, cuda_count, use_cuda)


if __name__ == "__main__":
	main()



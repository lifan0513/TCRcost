import os
import sys
import csv
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset

class Dataset_MLHDF(Dataset):
	def __init__(self, mlhdf_path, mlhdf_ver, is_crystal=False, rmsd_weight=False, rmsd_thres=2,max_atoms=5000, feat_dim=11):
		super(Dataset_MLHDF, self).__init__()
		self.mlhdf_ver = mlhdf_ver
		self.mlhdf_path = mlhdf_path
		self.is_crystal = is_crystal
		self.rmsd_weight = rmsd_weight
		self.rmsd_thres = rmsd_thres
		self.max_atoms = max_atoms
		self.feat_dim = feat_dim

		self.mlhdf = h5py.File(self.mlhdf_path, 'r')
		self.data_info_list = []
		if self.mlhdf_ver == 1: # for fusion model
			for comp_id in self.mlhdf.keys():
				if self.is_crystal:
					self.data_info_list.append([comp_id, 0, 0, 0])
				else:
					#pose_ids = self.mlhdf[comp_id]["pybel"]["processed"]["pdbbind"].keys()##["docking"].keys()
					pose_ids = self.mlhdf[comp_id].keys()  ##["docking"].keys()
					for pose_id in pose_ids:
						self.data_info_list.append([comp_id, pose_id, 0, 0])
		elif self.mlhdf_ver == 1.5: # for cfusion model
			if is_crystal:
				for pdbid in self.mlhdf["regression"].keys():
					affinity = float(self.mlhdf["regression"][pdbid].attrs["affinity"])
					self.data_info_list.append([pdbid, 0, 0, affinity])
			else:
				print("not supported!")

	def close(self):
		self.mlhdf.close()

	def __len__(self):
		count = len(self.data_info_list)
		return count

	def __getitem__(self, idx):
		pdbid, poseid, rmsd, affinity = self.data_info_list[idx]

		data = np.zeros((self.max_atoms, self.feat_dim), dtype=np.float32)
		if self.mlhdf_ver == 1:
			if self.is_crystal:
				mlhdf_ds = self.mlhdf[pdbid]["pybel"]["processed"]["crystal"]
			else:
				mlhdf_ds = self.mlhdf[pdbid]
			actual_data = mlhdf_ds["data"][:]
			affinity=self.mlhdf[pdbid].attrs["binding"]
			key = [pdbid]
			data[:actual_data.shape[0],:] = actual_data


		x = torch.tensor(data)
		y = torch.tensor(np.expand_dims(affinity, axis=0))

		if self.rmsd_weight == True:
			data_w = 0.5 + self.rmsd_thres - rmsd
			w = torch.tensor(np.expand_dims(data_w, axis=0))
			return x, y, w
		else:
			return x, y, key


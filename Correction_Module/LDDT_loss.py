import numpy as np
import torch
from math import sqrt
from torch.autograd import Variable
import torch.nn as nn
import os
torch.set_printoptions(profile="full")

def get_flattened(dmap):
    if dmap.ndim == 1:
        return dmap
    elif dmap.ndim == 2:
        #上三角矩阵
        dmap_up = []
        for i in range(len(dmap)):
            for j in range(len(dmap[0])):
                if j > i :
                    dmap_up.append(dmap[i][j])
        return torch.tensor(dmap_up)
        #return dmap[np.triu_indices_from(dmap,k=1)]
    else:
        assert False, "ERROR: the passes array has dimension not equal to 2 or 1!"


def get_separations(dmap):
    #t_indices = np.triu_indices_from(dmap, k=1)
    #print(t_indices)
    #print(np.abs(t_indices[0] - t_indices[1]))
    dmap_up_i, dmap_up_j = [], []
    for i in range(len(dmap)):
        for j in range(len(dmap[0])):
            if j > i:
                #dmap_up.append(dmap[i][j])
                dmap_up_i.append(i)
                dmap_up_j.append(j)
    t_indices_i = torch.tensor(dmap_up_i)
    t_indices_j = torch.tensor(dmap_up_j)
    separations = torch.abs(t_indices_i - t_indices_j)
    return separations


# return a 1D boolean array indicating where the sequence separation in the
# upper triangle meets the threshold comparison
def get_sep_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = get_flattened(dmap)
  separations = get_separations(dmap)
  if comparator == 'gt':
    threshed = separations > thresh
  elif comparator == 'lt':
    threshed = separations < thresh
  elif comparator == 'ge':
    threshed = separations >= thresh
  elif comparator == 'le':
    threshed = separations <= thresh

  return threshed

# return a 1D boolean array indicating where the distance in the
# upper triangle meets the threshold comparison
def get_dist_thresh_b_indices(dmap, thresh, comparator):
  assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
  dmap_flat = get_flattened(dmap)
  if comparator == 'gt':
    threshed = dmap_flat > thresh
  elif comparator == 'lt':
    threshed = dmap_flat < thresh
  elif comparator == 'ge':
    threshed = dmap_flat >= thresh
  elif comparator == 'le':
    threshed = dmap_flat <= thresh
  return threshed

def get_LDDT(true_map, pred_map, R=15, sep_thresh=-1, T_set=[0.5,1.0,2.0,4.0]):
    #true_map, pred_map 为距离矩阵

    # Helper for number preserved in a threshold
    def get_n_preserved(ref_flat, mod_flat, thresh):
        err = torch.abs(ref_flat - mod_flat)
        n_preserved = torch.sum((err < thresh))
        return n_preserved

    # flatten upper triangles
    true_flat_map = get_flattened(true_map)
    pred_flat_map = get_flattened(pred_map)
    #print("true_flat_map:",true_flat_map)
    #print("pred_flat_map:",pred_flat_map)

    # Find set L
    S_thresh_indices = get_sep_thresh_b_indices(true_map, sep_thresh, 'gt')#序列分布
    R_thresh_indices = get_dist_thresh_b_indices(true_flat_map, R, 'lt')#距离
    #print("S_thresh_indices:",S_thresh_indices)
    #print("R_thresh_indices:",R_thresh_indices)

    L_indices = S_thresh_indices & R_thresh_indices
    #print("L_indices:",L_indices)

    true_flat_in_L = true_flat_map[L_indices]
    pred_flat_in_L = pred_flat_map[L_indices]
    #print("true_flat_in_L:",true_flat_in_L)
    #print("pred_flat_in_L:", pred_flat_in_L)

    # Number of pairs in L
    #L_n = L_indices.sum()
    L_n = torch.sum(L_indices)
    #print("L_n:",L_n)

    # Calculated lDDT
    #preserved_fractions = []
    preserved_fractions = torch.tensor(0,dtype=float)

    for _thresh in T_set:
        _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)
        _f_preserved = _n_preserved / L_n
        #print(_thresh,"  _f_preserved:",_f_preserved)
        #preserved_fractions.append(_f_preserved)
        preserved_fractions += _f_preserved

    #lDDT = torch.mean(preserved_fractions)
    lDDT = preserved_fractions/len(T_set)
    #print("lDDT:",lDDT)
    return lDDT

def dist_map(position,ca):
    map = torch.full((ca.sum(),ca.sum()),torch.nan)
    map_i = -1
    for i in range(len(position[0])):
        if ca[i] == 1:
            x1, y1, z1 = position[0][i], position[1][i], position[2][i] #x,y,z
            map_i += 1
            map_j = -1
            for j in range(len(position[0])):
                if ca[j] == 1:
                    x2, y2, z2 = position[0][j], position[1][j], position[2][j]  # x,y,z
                    map_j += 1
                    map[map_i][map_j]=torch.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )
    return map

def alphafold2_lddt(pred_map,true_map,cutoff=15):
    #print((true_map < cutoff).shape)
    dists_to_score = (
            (true_map < cutoff)*(1. - torch.eye(true_map.shape[1]))  # Exclude self-interaction.
    )
    #print(dists_to_score)
    """dist_0_5, dist_1_0, dist_2_0, dist_4_0 = 0, 0, 0, 0

    for i in range(len(dists_to_score)):
        for j in range(len(dists_to_score[0])):
            if dists_to_score[i][j] == 1:
                pred_dist = torch.sqrt(torch.sum(pred_point[i]-pred_point[j]))
                dist = torch.abs(true_map[i][j] - pred_dist)
                if dist<0.5:
                    dist_0_5 += 1
                    dist_1_0 += 1
                    dist_2_0 += 1
                    dist_4_0 += 1
                elif dist<1.0:
                    dist_1_0 += 1
                    dist_2_0 += 1
                    dist_4_0 += 1
                elif dist<2.0:
                    dist_2_0 += 1
                    dist_4_0 += 1
                elif dist<4.0:
                    dist_4_0 += 1

    norm = 1. / (1e-10 + torch.sum(dists_to_score))
    score = 0.25 * (dist_0_5+dist_1_0+dist_2_0+dist_4_0) * norm"""
    # Shift unscored distances to be far away.
    dist_l1 = torch.abs(true_map - pred_map)
    #print(dist_l1)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    #print(torch.tensor((dist_l1 < 0.5),dtype=float).shape,(dist_l1 < 1.0).shape)
    """score = 0.25 * ((dist_l1 < 0.5).type(torch.float32)+
                    (dist_l1 < 1.0).type(torch.float32) +
                    (dist_l1 < 2.0).type(torch.float32) +
                    (dist_l1 < 4.0).type(torch.float32))"""

    # Normalize over the appropriate axes.
    relu = nn.ReLU()

    dist_0_5 = -1 * (dist_l1 - 0.5)
    x_0_5 = relu(dist_0_5)
    dist_score_0_5 = dists_to_score * x_0_5 / 0.5

    dist_1_0 = -1 * (dist_l1 - 1.0)
    x_1_0 = relu(dist_1_0)
    dist_score_1_0 = dists_to_score * x_1_0 / 1

    dist_2_0 = -1 * (dist_l1 - 2.0)
    x_2_0 = relu(dist_2_0)
    dist_score_2_0 = dists_to_score * x_2_0 / 2

    dist_4_0 = -1 * (dist_l1 - 4.0)
    x_4_0 = relu(dist_4_0)
    dist_score_4_0 = dists_to_score * x_4_0 / 4
    #print(dist_score_4_0)


    norm = 1. / (1e-10 + torch.sum(dists_to_score))
    score = norm * (1e-10 + torch.sum(dist_score_4_0)+
                    torch.sum(dist_score_2_0)+
                    torch.sum(dist_score_1_0)+
                    torch.sum(dist_score_0_5)) * 0.25 # torch.sum(dist_score)


    return score
def lddt_loss(pred_batch,true_batch,Ca):
    batch_lddt = 0
    if not Ca:
        Ca = torch.zeros((len(true_batch),len(pred_batch[0][0])))
        for i in range(len(pred_batch[0][0])):
            for j in range(len(true_batch)):
                if true_batch[j][0][i]!=0:
                    Ca[j][i] = 1

    for i in range(len(pred_batch)):
        #pred_map = dist_map(pred_batch[i],Ca[i])
        #true_map = dist_map(true_batch[i],Ca[i])
        ca = []
        for j in range(len(Ca[i])):
            if Ca[i][j] == 1:
                ca.append(j)
        this_ca = torch.zeros([len(pred_batch[0][0]), len(ca)]) # torch.zeros([400, len(ca)])
        k = 0
        for c in ca:
            this_ca[c][k] = 1
            k = k + 1
        pred_point = torch.matmul(pred_batch[i], this_ca).t()#仅Ca原子
        true_point = torch.matmul(true_batch[i], this_ca).t()#仅Ca原子
        #print("pred_point_ca:",pred_point)

        #pred_point = pred_batch[i].t()#全部原子
        #true_point = true_batch[i].t()#全部原子
        #print("pred_point:", pred_point)

        pred_map = torch.sqrt(1e-10 + torch.sum((pred_point[ :, None] - pred_point[ None, :])**2,dim=-1))
        #print(pred_map)
        true_map = torch.sqrt(1e-10 + torch.sum((true_point[ :, None] - true_point[ None, :])**2,dim=-1))
        #print("pred_map",pred_map)
        #print("true_map", true_map.shape)

        #lddt = get_LDDT(true_map, pred_map, R=15, sep_thresh=-1, T_set=[0.5, 1, 2, 4])
        #print(lddt)
        lddt = alphafold2_lddt(pred_map, true_map, cutoff=15)
        #print(lddt)

        batch_lddt += lddt
    #batch_lddt=batch_lddt0
    #print("batch_lddt:",batch_lddt)
    return batch_lddt/len(pred_batch)

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
                        ca.append(0)
                        ca1.append(1)
                elif a[2] == '  N   ':
                    ca1.append(1)
                    if a[3] == 'PRO':
                        ca.append(11)#(22)
                    else:
                        ca.append(1)#(2)
                elif a[2] == '  C   ':
                    ca.append(2)#(3)
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
                    print(file)

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
if __name__ == "__main__":
    #a = torch.tensor([[[1, 2, 3,4], [1, 2, 3,3], [1, 2, 3,5]]])
    #b = torch.tensor([[[1, 1, 1,3], [1, 5, 1,2], [1, 1, 1,2]]])
    #lddt_loss(a,b,a)
    #输出距离矩阵
    path = 'E:/alphafold_structure/1DCNN/1DCNN_train_alphafold'
    w_path = 'C:/Users/lifan/Desktop/model/肽键长度/af'#/train_data_dist_map.txt'
    true_data, true_others, true_files, true_data_ca, true_data_ca1 = all_data_read(path)
    for i in range(len(true_files)):
        print("*"*30,'\n',true_files[i])
        position = true_data[i]
        ca = true_data_ca[i]
        #map = list(dist_map(position, ca))
        map = []
        for k in range(len(position[0])):
            if ca[k] == 2:
                x1, y1, z1 = position[0][k], position[1][k], position[2][k]  # x,y,z
                for j in range(k,len(position[0])):
                    if ca[j] == 1 or ca[j] == 11:
                        x2, y2, z2 = position[0][j], position[1][j], position[2][j]  # x,y,z
                        map.append(torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))
                        break
        #print(map.shape)
        wf = open(w_path + "/" + true_files[i], "a")
        for l in map:
            wf.write(str(format(l, '.3f')))
            wf.write('\n')
        """for l in map:
            for a in l:
                wf.write(str(format(a.item(),'.3f')))
                wf.write('\t')
            wf.write('\n')"""
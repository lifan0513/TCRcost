import collections
import random
from numpy import *
import torch
import r3
import numpy as np
from typing import Optional
#torch.set_printoptions(profile="full")
#import jax

def frame_aligned_point_error(
    pred_frames: r3.Rigids,  # shape (num_frames)
    target_frames: r3.Rigids,  # shape (num_frames)
    frames_mask: np.ndarray,  # shape (num_frames)
    pred_positions: r3.Vecs,  # shape (num_positions)
    target_positions: r3.Vecs,  # shape (num_positions)
    positions_mask: np.ndarray,  # shape (num_positions)
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    epsilon=1e-4) -> np.ndarray:  # shape ()

  #assert pred_frames.rot.xx.ndim == 15
  #assert target_frames.rot.xx.ndim == 1
  assert frames_mask.ndim == 1, frames_mask.ndim
  assert pred_positions.x.ndim == 1
  assert target_positions.x.ndim == 1
  assert positions_mask.ndim == 1

  # Compute array of predicted positions in the predicted frames.
  # r3.Vecs (num_frames, num_positions)
  """local_pred_pos = r3.rigids_mul_vecs(
      jax.tree_map(lambda r: r[:, None], r3.invert_rigids(pred_frames)),#增加一个维度，且都赋予none
      jax.tree_map(lambda x: x[None, :], pred_positions))
  print(local_pred_pos)

  # Compute array of target positions in the target frames.
  # r3.Vecs (num_frames, num_positions)
  local_target_pos = r3.rigids_mul_vecs(
      jax.tree_map(lambda r: r[:, None], r3.invert_rigids(target_frames)),
      jax.tree_map(lambda x: x[None, :], target_positions))"""
  #local_pred_pos = pred_positions
  local_pred_pos = r3.rigids_mul_vecs(r3.invert_rigids(pred_frames), pred_positions)#矩阵相乘，获得变换后坐标

  local_target_pos = r3.rigids_mul_vecs(r3.invert_rigids(target_frames), target_positions)
  #local_target_pos = r3.rigids_mul_vecs(pred_frames, local_target_pos)
  # Compute errors between the structures.
  # jnp.ndarray (num_frames, num_positions)
  """error_dist = np.sqrt( #开平方
      r3.vecs_squared_distance(local_pred_pos, local_target_pos) #平方距离 X2+Y2+Z2
      + epsilon)
  """
  error_dist = (  # 开平方
      r3.vecs_squared_distance(local_pred_pos, local_target_pos)  # 平方距离 X2+Y2+Z2
      + epsilon).sqrt()
  #print("error_dist:", error_dist)
  if l1_clamp_distance:
    #error_dist = np.clip(error_dist, 0, l1_clamp_distance)
    error_dist = torch.clip(error_dist, 0, l1_clamp_distance)
  #print("error_dist:", error_dist)

  normed_error = error_dist / length_scale
  #print("normed_error:", normed_error)
  #expand_dims(a, axis)中，a为numpy数组，axis为需添加维度的轴,扩展数组形状
  #normed_error *= np.expand_dims(frames_mask, axis=-1)

  #normed_error = torch.matmul(frames_mask.unsqueeze(0),normed_error) # axis=-1，插入维度在最后 生成矩阵shape=(3,1)*(1,15)=(3,15)
  #normed_error = torch.matmul(positions_mask.unsqueeze(-2),normed_error)#normed_error.mul(positions_mask.unsqueeze(-2)) # 生成矩阵shape=(3,15) positions_mask.unsqueeze(-2)为(1,15)，两个向量中对应位相乘

  #normed_error *= np.expand_dims(positions_mask, axis=-2)

  """normalization_factor = (
      np.sum(frames_mask, axis=-1) *
      np.sum(positions_mask, axis=-1))#以最后维度进行相加"""
  normalization_factor = (
          frames_mask.sum(-1) *
          positions_mask.sum(-1))  # 以最后维度进行相加
  normalization_factor.requires_grad_(True)
  #print("normalization_factor:", normalization_factor.grad)
  """return (np.sum(normed_error, axis=(-2, -1)) /
          (epsilon + normalization_factor))"""
  #print(normed_error.shape)
  return normed_error.sum().div(epsilon + normalization_factor)

def get_identity_rigid(atom_num,ca,atom_position):
  #print(ca)
  """Returns identity rigid transform."""

  """ones = np.random.random((shape,1))
  zeros = np.zeros((shape,1))
  rot = r3.Rots(ones, ones, zeros,
                zeros, ones, zeros,
                zeros, zeros, ones)
  trans = r3.Vecs(zeros, zeros, zeros)"""
  #length = int(atom_num/10)
  xx,xy,xz,yx,yy,yz,zx,zy,zz=[],[],[],[],[],[],[],[],[]
  tranx,trany,tranz=[],[],[]
  for j in range(len(ca)):
      if ca[j].item()==1:
          a1 = torch.tensor([[atom_position[0][j-1]], [atom_position[1][j-1]], [atom_position[2][j-1]]],requires_grad=True)#N
          a2 = torch.tensor([[atom_position[0][j]], [atom_position[1][j]], [atom_position[2][j]]],requires_grad=True)#Ca
          a3 = torch.tensor([[atom_position[0][j + 1]], [atom_position[1][j + 1]], [atom_position[2][j + 1]]],requires_grad=True)#C
          vec1_a = r3.vecs_from_tensor(a1.t())#N
          vec2_a = r3.vecs_from_tensor(a2.t())#Ca
          vec3_a = r3.vecs_from_tensor(a3.t())#C
          rigid_a = r3.rigids_from_3_points(vec3_a, vec2_a, vec1_a)
          xx.append([rigid_a[0][0].item()])
          xy.append([rigid_a[0][1].item()])
          xz.append([rigid_a[0][2].item()])
          yx.append([rigid_a[0][3].item()])
          yy.append([rigid_a[0][4].item()])
          yz.append([rigid_a[0][5].item()])
          zx.append([rigid_a[0][6].item()])
          zy.append([rigid_a[0][7].item()])
          zz.append([rigid_a[0][8].item()])
          tranx.append([rigid_a[1][0].item()])
          trany.append([rigid_a[1][1].item()])
          tranz.append([rigid_a[1][2].item()])
  rot = r3.Rots(torch.tensor(xx,requires_grad=True), torch.tensor(xy,requires_grad=True), torch.tensor(xz,requires_grad=True),
                torch.tensor(yx,requires_grad=True), torch.tensor(yy,requires_grad=True), torch.tensor(yz,requires_grad=True),
                torch.tensor(zx,requires_grad=True), torch.tensor(zy,requires_grad=True), torch.tensor(zz,requires_grad=True))
  trans = r3.Vecs(torch.tensor(tranx,requires_grad=True), torch.tensor(trany,requires_grad=True), torch.tensor(tranz,requires_grad=True))
  return r3.Rigids(rot, trans)


def get_identity_vec(ca,atom_position):

  x,y,z=[],[],[]
  for j in range(len(ca)):
      if ca[j].item()==1:
          x.append(atom_position[0][j])
          y.append(atom_position[1][j])
          z.append(atom_position[2][j])
  vec = r3.Vecs(torch.tensor(x,requires_grad=True), torch.tensor(y,requires_grad=True), torch.tensor(z,requires_grad=True))
  return vec


def fape_loss(batch_size,ca,a,b,train=False):
    batch_loss = torch.tensor(0,dtype=float)
    if ca == None:
        ca = torch.zeros((len(b),150))
        for i in range(150):
            if i % 3 == 1:
                for j in range(len(b)):
                    if b[j][0][i]!=0:
                        ca[j][i] = 1
    #print(ca)
    for i in range(batch_size):
        vec_a = r3.vecs_from_tensor(a[i].t())
        vec_b = r3.vecs_from_tensor(b[i].t())
        if train :
            #vec_a = get_identity_vec(ca[i], a[i])
            #vec_b = get_identity_vec(ca[i], b[i])

            positions_mask = torch.ones(vec_b.x.shape[0])
            frames_mask = torch.ones(1)

            #rigid_a = get_identity_rigid(vec_a.x.shape[0],ca[i],a[i])
            #rigid_b = get_identity_rigid(vec_b.x.shape[0],ca[i],b[i])
            """
            #一个坐标转换
            a1 = torch.tensor([[a[i][0][0]], [a[i][1][0]], [a[i][2][0]]])  # N
            a2 = torch.tensor([[a[i][0][1]], [a[i][1][1]], [a[i][2][1]]])  # Ca
            a3 = torch.tensor([[a[i][0][2]], [a[i][1][2]], [a[i][2][2]]])  # C
            vec1_a = r3.vecs_from_tensor(a1.t())  # N
            vec2_a = r3.vecs_from_tensor(a2.t())  # Ca
            vec3_a = r3.vecs_from_tensor(a3.t())  # C
            rigid_a = r3.rigids_from_3_points(vec3_a, vec2_a, vec1_a)

            b1 = torch.tensor([[b[i][0][0]], [b[i][1][0]], [b[i][2][0]]])  # N
            b2 = torch.tensor([[b[i][0][1]], [b[i][1][1]], [b[i][2][1]]])  # Ca
            b3 = torch.tensor([[b[i][0][2]], [b[i][1][2]], [b[i][2][2]]])  # C
            vec1_b = r3.vecs_from_tensor(b1.t())  # N
            vec2_b = r3.vecs_from_tensor(b2.t())  # Ca
            vec3_b = r3.vecs_from_tensor(b3.t())  # C
            rigid_b = r3.rigids_from_3_points(vec3_b, vec2_b, vec1_b)
            """
            #以每个氨基酸的Ca作为原点进行坐标转换
            loss_pr = 0
            ca_num = 0
            for j in range(len(ca[i])):

                if ca[i][j].item() == 1:
                    if j+1 >= 400:
                        break
                    ca_num += 1
                    a1 = torch.tensor([[a[i][0][j - 1]], [a[i][1][j - 1]], [a[i][2][j - 1]]])  # N
                    a2 = torch.tensor([[a[i][0][j]], [a[i][1][j]], [a[i][2][j]]])  # Ca
                    a3 = torch.tensor([[a[i][0][j + 1]], [a[i][1][j + 1]], [a[i][2][j + 1]]])  # C
                    vec1_a = r3.vecs_from_tensor(a1.t())  # N
                    vec2_a = r3.vecs_from_tensor(a2.t())  # Ca
                    vec3_a = r3.vecs_from_tensor(a3.t())  # C
                    rigid_a = r3.rigids_from_3_points(vec3_a, vec2_a, vec1_a)

                    b1 = torch.tensor([[b[i][0][j-1]], [b[i][1][j-1]], [b[i][2][j-1]]])  # N
                    b2 = torch.tensor([[b[i][0][j]], [b[i][1][j]], [b[i][2][j]]])  # Ca
                    b3 = torch.tensor([[b[i][0][j+1]], [b[i][1][j+1]], [b[i][2][j+1]]])  # C
                    vec1_b = r3.vecs_from_tensor(b1.t())  # N
                    vec2_b = r3.vecs_from_tensor(b2.t())  # Ca
                    vec3_b = r3.vecs_from_tensor(b3.t())  # C
                    rigid_b = r3.rigids_from_3_points(vec3_b, vec2_b, vec1_b)
                    loss_aa = frame_aligned_point_error(rigid_a,rigid_b,frames_mask,vec_a,vec_b,positions_mask,10,10)
                    loss_pr += loss_aa
            #print(loss_pr)
            batch_loss += loss_pr

        """print(r3.invert_rigids(rigid_a)[0])
        for i in r3.invert_rigids(rigid_a)[0]:
            print(i[:, None],i[:, None].shape)   # 增加一个维度，且都赋予none
        print(r3.invert_rigids(rigid_a))
        for j in range(len(vec_a)):
            print(vec_a[j][None, :],vec_a[j][None, :].shape)
            y = vec_a[j].unsqueeze(0)
            print(y,y.shape)
          return Vecs(m.xx * v.x + m.xy * v.y + m.xz * v.z,
              m.yx * v.x + m.yy * v.y + m.yz * v.z,
              m.zx * v.x + m.zy * v.y + m.zz * v.z)
        vx=torch.tensor([[1,2,5,1,1,1,8]])
        vy = torch.tensor([[4, 2, 3, 2, 5, 1,5 ]])
        vz = torch.tensor([[6, 1, 2, 2, 1, 9, 3]])
        mxx = torch.tensor([[2]])
        mxy = torch.tensor([[5]])
        mxz = torch.tensor([[4]])
        myx = torch.tensor([[5]])
        myy = torch.tensor([[1]])
        myz = torch.tensor([[8]])
        mzx = torch.tensor([[4]])
        mzy = torch.tensor([[6]])
        mzz = torch.tensor([[3]])
        print("**********",mxx * vx + mxy * vy + mxz * vz)#(1,15)与(15)的区别
        
        print("原始pred_pos",vec_a)
        local_pred_pos = r3.rigids_mul_vecs(r3.invert_rigids(rigid_a), vec_a)#矩阵相乘，获得变换后坐标
        print(rigid_a)
        #print(r3.invert_rigids(rigid_a))
        print(r3.invert_rigids(r3.invert_rigids(rigid_a)))
        #trans = r3.Vecs(rigid_a.trans.x,rigid_a.trans.x,rigid_a.trans.x
        #print('变换后的pred_pos:',local_pred_pos)
        local_pred_pos_2 = r3.rigids_mul_vecs(rigid_a, local_pred_pos)
        print('再变换后的pred_pos:', local_pred_pos_2)
       
        # Compute array of target positions in the target frames.
        # r3.Vecs (num_frames, num_positions)
        print("原始target_pos",vec_b)
        local_target_pos = r3.rigids_mul_vecs(r3.invert_rigids(rigid_b), vec_b)
        print('变换后的traget_pos:', local_target_pos)
        """

        """loss = frame_aligned_point_error(
                rigid_a, #pred_frames: r3.Rigids,  # shape (num_frames)
                rigid_b, #target_frames: r3.Rigids,  # shape (num_frames)
                frames_mask,#frames_mask: np.ndarray,  # shape (num_frames)
                vec_a, #pred_positions: r3.Vecs,  # shape (num_positions)
                vec_b, #target_positions: r3.Vecs,  # shape (num_positions)
                positions_mask,#positions_mask: np.ndarray,  # shape (num_positions)
                10)#length_scale: float"""
    #print(batch_loss)
    #batch_loss = torch.tensor((len(ca)/7)**3)
    return batch_loss/batch_size
if __name__ == "__main__":
    a = torch.tensor([[[3.3053e-01, 4.6007e-01, 6.8907e-01, 7.6416e-01, 8.4817e-01,
                        8.9059e-01, 9.1378e-01, 9.4728e-01, 9.8095e-01, 1.0233e+00,
                        1.0697e+00, 1.0847e+00, 1.0531e+00, 9.4087e-01, 8.1212e-01],
                       [-2.9372e-01, -4.9709e-01, -6.7954e-01, -6.9637e-01, -6.0741e-01,
                        -6.3215e-01, -6.3444e-01, -6.1823e-01, -6.4622e-01, -6.2805e-01,
                        -6.7054e-01, -6.5053e-01, -6.1740e-01, -5.7762e-01, -5.7591e-01],
                       [-1.7535e-01, -3.4360e-01, -4.7266e-01, -6.2106e-01, -6.2967e-01,
                        -6.5871e-01, -6.8629e-01, -6.9996e-01, -7.4000e-01, -8.1811e-01,
                        -8.6925e-01, -9.0646e-01, -8.2718e-01, -7.3721e-01, -6.0934e-01, ]],
                      [[-4.1974e-01, -5.7846e-01, -9.0139e-01, -1.0635e+00, -1.2009e+00,
                        -1.1995e+00, -1.1691e+00, -1.1174e+00, -1.0437e+00, -9.7729e-01,
                        -8.9452e-01, -8.2151e-01, -7.5108e-01, -6.7814e-01, -6.0210e-01],
                       [-1.2507e-01, -5.0884e-03, -7.8833e-03, -7.9285e-02, -1.4190e-01,
                        -2.7916e-01, -3.9597e-01, -4.5072e-01, -5.2041e-01, -5.4038e-01,
                        -5.4244e-01, -5.5686e-01, -5.2920e-01, -5.3430e-01, -4.8865e-01],
                       [5.9738e-02, 3.4467e-01, 6.2221e-01, 9.5078e-01, 1.0566e+00,
                        1.1560e+00, 1.1337e+00, 1.1120e+00, 1.0670e+00, 9.9290e-01,
                        9.5200e-01, 8.8840e-01, 8.2825e-01, 7.5250e-01, 6.4652e-01]]
                      ])
    b = torch.tensor([[[-4.1974e-01, -5.7846e-01, -9.0139e-01, -1.0635e+00, -1.2009e+00,
                        -1.1995e+00, -1.1691e+00, -1.1174e+00, -1.0437e+00, -9.7729e-01,
                        -8.9452e-01, -8.2151e-01, -7.5108e-01, -6.7814e-01, -6.0210e-01],
                       [-1.2507e-01, -5.0884e-03, -7.8833e-03, -7.9285e-02, -1.4190e-01,
                        -2.7916e-01, -3.9597e-01, -4.5072e-01, -5.2041e-01, -5.4038e-01,
                        -5.4244e-01, -5.5686e-01, -5.2920e-01, -5.3430e-01, -4.8865e-01],
                       [5.9738e-02, 3.4467e-01, 6.2221e-01, 9.5078e-01, 1.0566e+00,
                        1.1560e+00, 1.1337e+00, 1.1120e+00, 1.0670e+00, 9.9290e-01,
                        9.5200e-01, 8.8840e-01, 8.2825e-01, 7.5250e-01, 6.4652e-01]],
                      [[3.3053e-01, 4.6007e-01, 6.8907e-01, 7.6416e-01, 8.4817e-01,
                        8.9059e-01, 9.1378e-01, 9.4728e-01, 9.8095e-01, 1.0233e+00,
                        1.0697e+00, 1.0847e+00, 1.0531e+00, 9.4087e-01, 8.1212e-01],
                       [-2.9372e-01, -4.9709e-01, -6.7954e-01, -6.9637e-01, -6.0741e-01,
                        -6.3215e-01, -6.3444e-01, -6.1823e-01, -6.4622e-01, -6.2805e-01,
                        -6.7054e-01, -6.5053e-01, -6.1740e-01, -5.7762e-01, -5.7591e-01],
                       [-1.7535e-01, -3.4360e-01, -4.7266e-01, -6.2106e-01, -6.2967e-01,
                        -6.5871e-01, -6.8629e-01, -6.9996e-01, -7.4000e-01, -8.1811e-01,
                        -8.6925e-01, -9.0646e-01, -8.2718e-01, -7.3721e-01, -6.0934e-01, ]]])
    batch_size = 2
    ca = [[0, 0, 0, 1, 0,0, 0, 0, 1, 0,0, 0, 0, 1, 0],[0, 0, 0, 1, 0,0, 0, 0, 1, 0,0, 0, 0, 1, 0]]
    fape_loss(batch_size,ca,a,b,train=True)
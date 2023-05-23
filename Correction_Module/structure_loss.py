import residue_constants
import torch


def distance(c, n, epsilon =1e-6):
    d = torch.zeros(len(c))
    for i in range(len(c)):
        d[i] = torch.sqrt((c[i][0] - n[i][0])**2 + (c[i][1] - n[i][1])**2 + (c[i][2] - n[i][2])**2)
    return d
def get_cos(pos1,pos2,pos3):
    c = torch.zeros(len(pos1))
    d1_2 = distance(pos1, pos2)
    d2_3 = distance(pos2, pos3)
    for i in range(len(pos1)):
        b1_2 = pos2[i]-pos1[i]
        b2_3 = pos2[i] - pos3[i]
        c[i] = (b1_2[0]*b2_3[0] + b1_2[1]*b2_3[1] + b1_2[2]*b2_3[2])/ d1_2[i] / d2_3[i]
    return c
def between_residue_bond_loss(
    pred_atom_positions, ca_n_c, tolerance_factor_soft=12.0):
    """Flat-bottom loss to penalize structural violations between residues."""
    ca,n,c = [],[],[]
    pro = []
    for i in range(len(ca_n_c)):
        if ca_n_c[i] == 1:
            ca.append(i)
        elif ca_n_c[i] == 2:
            n.append(i)
            pro.append(0)
        elif ca_n_c[i] == 22:
            n.append(i)
            pro.append(1)
        elif ca_n_c[i] == 3:
            c.append(i)
    aa_num = len(ca)
    this_ca = torch.zeros([len(pred_atom_positions[0]), len(ca) - 1])
    this_c = torch.zeros([len(pred_atom_positions[0]), len(ca) - 1])
    next_n = torch.zeros([len(pred_atom_positions[0]), len(ca) - 1])
    next_ca = torch.zeros([len(pred_atom_positions[0]), len(ca) - 1])
    k=0
    for i in ca[0:len(ca)-1]:
        this_ca[i][k]=1
        k = k+1
    k = 0
    for i in c[0:len(c)-1]:
        this_c[i][k] = 1
        k = k + 1
    k = 0
    for i in n[1:len(n)]:
        next_n[i][k] = 1
        k = k + 1
    k = 0
    for i in ca[1:len(ca)]:
        next_ca[i][k] = 1
        k = k + 1
    this_ca_pos = torch.matmul(pred_atom_positions , this_ca).t()
    this_c_pos = torch.matmul(pred_atom_positions, this_c).t()
    next_n_pos = torch.matmul(pred_atom_positions, next_n).t()
    next_ca_pos = torch.matmul(pred_atom_positions, next_ca).t()
    c_n_bond_length = distance(this_c_pos,next_n_pos, 1e-6) # N-1 this_c  next_n 肽键长度
    next_is_proline = torch.tensor(pro[1:])
    gt_length = (
            (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
            + next_is_proline * residue_constants.between_res_bond_length_c_n[1])#PRO
    gt_stddev = (
            (1. - next_is_proline) * residue_constants.between_res_bond_length_stddev_c_n[0] +
            next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = torch.sqrt(1e-6 + torch.square(c_n_bond_length - gt_length))
    relu = torch.nn.ReLU()
    c_n_loss_per_residue = relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    c_n_loss = torch.sum(c_n_loss_per_residue / aa_num + 1e-6)

    # Compute loss for the angles.
    ca_c_n_cos_angle = get_cos(this_ca_pos,this_c_pos,next_n_pos)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_cos_angles_ca_c_n[1]
    ca_c_n_cos_angle_error = torch.sqrt(1e-6 + torch.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    ca_c_n_loss = torch.sum(ca_c_n_loss_per_residue) / (len(ca_c_n_cos_angle) + 1e-6)

    c_n_ca_cos_angle = get_cos(this_c_pos,next_n_pos,next_ca_pos)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(1e-6 + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    c_n_ca_loss = torch.sum(c_n_ca_loss_per_residue) / (len(c_n_ca_cos_angle) + 1e-6)

    return c_n_loss,ca_c_n_loss,c_n_ca_loss


def loss4(pred_atom_positions, ca_n_c,chain1_D=None,chain2_B=None,chain3_C=None):
    batch_loss = 0
    for i in range(len(pred_atom_positions)):
        if chain1_D == None:
            len_loss,angle_loss1,angle_loss2 = between_residue_bond_loss(pred_atom_positions[i], ca_n_c[i])
            len_loss.requires_grad_(True)
            angle_loss1.requires_grad_(True)
            angle_loss2.requires_grad_(True)
            batch_loss = batch_loss + angle_loss1 + angle_loss2 + len_loss
        else:
            chain1_len_loss, chain1_angle_loss1, chain1_angle_loss2 = between_residue_bond_loss(pred_atom_positions[i],
                                                                                                ca_n_c[i]*chain1_D[i])
            chain2_len_loss, chain2_angle_loss1, chain2_angle_loss2 = between_residue_bond_loss(pred_atom_positions[i],
                                                                                                ca_n_c[i] * chain2_B[i])
            chain3_len_loss, chain3_angle_loss1, chain3_angle_loss2 = between_residue_bond_loss(pred_atom_positions[i],
                                                                                                ca_n_c[i] * chain3_C[i])
            len_loss = chain3_len_loss+chain1_len_loss+chain2_len_loss
            angle_loss1 = chain1_angle_loss1+chain2_angle_loss1+chain3_angle_loss1
            angle_loss2 = chain1_angle_loss2+chain2_angle_loss2+chain3_angle_loss2
            len_loss.requires_grad_(True)
            angle_loss1.requires_grad_(True)
            angle_loss2.requires_grad_(True)
            batch_loss = batch_loss + angle_loss1 + angle_loss2+ len_loss#
    return batch_loss/len(pred_atom_positions)
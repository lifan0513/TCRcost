#!/usr/bin/python
'''
calculates RMSD differences between 2 conformation with different atom names.
@author: JC <yangjincai@nibs.ac.cn>
'''
import os
import sys
import math

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import AlignMol


def GetBestRMSD(probe, ref, refConfId=-1, probeConfId=-1, maps=None):
    """ Returns the optimal RMS for aligning two molecules, taking
    symmetry into account. As a side-effect, the probe molecule is
    left in the aligned state.
    Arguments:
      - ref: the reference molecule
      - probe: the molecule to be aligned to the reference
      - refConfId: (optional) reference conformation to use
      - probeConfId: (optional) probe conformation to use
      - maps: (optional) a list of lists of (probeAtomId,refAtomId)
        tuples with the atom-atom mappings of the two molecules.
        If not provided, these will be generated using a substructure
        search.
    Note:
    This function will attempt to align all permutations of matching atom
    orders in both molecules, for some molecules it will lead to 'combinatorial
    explosion' especially if hydrogens are present.
    Use 'rdkit.Chem.AllChem.AlignMol' to align molecules without changing the
    atom order.
    """
    # When mapping the coordinate of probe will changed!!!
    ref.pos = orginXYZ(ref)
    probe.pos = orginXYZ(probe)

    if not maps:
        matches = ref.GetSubstructMatches(probe, uniquify=True)
        if not matches:
            raise ValueError('mol %s does not match mol %s' % (ref.GetProp('_Name'),
                                                               probe.GetProp('_Name')))
        maps = [list(enumerate(match)) for match in matches]
    bestRMS = 1000.0
    bestRMSD = 1000.0
    #print("*******",maps)
    for amap in maps:
        rms = AlignMol(probe, ref, probeConfId, refConfId, atomMap=amap)#AlignMol在不改变原子顺序的情况下对齐分子
        rmsd = RMSD(probe, ref, amap)
        if rmsd < bestRMSD:
            bestRMSD = rmsd
        if rms < bestRMS:
            bestRMS = rms
            bestMap = amap

    # finally repeate the best alignment :
    if bestMap != amap:
        AlignMol(probe, ref, probeConfId, refConfId, atomMap=bestMap)

    return bestRMS, bestRMSD


# Map is probe -> ref
# [(1:3),(2:5),...,(10,1)]
def RMSD(probe, ref, amap):
    rmsd = 0.0
    # print(amap)
    atomNum = ref.GetNumAtoms() + 0.0
    #print(atomNum)
    for (pi, ri) in amap:
        posp = probe.pos[pi]
        posf = ref.pos[ri]

        rmsd += dist_2(posp, posf)
    rmsd = math.sqrt(rmsd / atomNum)
    return rmsd


def dist_2(atoma_xyz, atomb_xyz):
    dis2 = 0.0
    for i, j in zip(atoma_xyz, atomb_xyz):
        dis2 += (i - j) ** 2
    return dis2


def orginXYZ(mol):
    mol_pos = {}
    for i in range(0, mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        mol_pos[i] = pos
    return mol_pos


def print_rmsd(ref_path = "C:/Users/lifan/Desktop/1DCNN_test_true", probe_path = "C:/Users/lifan/Desktop/model/lddt"):
    """isoRMSD.py will output two RMSD, one is fitted, another is no fit.
     Not fit RMSD mean no change in molecules coordinates.

     Usage:python isoRMSD.py mol1.pdb mol2.pdb rmsd.txt
    """
    """if len(sys.argv) < 4:
        print(usage)
        sys.exit()"""
    file_path=probe_path
    sum_rms = 0
    files = os.listdir(file_path)
    for file in files:
        if file == '4P2Q.pdb' or file == '4QRP.pdb':
            pass
        else:
            ref = Chem.MolFromPDBFile(ref_path + "/" + file,proximityBonding=False)
            probe = Chem.MolFromPDBFile(probe_path + "/" + file,proximityBonding=False)

            # here, rms is Fitted, rmsd is NOT Fit!!!
            map1 = [[]]
            for i in range(len(open(file_path + "/" + file, 'r').readlines())):
                map1[0].append((i, i))
            rms, rmsd = GetBestRMSD(probe, ref,maps=map1)
            sum_rms = sum_rms + rms
            #print(file,end='\t')
            #print("Best_RMSD: %.3f\tBest_Not_Fit_RMSD: %.3f" % (rms, rmsd))
    return sum_rms/len(files)


if __name__ == "__main__":

    alphafold_files = "E:/paper_data/5-fold-structure-data/all_af"
    true_files = "E:/paper_data/5-fold-structure-data/all_true"
    files = os.listdir(alphafold_files)
    for filename in files:
        print(filename, end="\t")
        #filetype, name = filename.split('_')
        # CDR3A
        name = filename
        if name not in ['4P2Q.pdb', '4QRP.pdb']:
            Aprobe = Chem.MolFromPDBFile(alphafold_files +'/'+ name, proximityBonding=False)
            Aref = Chem.MolFromPDBFile(true_files +'/'+ name, proximityBonding=False)
            Arms, Armsd = GetBestRMSD(Aprobe, Aref)
            print('CDR3A', Arms, Armsd, end="\t")






from rdkit.Chem import rdmolfiles
import rdkit
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import os

aminoAcids = ['ALA', 'VAL', 'ASP', 'LYS', 'ASN', 'LEU', 'ILE', 'CYS', 'THR', 'PHE',
              'GLY', 'TRP', 'ARG', 'TYR', 'GLN', 'PRO', 'HIS', 'MET', 'GLU', 'SER',
              ' MG', ' ZN', ' MN', 'LLP', 'KCX', ' CO', ' CA', ' NA', 'MSE'] # 29

atomicNums =  [5,6,7,8,15,16,34]
halogens = [9,17,35,53]
metals = [3, 4, 11, 12, 13] 
metals += list(range(19, 32))
metals += list(range(37, 51)) 
metals += list(range(55, 84))
metals += list(range(87, 104))

hyb_types = [rdkit.Chem.rdchem.HybridizationType.S,
             rdkit.Chem.rdchem.HybridizationType.SP,
            rdkit.Chem.rdchem.HybridizationType.SP2,
            rdkit.Chem.rdchem.HybridizationType.SP3,
            rdkit.Chem.rdchem.HybridizationType.SP3D,
            rdkit.Chem.rdchem.HybridizationType.SP3D2,
            rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED]

possible_numH_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
possible_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
charges = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
valences = list(range(0,8))
possible_chirality_list = [
        rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        rdkit.Chem.rdchem.ChiralType.CHI_OTHER
    ]

bond_types = [rdkit.Chem.rdchem.BondType.SINGLE,
            rdkit.Chem.rdchem.BondType.DOUBLE,
            rdkit.Chem.rdchem.BondType.TRIPLE,
            rdkit.Chem.rdchem.BondType.AROMATIC,
            rdkit.Chem.rdchem.BondType.UNSPECIFIED,
            rdkit.Chem.rdchem.BondType.ZERO,
            rdkit.Chem.rdchem.BondType.OTHER]

bond_dirs = [rdkit.Chem.rdchem.BondDir.NONE,
            rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
            rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
            rdkit.Chem.rdchem.BondDir.EITHERDOUBLE,
            rdkit.Chem.rdchem.BondDir.UNKNOWN]



def parseIndexFile(indexFilePath):
    with open(indexFilePath, "r") as index_file:
            pdbIDs = []
            logKvalues = {}
            for line in index_file:
                if not line.startswith('#'):
                    pdbIDs.append(str(line.split()[0]))
                    logKvalues[str(line.split()[0])] = float(line.split()[3])
    return pdbIDs, logKvalues

                 

def distance(mol1,mol2,ind1,ind2):
    i_coords = mol1.GetConformer().GetAtomPosition(ind1)
    j_coords = mol2.GetConformer().GetAtomPosition(ind2)
    i_coords = np.array([i_coords[0],i_coords[1],i_coords[2]])
    j_coords = np.array([j_coords[0],j_coords[1],j_coords[2]])
    dist = np.floor(np.linalg.norm(i_coords - j_coords))
    return dist 


def calcNearest(ligand,protein):
    nearest= {}
    for i in range(len(ligand.GetAtoms())):
        distances = []
        for j in range(len(protein.GetAtoms())):
            if protein.GetAtomWithIdx(j).GetPDBResidueInfo().GetResidueName() == 'HOH':
                continue
            dist = distance(ligand,protein,i,j)
            if dist <= 5:
                distances.append((j, dist))
        nearest[i] = sorted(distances, key=lambda x: x[1])
    return nearest

def atomicOneHot(atom):
    num = atom.GetAtomicNum()
    if num in atomicNums:
        return [atomicNums.index(num)]
    elif num in halogens:
        return [len(atomicNums)]
    elif num in metals:
        return [len(atomicNums)+1]
    else:
        return [len(atomicNums)+2] 

def hybridizationOneHot(type):
    if type in hyb_types:
        return [hyb_types.index(type)]
    else:
        return [len(hyb_types)]

def formalChargeOneHot(charge):
    if charge in charges:
        return [charges.index(charge)]
    else:
        return [len(charges)]
    
def getDegree(atom):
    try:
        degree = possible_degree_list.index(atom.GetTotalDegree())
    except:
        degree = 11
    return [degree]

def explicitValenceOneHot(valence):
    if valence in valences:
        return [valences.index(valence)]
    else:
        return [len(valences)]
    
def getNumH(atom):
    try:
        numH = possible_numH_list.index(atom.GetTotalNumHs())
    except:
        numH = 9
    return [numH]

def getChirality(atom):
    try:
        chirality = possible_chirality_list.index(atom.GetChiralTag())
    except:
        chirality = 0
    return [chirality]

def aromaticOneHot(aromatic):
    a = int(aromatic)
    return [a]


def atomFeatures(atom, is_protein):
    atom_features = atomicOneHot(atom)
    atom_features += getChirality(atom)
    atom_features += formalChargeOneHot(atom.GetFormalCharge())
    atom_features += hybridizationOneHot(atom.GetHybridization())
    atom_features += getNumH(atom)
    atom_features += explicitValenceOneHot(atom.GetTotalValence()) 
    atom_features += getDegree(atom)
    atom_features += aromaticOneHot(atom.GetIsAromatic())
    atom_features += [int(is_protein)]
    atom_features += [atom.GetMass()/100]
    return atom_features

def bondType(bond):  
    try:
        bond_type = [bond_types.index(bond.GetBondType())]
        return bond_type
    except:
        bond_type = [6]
        return bond_type
    
def getStereo(bond):
    if bond is None:
        return [0,0]
    stereo = int(bond.GetStereo())
    inRing = int(bond.IsInRing())
    return [stereo,inRing]

def getBondDir(bond):
    try:
        dir = bond_dirs.index(bond.GetBondDir())
        return [dir]
    except:
        return [4]
    
def bondFeatures(dist, bond, inner = False):
    bond_feature = [dist]
    bond_feature += bondType(bond)
    bond_feature += getBondDir(bond)
    bond_feature += getStereo(bond)
    bond_feature += [int(inner)]
    return bond_feature


def residueOneHot(atom):
    if atom is None:
        return [len(aminoAcids)]
    residue_info = atom.GetPDBResidueInfo()
    res_name = residue_info.GetResidueName()
    if res_name not in aminoAcids:
        return [len(aminoAcids)+1]
    else:
        return [aminoAcids.index(res_name)]
        
   
def createGraph(ligand,protein):
    graph = nx.Graph()
    nearest = calcNearest(ligand,protein)
    ligand_size = len(ligand.GetAtoms())
    proteinNodes= []
    bonds = []

    for i in range(ligand_size):
        for p_atom in nearest[i]:
            if p_atom[0] not in proteinNodes:
                proteinNodes.append(p_atom[0])
            bonds.append((i,ligand_size + proteinNodes.index(p_atom[0]),p_atom[1]))

    for i,atom in enumerate(ligand.GetAtoms()):
        feature = atomFeatures(atom,is_protein = False) + residueOneHot(None)
        graph.add_node(i,feature=feature) 



    for i,oldIdx in enumerate(proteinNodes):
        atom = protein.GetAtomWithIdx(oldIdx)
        feature= atomFeatures(atom, is_protein = True) + residueOneHot(atom)
        graph.add_node(i + ligand_size, feature=feature)


    for bond in ligand.GetBonds():
        dist = distance(ligand,ligand,bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
        feature = bondFeatures(dist, bond, inner=True)
        graph.add_edge(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),feature = feature)
        graph.add_edge(bond.GetEndAtomIdx(),bond.GetBeginAtomIdx(),feature = feature)

    for bond in bonds:
        feature = bondFeatures(dist = bond[2],bond=None,inner=False)
        graph.add_edge(bond[0],bond[1], feature = feature)
        graph.add_edge(bond[1],bond[0], feature = feature)
    return graph
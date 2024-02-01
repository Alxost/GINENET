from oddt import toolkits
from oddt.fingerprints import PLEC
from rdkit.Chem import rdmolfiles
import numpy as np
import os


dataDir = "Sirt2"   #directory with ligand files in .mol format and protein file in .pdb format
output_big = "sirt2_plecs_big" # output directory for plecs with size 65536
output_small = "sirt2_plecs_small" # output directory for plecs with size 16384

def createIndex(dataDir):
    ligandIndex = []
    for file in os.listdir(dataDir):
        if file.endswith(".pdb"):
            proteinFile = file
        elif file.endswith(".mol"):
            ligandIndex.append(file[:-4]) 
    return ligandIndex,proteinFile

def generatePLECS(ligandIndex, proteinFile):
    for name in ligandIndex:
        ligand  = rdmolfiles.MolFromMolFile(f"{dataDir}/{name}", removeHs=True)
        proteinPocket  = rdmolfiles.MolFromPDBFile(f"{dataDir}/{proteinFile}", removeHs=True)
        if proteinPocket is None and ligand is None:
                continue
        if proteinPocket is None:
                continue
        if ligand is None:
                continue
        else:
            try:    
                lig = toolkits.rdk.Molecule(ligand)
                prot= toolkits.rdk.Molecule(proteinPocket)
                small_plec = PLEC(lig, prot, sparse = False, size = 16384, depth_ligand=1, depth_protein=5)
                big_plec = PLEC(lig, prot, sparse = False, size = 65536, depth_ligand=1, depth_protein=5)
                np.save(f"{output_small}/{name}.npy", np.array(small_plec))
                np.save(f"{output_big}/{name}.npy", np.array(big_plec))
            except:
                continue


ligandIndex, proteinFile = createIndex(dataDir)
generatePLECS(ligandIndex,proteinFile)
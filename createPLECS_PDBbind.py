
import os
import logging
from oddt import toolkits
from oddt.fingerprints import PLEC
from rdkit import rdBase
from rdkit.Chem import rdmolfiles
import numpy as np
from multiprocessing import Pool


logging.basicConfig(filename='log.txt', encoding='utf-8', level=logging.DEBUG)
rdBase.EnableLog("rdApp.debug")
rdBase.EnableLog("rdApp.info")
rdBase.LogToPythonLogger()

logger = logging.getLogger("rdkit")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("generate_PLECS_log.txt")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


indexFullSet = "INDEX_general_PL_data.2020"
fullSetPath = "PDBbind_all"
small_plecs_path = "PLECS_16384"
big_plecs_path = "PLECS_65536"

def parseIndexFile(indexFilePath):
    with open(indexFilePath, "r") as index_file:
            pdbIDs = []
            logKvalues = {}
            for line in index_file:
                if not line.startswith('#'):
                    pdbIDs.append(str(line.split()[0]))
                    logKvalues[str(line.split()[0])] = float(line.split()[3])
    return pdbIDs, logKvalues



index, logK = parseIndexFile(indexFullSet)



def generatePLECS(index):
    for i,pdbID in enumerate(index):
        logger.info(f"Logging output for complex: {pdbID}")
        ligand  = rdmolfiles.MolFromMol2File(f"{fullSetPath}/{pdbID}/{pdbID}_ligand.mol2",removeHs=True)
        proteinPocket  = rdmolfiles.MolFromPDBFile(f"{fullSetPath}/{pdbID}/{pdbID}_pocket.pdb",removeHs=True)
        if proteinPocket is None and ligand is None:
            logger.info(f"Could not read protein file and ligand file for {pdbID}")
            continue
        if proteinPocket is None:
            logger.info(f"Could not read protein file for {pdbID} ")
            continue
        if ligand is None:
            logger.info(f"Could not read ligand file for {pdbID}")
            continue

        else:
            try:    
                lig = toolkits.rdk.Molecule(ligand)
                prot= toolkits.rdk.Molecule(proteinPocket)
                small_plec = PLEC(lig, prot, sparse = False, size = 16384, depth_ligand=1, depth_protein=5)
                big_plec = PLEC(lig, prot, sparse = False, size = 65536, depth_ligand=1, depth_protein=5)
                np.save(f"{small_plecs_path}/{pdbID}.npy", np.array(small_plec))
                np.save(f"{big_plecs_path}/{pdbID}.npy", np.array(big_plec))
            except:
                logger.info(f"Could not create PLEC for {pdbID}")
                continue


index = np.array(index)
split = np.array_split(index,3)

with Pool(3) as p:
    p.map(generatePLECS,split)

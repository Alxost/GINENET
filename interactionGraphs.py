from rdkit.Chem import rdmolfiles
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import os
from graphUtils import createGraph,convertNetworkxToPytorch





def createGraphs(dataDir, proteinPocket, outputPath):
    """Create Interaction Graph between protein pocket and every ligand(.mol format) in the dataDir directory. 
    The graphs are saved in "outputPath" directory."""
    for file in os.listdir(dataDir):
        if not file.endswith(".mol2"):
            continue
        ligand  = rdmolfiles.MolFromMol2File(f"{dataDir}/{file}",removeHs=True)
        if proteinPocket is None and ligand is None:
                continue
        if proteinPocket is None:
                continue
        if ligand is None:
                continue
        else:
            graph = createGraph(ligand,proteinPocket)
            graph_pt = convertNetworkxToPytorch(graph)
            with open(f"{outputPath}/{file[:-4]}_graph.pt","wb") as file:
                torch.save(graph_pt,file)


      
proteinPath = "" #path to protein .pdb file
dataDir = ""                      #path to protein .pdb file
outputPath = ""

proteinPocket = rdmolfiles.MolFromPDBFile(proteinPath, removeHs = True)
createGraphs(dataDir,proteinPocket, outputPath)

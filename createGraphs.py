from rdkit.Chem import rdmolfiles
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import os
from graphUtils import createGraph



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
            node_features = nx.get_node_attributes(graph,"feature")
            edges = graph.edges
            edges = torch.tensor(np.array(graph.edges).T, dtype=torch.long)
            edge_features = nx.get_edge_attributes(graph,"feature")
            node_features = torch.tensor(np.array(list(node_features.values())), dtype=torch.float)
            edge_features = torch.tensor(np.array(list(edge_features.values())), dtype=torch.float)
            node_features = node_features.reshape((len(node_features),11))
            edge_features = edge_features.reshape(len(edge_features),6)
            graph_pt = Data(node_features,edges,edge_features)
            with open(f"{outputPath}/{file[:-4]}_graph.pt","wb") as file:
                torch.save(graph_pt,file)

proteinPath = "/home/alex/Documents/Projekte/4ezx/4ezx_pocket.pdb" #change according to data set
dataDir = "/home/alex/Documents/Projekte/4ezx"
outputPath = "/home/alex/Documents/Projekte/test_output"

proteinPocket = rdmolfiles.MolFromPDBFile(proteinPath, removeHs = True)
createGraphs(dataDir,proteinPocket, outputPath)
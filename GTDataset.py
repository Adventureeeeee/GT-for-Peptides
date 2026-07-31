import dgl
import pandas as pd
import torch
import numpy as np

from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset


device = torch.device('cuda')
print(f'use {device}')

class GTDataset(Dataset):


    def __init__(self, dataset='', dataset_pre='',protein_graph=None, protein_id=None,label=None):

        self.dataset = dataset
        self.dataset_pre = dataset_pre
        self.protein_graph, _ = load_graphs(protein_graph)
        self.protein_graph = list(self.protein_graph)



        self.protein_id = np.load(protein_id, allow_pickle=True)
        if label is not None:
            self.label = np.load(label, allow_pickle=True)
        else:
            self.label = None

    def __len__(self):
       
        return len(self.label) if self.label is not None else len(self.protein_graph)




    def __getitem__(self, idx):

        protein_len = self.protein_graph[idx].num_nodes()

        label = self.label[idx] if self.label is not None else None
       

        return self.protein_graph[idx], protein_len, label
       





    def collate(self, sample):
       
        protein_graph,  protein_len,  label= map(list, zip(*sample))
       

        protein_graph = dgl.batch(protein_graph).to(device)

      
        if label[0] is not None:
            labels = torch.FloatTensor(label).to(device)
        else:
            labels = None
        return protein_graph, labels




       

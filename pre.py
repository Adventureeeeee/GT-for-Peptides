import dgl
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from dgl.data.utils import save_graphs

from scipy import sparse as sp
from itertools import permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings



CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}




def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=168):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]




def atom_features(atom, explicit_H=False, use_chirality=True):
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  
    degree = [0, 1, 2, 3, 4, 5, 6]  
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  
    return results


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    g = dgl.graph([])



    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    g.ndata["atom"] = torch.tensor(atom_feats)

    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g.add_edges(src_list, dst_list)

    g.edata["bond"] = torch.tensor(np.array(bond_feats_all))

    return g


def protein_graph_construction(id, protein_values, dir_output):
    N = len(protein_values)
    for no, data in enumerate(id):
        proteins_g = list()
        print('/'.join(map(str, [no + 1, N])))
        smiles_data = protein_values[no]
        protein_graph = smiles_to_graph(smiles_data)
        proteins_g.append(protein_graph)
        dgl.save_graphs(dir_output + '/protein_graph/' + str(data) + '.bin', list(proteins_g))


def protein_graph_process(dataset, fold, dir_output, id_train, id_test):
    proteins_graph_train, proteins_graph_test = [], []
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        protein_graph_train, _ = load_graphs( '.bin')

        proteins_graph_train.append(protein_graph_train[0])
    print(len(proteins_graph_train))
    dgl.save_graphs(dir_output + '/train/fold/' + str(fold) + '/protein_graph.bin', proteins_graph_train)

    N = len(id_test)
    for no, id in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        protein_graph_test, _ = load_graphs('.bin')
        proteins_graph_test.append(protein_graph_test[0])
    print(len(proteins_graph_test))
    dgl.save_graphs(dir_output + '/test/fold/' + str(fold) + '/protein_graph.bin', proteins_graph_test)



def protein_id_process(dataset, fold, dir_output, id_train, id_test):
    proteins_id_train, proteins_id_test = [], []
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        proteins_id_train.append(id)
    np.save(dir_output + '/train/fold/' + str(fold) + '/protein_id.npy', proteins_id_train)

    N = len(id_test)
    for no, id in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        proteins_id_test.append(id)
    np.save(dir_output + '/test/fold/' + str(fold) +'/protein_id.npy', proteins_id_test)


def Label_process(dataset, fold, dir_output, label_train, label_test):
    labels_train, labels_test = [], []
    N = len(label_train)
    for no, data in enumerate(label_train):
        print('/'.join(map(str, [no + 1, N])))
        labels_train.append(data)
    np.save(dir_output + '/train/fold/' + str(fold) + '/label.npy', labels_train)

    N = len(label_test)
    for no, data in enumerate(label_test):
        print('/'.join(map(str, [no + 1, N])))
        labels_test.append(data)
    np.save(dir_output + '/test/fold/' + str(fold) + '/label.npy', labels_test)


if __name__ == '__main__':
    dataset = ''
    file_path = ''
    file_path_protein = '.CSV'
    dir_output = ('/processed')
    os.makedirs(dir_output, exist_ok=True)
    raw_data_protein = pd.read_csv(file_path_protein)
    protein_values = raw_data_protein['protein_SMILES'].values
    protein_id_unique = raw_data_protein['protein_ID'].values

    N = len(protein_values)
    protein_max_len = 168

    protein_graph_construction(id=protein_id_unique, protein_values=protein_values, dir_output=dir_output)

    for fold in range(1, 2):
        train_data = pd.read_csv('train.CSV')
        test_data = pd.read_csv('test.CSV')
        protein_id_train = train_data['protein_ID'].values
        protein_id_test = test_data['protein_ID'].values

        label_train = train_data['Label'].values
        label_test = test_data['Label'].values

        protein_graph_process(dataset=dataset, fold=fold, id_train=protein_id_train, id_test=protein_id_test, dir_output=dir_output)
        protein_id_process(dataset=dataset, fold=fold, id_train=protein_id_train, id_test=protein_id_test, dir_output=dir_output)
        Label_process(dataset=dataset, fold=fold, dir_output=dir_output, label_train=label_train, label_test=label_test)

    print('The preprocess of ' + dataset + ' dataset has finished!')

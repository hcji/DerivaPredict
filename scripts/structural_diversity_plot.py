# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:35:30 2025

@author: DELL
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import umap
import matplotlib.pyplot as plt


def get_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Generate Morgan fingerprints for a list of SMILES.
    """
    fingerprints = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(fp)
            valid_smiles.append(smi)
    return fingerprints, valid_smiles


def visualize_chemical_space(fps_list, labels):
    """
    Visualize chemical spaces using UMAP.
    """
    combined_fps = np.vstack(fps_list)
    label_list = [label for i, fps in enumerate(fps_list) for label in [labels[i]] * len(fps)]
    
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1, metric="jaccard")
    embeddings = reducer.fit_transform(combined_fps)
    
    plt.figure(figsize=(4, 4), dpi=300)
    for label in labels:
        indices = [i for i, lbl in enumerate(label_list) if lbl == label]
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], s=5 , label=label, alpha=0.4)
    
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.show()


def plot_histograms(data_list, labels, bins=30, xlabel="Values", ylabel="Frequency"):
    """
    Plot histograms for multiple sets of values in one figure.

    Parameters:
        data_list (list of lists): A list of datasets to plot.
        labels (list of str): Labels for the datasets.
        bins (int): Number of bins for the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(4, 4), dpi=300)
    for data, label in zip(data_list, labels):
        plt.hist(data, bins=bins, alpha=0.4, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


biochemical = pd.read_csv('example/curcumin/biochemical/DTI_table_Biochemical-Template-based_30xx2_0110_085149.csv')
chemical = pd.read_csv('example/curcumin/chemical/DTI_table_Chemical-Template-based_30xx2_0110_085233.csv')
metabolic = pd.read_csv('example/curcumin/metabolic/DTI_table_BioTransformer-Environmental microbial_30xx3_0110_085512.csv')


smi_list1 = biochemical['SMILES'].values
smi_list2 = chemical['SMILES'].values
smi_list3 = metabolic['SMILES'].values

fps1, valid_smiles1 = get_morgan_fingerprints(smi_list1)
fps2, valid_smiles2 = get_morgan_fingerprints(smi_list2)
fps3, valid_smiles3 = get_morgan_fingerprints(smi_list3)

visualize_chemical_space(
    fps_list=[np.array(fps1), np.array(fps2), np.array(fps3)],
    labels=["Biochemical", "Chemical", "Metabolic"],
)


smi = 'COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O'
fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048)

sim1 = np.array([DataStructs.TanimotoSimilarity(fp, f) for f in fps1])
sim2 = np.array([DataStructs.TanimotoSimilarity(fp, f) for f in fps2])
sim3 = np.array([DataStructs.TanimotoSimilarity(fp, f) for f in fps3])

plot_histograms(
    data_list=[sim1, sim2, sim3],
    labels=["Biochemical", "Chemical", "Metabolic"],
    bins=30,
    xlabel="Tanimoto Similarity",
    ylabel="Frequency"
)


data1 = biochemical['QED'].values
data2 = chemical['QED'].values
data3 = metabolic['QED'].values

plot_histograms(
    data_list=[data1, data2, data3],
    labels=["Biochemical", "Chemical", "Metabolic"],
    bins=40,
    xlabel="QED",
    ylabel="Frequency"
)


data1 = biochemical['SCScore'].values
data2 = chemical['SCScore'].values
data3 = metabolic['SCScore'].values

plot_histograms(
    data_list=[data1, data2, data3],
    labels=["Biochemical", "Chemical", "Metabolic"],
    bins=40,
    xlabel="SCScore",
    ylabel="Frequency"
)

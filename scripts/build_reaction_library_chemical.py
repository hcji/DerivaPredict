# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:46:21 2024

@author: DELL
"""


import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction


rxns = pd.read_csv('data/ChemReaction.csv')
mapped_rxns = rxns['rxn_smiles'].values

templates = []
for r in tqdm(mapped_rxns):
    try:
        rxn = ChemicalReaction(r).generate_reaction_template(radius=1)
    except:
        continue
    for r in rxn:
        templates.append(r.smarts)
templates = list(set(templates))


txt = open('data/ChemTemplates.txt', 'a')
for temp in templates:
    txt.write(temp)
    txt.write('\n')
txt.close()


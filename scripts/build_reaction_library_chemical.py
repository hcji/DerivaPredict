# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:46:21 2024

@author: DELL
"""

import json
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction


def update_rxn(smiles):
    reactants, products = smiles.split(">>")
    reactant_list = reactants.split(".")
    largest_molecule = max(reactant_list, key=lambda r: Chem.MolFromSmiles(r).GetNumHeavyAtoms() if Chem.MolFromSmiles(r) else 0)
    return f"{largest_molecule}>>{products}"


rxns = pd.read_csv('data/ChemReaction.csv')
mapped_rxns = rxns['rxn_smiles'].values

templates = []
rxn_mapper = RXNMapper()
for rxn in tqdm(mapped_rxns):
    rxn = update_rxn(rxn)
    try:
        mapping = rxn_mapper.get_attention_guided_atom_maps([rxn])
    except:
        continue
    mapped_rxn = mapping[0]['mapped_rxn']
    try:
        mapped_rxn = ChemicalReaction(mapped_rxn).generate_reaction_template(radius=1)
    except:
        continue
    for r in mapped_rxn:
        templates.append(r.smarts)
templates = list(set(templates))

with open('data/ChemTemplates.json', 'w') as f:
    json.dump(templates, f, indent=1)


# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:46:21 2024

@author: DELL
"""


import json
from tqdm import tqdm
from rdkit import Chem
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction


with open('data/BioReactions.txt') as txt:
    rxns = txt.readlines()
print(len(rxns))


def update_rxn(smiles):
    reactants, products = smiles.split(">>")
    reactant_list = reactants.split(".")
    largest_molecule = max(reactant_list, key=lambda r: Chem.MolFromSmiles(r).GetNumHeavyAtoms() if Chem.MolFromSmiles(r) else 0)
    return f"{largest_molecule}>>{products}"


mapped_rxns = []
rxn_mapper = RXNMapper()
for rxn in tqdm(rxns):
    rxn = rxn.replace('\n', '')
    rxn = update_rxn(rxn)
    if '.' in rxn.split('>>')[0]:
        break
    
    try:
        mapping = rxn_mapper.get_attention_guided_atom_maps([rxn])
    except:
        continue
    mapped_rxn = mapping[0]['mapped_rxn']
    mapped_rxns.append(mapped_rxn)


templates = []
for r in tqdm(mapped_rxns):
    try:
        rxn = ChemicalReaction(r).generate_reaction_template(radius=1)
    except:
        continue
    for r in rxn:
        templates.append(r.smarts)
templates = list(set(templates))


with open('data/BioTemplates.json', 'w') as f:
    json.dump(templates, f, indent=1)


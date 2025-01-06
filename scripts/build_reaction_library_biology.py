# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:46:21 2024

@author: DELL
"""


from tqdm import tqdm
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction


with open('data/BioReactions.txt') as txt:
    rxns = txt.readlines()
print(len(rxns))


mapped_rxns = []
rxn_mapper = RXNMapper()
for rxn in tqdm(rxns):
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


txt = open('data/BioTemplates.txt', 'a')
for temp in templates:
    txt.write(temp)
    txt.write('\n')
txt.close()


# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:39:16 2024

@author: DELL
"""


import os
import json
import requests
import subprocess
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.QED import qed

from admet_ai import ADMETModel
from DeepPurpose import utils, DTI
from chembl_webresource_client.new_client import new_client

from core.scscore import SCScorer



with open('data/ChemTemplates.json') as js:
    chemical_templetes = list(json.load(js))
    chemical_templetes = [temp for temp in chemical_templetes if ('.' not in temp) and ('*' not in temp)]
    chemical_rxns = [AllChem.ReactionFromSmarts(temp) for temp in chemical_templetes]

with open('data/BioTemplates.json') as js:
    biochemical_templetes = list(json.load(js))
    biochemical_templetes = [temp for temp in biochemical_templetes if ('.' not in temp) and ('*' not in temp)]
    biochemical_rxns = [AllChem.ReactionFromSmarts(temp) for temp in biochemical_templetes]

'''
with open('data/ChemTemplates.txt') as txt:
    chemical_templetes = txt.readlines()
    chemical_templetes = [temp for temp in chemical_templetes if ('.' not in temp) and ('*' not in temp)]
    chemical_rxns = [AllChem.ReactionFromSmarts(temp) for temp in chemical_templetes]

with open('data/BioTemplates.txt') as txt:
    biochemical_templetes = txt.readlines()
    biochemical_templetes = [temp for temp in biochemical_templetes if ('.' not in temp) and ('*' not in temp)]
    biochemical_rxns = [AllChem.ReactionFromSmarts(temp) for temp in biochemical_templetes]
'''

def retrieve_gene_from_name(gene_name):
    gene_results = new_client.target.filter(target_synonym__icontains=gene_name)
    uniprot_ids, results = [], []
    if gene_results:
        for gene in gene_results:
            gene = gene['target_components'][0]
            uniprot_id = gene['accession']
            if uniprot_id in uniprot_ids:
                continue
            else:
                uniprot_ids.append(uniprot_id)
            description = gene['component_description']
            gene_symbols = gene['target_component_synonyms']
            gene_symbols = ','.join([entry['component_synonym'] for entry in gene_symbols if entry['syn_type'] == 'GENE_SYMBOL'])
            results.append([uniprot_id, gene_symbols, description])
        return pd.DataFrame(results, columns = ['uniprot_id', 'gene_symbols', 'description'])
    else:
        return None


def retrieve_protein_sequence(protein_id):
    url = f"https://www.uniprot.org/uniprot/{protein_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_lines = response.text.split('\n')
        sequence = ''.join(fasta_lines[1:])
        return sequence
    else:
        return None


def predict_compound_derivative_chemical_templete(smiles_list, n_loop = 2, n_branch = 20, sim_filter = 0.5, rxn_data = 'chemical_rxns'):
    '''
    with open('example/taxoids.smi') as txt:
        smiles_list = txt.readlines()
    '''
    get_fingerprint = lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
    
    def flatten_list(lst):
        flattened = []
        for item in lst:
            if isinstance(item, tuple):
                flattened.extend(flatten_list(item))
            else:
                flattened.append(item)
        return flattened
    
    def predict_products(smi):
        mol = Chem.MolFromSmiles(smi)
        fingerprint = get_fingerprint(mol)
        if rxn_data == 'chemical_rxns':
            products = [rxn.RunReactants([mol]) for rxn in chemical_rxns]
        else:
            products = [rxn.RunReactants([mol]) for rxn in biochemical_rxns]
        products = [s for s in products if len(s) > 0]
        products = flatten_list(products)
        products_smiles = np.unique([Chem.MolToSmiles(p) for p in products])        
        products_similarity = []
        for p in products_smiles:
            try:
                m = Chem.MolFromSmiles(p)
                simi = DataStructs.FingerprintSimilarity(fingerprint, get_fingerprint(m))
                if simi == 1:
                    simi = 0
                products_similarity.append(simi)
            except:
                products_similarity.append(0)
        products_similarity = np.array(products_similarity)
        
        products_smiles = products_smiles[products_similarity >= sim_filter]
        products_similarity = products_similarity[products_similarity >= sim_filter]
        k_indices = np.argsort(-products_similarity)
        k_indices = k_indices[:min(len(k_indices), n_branch)]
        products_smiles = products_smiles[k_indices]
        products_similarity = products_similarity[k_indices]
        output = pd.DataFrame({'precursor': smi, 'derivant': products_smiles})
        return output

    derivative_list = pd.concat([predict_products(smi) for smi in smiles_list], ignore_index=True)
    derivative_list = derivative_list.loc[np.unique(derivative_list.loc[:,'derivant'], return_index=True)[1], :]
    derivative_list = derivative_list.reset_index(drop = True)
    if not derivative_list.empty: ## change here
        if n_loop >= 2:
            derivative_list_2 = pd.concat([predict_products(smi) for smi in derivative_list.loc[:,'derivant']], ignore_index=True)
            if n_loop >= 3:
                derivative_list_3 = pd.concat([predict_products(smi) for smi in derivative_list_2.loc[:,'derivant']], ignore_index=True)
                derivative_list = pd.concat([derivative_list, derivative_list_2, derivative_list_3])
                derivative_list = derivative_list.reset_index(drop = True)
                derivative_list = derivative_list.loc[np.unique(derivative_list.loc[:,'derivant'], return_index=True)[1], :]
                derivative_list = derivative_list.reset_index(drop = True)
            else:
                derivative_list = pd.concat([derivative_list, derivative_list_2])
                derivative_list = derivative_list.reset_index(drop = True)
                derivative_list = derivative_list.loc[np.unique(derivative_list.loc[:,'derivant'], return_index=True)[1], :]
                derivative_list = derivative_list.reset_index(drop = True)
    return derivative_list


def predict_compound_derivative_biotransformer(smiles_list, n_loop = 2, sim_filter = 0.5, method = 'ecbased'):
    
    get_fingerprint = lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
    
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [m for m in mols if mols is not None]
    try:
        os.unlink('biotransformer/precursors.sdf')
        os.unlink('biotransformer/output.csv')
    except:
        pass
    with Chem.SDWriter('biotransformer/precursors.sdf') as w:
        for m in mols:
            w.write(m)
    cwd = 'biotransformer'
    cmdline = 'java -jar BioTransformer3.0_20230525.jar -k pred '
    cmdline += '-b {} '.format(method)
    cmdline += '-s {} '.format(n_loop)
    cmdline += '-isdf precursors.sdf '
    cmdline += '-ocsv output.csv '
    subprocess.call(cmdline, cwd = cwd)
    if os.path.isfile('biotransformer/output.csv'):
        derivative_list = pd.read_csv('biotransformer/output.csv')
        derivative_list = derivative_list.loc[:,['Precursor SMILES', 'SMILES']]
        derivative_list.columns = ['precursor', 'derivant']
        derivative_list = derivative_list.loc[np.unique(derivative_list.loc[:,'derivant'], return_index=True)[1], :]
        derivative_list = derivative_list.reset_index(drop = True)

        products_similarity = []
        for i in derivative_list.index:
            try:
                fp1 = get_fingerprint(Chem.MolFromSmiles(derivative_list.loc[i, 'precursor']))
                fp2 = get_fingerprint(Chem.MolFromSmiles(derivative_list.loc[i, 'derivant']))
                products_similarity.append(DataStructs.FingerprintSimilarity(fp1, fp2))
            except:
                products_similarity.append(0)
        products_similarity = np.array(products_similarity)

        derivative_list = derivative_list.loc[products_similarity >= sim_filter, :]
        derivative_list = derivative_list.reset_index(drop = True)
    else:
        derivative_list = pd.DataFrame()
    return derivative_list


def predict_compound_target_affinity(smiles_list, target_list, affinity_model_type = 'CNN-CNN'):

    model_DTIs = {
    'CNN-CNN': DTI.model_pretrained(model = 'CNN_CNN_BindingDB'),
    'MPNN-CNN': DTI.model_pretrained(model = 'MPNN_CNN_BindingDB'),
    'Morgan-CNN': DTI.model_pretrained(model = 'Morgan_CNN_BindingDB'),
    'baseTransformer-CNN': DTI.model_pretrained(model = 'Transformer_CNN_BindingDB')
    }
    
    results = []
    smiles_list = np.unique(smiles_list)
    target_list = np.unique(target_list)
    for target in target_list:
        X_pred = utils.data_process(X_drug = smiles_list, X_target = [target], y = [0]*len(smiles_list),
                                    drug_encoding = affinity_model_type.split('-')[0], 
                                    target_encoding = affinity_model_type.split('-')[1], 
                                    split_method='no_split')
        model_DTI = model_DTIs[affinity_model_type]
        y_pred = np.array(model_DTI.predict(X_pred))
        y_pred = 10**(-y_pred) / 1e-9
        output = X_pred.loc[:,['SMILES', 'Target Sequence']]
        output['Predicted Value'] = y_pred
        results.append(output)
    results = pd.concat(results, ignore_index=True)
    return results


def predict_compound_ADMET_property(smiles_list):
    results = {}
    model = ADMETModel()
    smiles_list = np.unique(smiles_list)
    results = model.predict(smiles=smiles_list)
    return results.reset_index()


def predict_compound_scscore(smiles_list):
    results = []
    model = SCScorer()
    model.restore()
    for smi in smiles_list:
        try:
            (smi, score) = model.get_score_from_smi(smi)
            score = np.round(score, 3)
        except:
            score = np.nan
        results.append(score)
    return results


def predict_compound_qed(smiles_list):
    results = []
    for smi in smiles_list:
        try:
            qed_score = np.round(qed(Chem.MolFromSmiles(smi)), 3)
        except:
            qed_score = np.nan
        results.append(qed_score)
    return results


def refine_compound_ADMET_property(ADMET_list, smiles, property_class = 'Physicochemical'):
    properties_mapping = {
        'Physicochemical': [
            'molecular_weight',
            'logP',
            'hydrogen_bond_acceptors',
            'hydrogen_bond_donors',
            'Lipinski',
            'stereo_centers',
            'tpsa',
            'molecular_weight_drugbank_approved_percentile',
            'logP_drugbank_approved_percentile',
            'hydrogen_bond_acceptors_drugbank_approved_percentile',
            'hydrogen_bond_donors_drugbank_approved_percentile',
            'Lipinski_drugbank_approved_percentile',
            'stereo_centers_drugbank_approved_percentile',
            'tpsa_drugbank_approved_percentile',
            'HydrationFreeEnergy_FreeSolv',
            'HydrationFreeEnergy_FreeSolv_drugbank_approved_percentile',
            'Lipophilicity_AstraZeneca',
            'Lipophilicity_AstraZeneca_drugbank_approved_percentile',
            'Solubility_AqSolDB',
            'Solubility_AqSolDB_drugbank_approved_percentile'
        ],
        'Absorption': [
            'HIA_Hou',
            'PAMPA_NCATS',
            'Pgp_Broccatelli',
            'Caco2_Wang',
            'HIA_Hou_drugbank_approved_percentile',
            'PAMPA_NCATS_drugbank_approved_percentile',
            'Pgp_Broccatelli_drugbank_approved_percentile',
            'Caco2_Wang_drugbank_approved_percentile'
        ],
        'Distribution': [
            'BBB_Martins',
            'BBB_Martins_drugbank_approved_percentile',
            'VDss_Lombardo',
            'VDss_Lombardo_drugbank_approved_percentile'
        ],
        'Metabolism': [
            'CYP1A2_Veith',
            'CYP2C19_Veith',
            'CYP2C9_Substrate_CarbonMangels',
            'CYP2C9_Veith',
            'CYP2D6_Substrate_CarbonMangels',
            'CYP2D6_Veith',
            'CYP3A4_Substrate_CarbonMangels',
            'CYP3A4_Veith',
            'CYP1A2_Veith_drugbank_approved_percentile',
            'CYP2C19_Veith_drugbank_approved_percentile',
            'CYP2C9_Substrate_CarbonMangels_drugbank_approved_percentile',
            'CYP2C9_Veith_drugbank_approved_percentile',
            'CYP2D6_Substrate_CarbonMangels_drugbank_approved_percentile',
            'CYP2D6_Veith_drugbank_approved_percentile',
            'CYP3A4_Substrate_CarbonMangels_drugbank_approved_percentile',
            'CYP3A4_Veith_drugbank_approved_percentile'
        ],
        'Excretion': [
            'Half_Life_Obach',
            'Half_Life_Obach_drugbank_approved_percentile',
            'Clearance_Hepatocyte_AZ',
            'Clearance_Hepatocyte_AZ_drugbank_approved_percentile',
            'Clearance_Microsome_AZ',
            'Clearance_Microsome_AZ_drugbank_approved_percentile'
        ],
        'Toxicity': [
            'AMES',
            'Carcinogens_Lagunin',
            'ClinTox',
            'DILI',
            'NR-AR-LBD',
            'NR-AR',
            'NR-AhR',
            'NR-Aromatase',
            'NR-ER-LBD',
            'NR-ER',
            'NR-PPAR-gamma',
            'SR-ARE',
            'SR-ATAD5',
            'SR-HSE',
            'SR-MMP',
            'SR-p53',
            'Skin_Reaction',
            'hERG',
            'LD50_Zhu',
            'Carcinogens_Lagunin_drugbank_approved_percentile',
            'ClinTox_drugbank_approved_percentile',
            'DILI_drugbank_approved_percentile',
            'NR-AR-LBD_drugbank_approved_percentile',
            'NR-AR_drugbank_approved_percentile',
            'NR-AhR_drugbank_approved_percentile',
            'NR-Aromatase_drugbank_approved_percentile',
            'NR-ER-LBD_drugbank_approved_percentile',
            'NR-ER_drugbank_approved_percentile',
            'NR-PPAR-gamma_drugbank_approved_percentile',
            'SR-ARE_drugbank_approved_percentile',
            'SR-ATAD5_drugbank_approved_percentile',
            'SR-HSE_drugbank_approved_percentile',
            'SR-MMP_drugbank_approved_percentile',
            'SR-p53_drugbank_approved_percentile',
            'Skin_Reaction_drugbank_approved_percentile',
            'hERG_drugbank_approved_percentile',
            'LD50_Zhu_drugbank_approved_percentile'
        ]}
    rows = np.where(ADMET_list['index'] == smiles)[0][0]
    cols = properties_mapping[property_class]
    results = pd.DataFrame({'predicted property': cols, 'values': np.round(ADMET_list.loc[rows, cols].astype(float), 4)})
    results = results.reset_index(drop = True)
    return results

'''
## pip install torchtext==0.6.0
from baseTransformer.translate import _get_parser, translate_fn
def predict_compound_derivative_base_transformer(smiles_list, n_loop = 2, n_branch = 20, sim_filter = 0.5, model_type = 'Chemical'):
    parser = _get_parser()
    opt = parser.parse_args()
    if model_type == 'Chemical':
        opt.models = ['baseTransformer/model/USPTO_50K_based_model4retrosynthesis.pt']
    else:   # model_type == 'Biological':
        opt.models = ['baseTransformer/model/Biochem_based_model4retrosynthesis.pt']
    ##↓↓ 进行预测并 获取预测结果
    result = {}
    smiles_list = [smi.replace(" ", "") for smi in smiles_list]
    lst_temp = smiles_list

    df_res_unit_lst = []
    for i_d in range(n_loop):
        result[f'{i_d+1}th_level_prediction'] = []
        for each in lst_temp:
            smi = each       #### 实际输入opt.input 为单个字符串。对此进行translate预测
            res = translate_fn(opt, smi, n_branch)
            res_smiles_lst = res['sequences']

            df_res_unit = pd.DataFrame({'precursor': smi, 'derivant': res_smiles_lst})
            df_res_unit_lst.append(df_res_unit)

            result[f'{i_d+1}th_level_prediction'].extend(res['sequences'])
        lst_temp = result[f'{i_d+1}th_level_prediction']
    derivative_list = pd.concat(df_res_unit_lst, ignore_index=True)

    get_fingerprint = lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
    products_similarity = []
    for i in derivative_list.index:
        try:
            fp1 = get_fingerprint(Chem.MolFromSmiles(derivative_list.loc[i, 'precursor']))
            fp2 = get_fingerprint(Chem.MolFromSmiles(derivative_list.loc[i, 'derivant']))
            products_similarity.append(DataStructs.FingerprintSimilarity(fp1, fp2))
        except:
            products_similarity.append(0)
    products_similarity = np.array(products_similarity)

    derivative_list = derivative_list.loc[products_similarity >= sim_filter, :]
    derivative_list = derivative_list.reset_index(drop=True)
    print(derivative_list)
    return derivative_list

from MolBART.translate import main, cano_smi_lst
import argparse
def predict_compound_derivative_MolBART(smiles_list, n_loop = 2, n_branch = 20, model_type = 'Chemical'):### 环境冲突，暂不可用。
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    if model_type == 'Chemical':
        args.model_path = 'MolBART/epoch=29-step=83249.ckpt'
    else:   # model_type == 'Biological':
        args.model_path = 'MolBART/epoch=29-step=83249.ckpt'
    args.vocab_path = 'MolBART/bart_vocab_downstream_4_Biochem.txt'
    args.chem_token_start_idx = 272
    args.task = "backward_prediction"
    args.batch_size = 16
    args.num_beams = n_branch
    ##↓↓ 进行预测并 获取预测结果
    print(f"Loading model from {args.model_path}...")

    result = {}
    df_res_unit_lst = []
    for i_d in range(n_loop):
        print(len(smiles_list))
        result[f'{i_d + 1}th_level_prediction'] = []
        result[f'{i_d + 1}th_level_prediction_valid'] = []
        result[f'{i_d + 1}th_level_prediction'] = main(args, smiles_list)
        pred_res_lst = main(args, smiles_list)

        for i0, smi0 in enumerate(smiles_list):
            per_unit_left = i0 * args.batch_size
            per_unit_right = (i0 + 1) * args.batch_size
            per_unit_lst = pred_res_lst[per_unit_left:per_unit_right]

            filtered_unit_lst = cano_smi_lst(per_unit_lst)
            result[f'{i_d + 1}th_level_prediction_valid'].extend(filtered_unit_lst)
            df_res_unit = pd.DataFrame({'precursor': smi0, 'derivant': filtered_unit_lst})
            df_res_unit_lst.append(df_res_unit)
        smiles_list = result[f'{i_d + 1}th_level_prediction_valid']

    derivative_list = pd.concat(df_res_unit_lst, ignore_index=True)
    df = pd.DataFrame(derivative_list)
    print(df)
    return derivative_list
'''

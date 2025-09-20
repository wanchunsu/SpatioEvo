import json
import os
import os.path as osp
import argparse
import numpy as np
from dictionary_tools import dump_json, load_json
from pdb_annotations_tools import load_pdb_resolution
from Bio.PDB.MMCIFParser import MMCIFParser

### This script computes the distance between residues based on the highest resolution PDB structure that these residues map to.

def store_res_dnds_and_auth_pos(auth_to_seq_pos_dict, res_type, dnds_split_by_prots):
    ''' Make a mapping dictionary to store residues on human target proteins and their corresponding PDB chains and auth residue positions 
        (auth residue positions are needed for computing the distance between residues -- since PDB files are in terms of auth rather than uniprot seq positions)

    Args:
        auth_to_seq_pos_dict: dicitonary storing auth to uniprot sequence position mappings
        res_type: the interfacial residue type that we're interested in
        dnds_split_by_prots: dictionary storing dnds values for each residue category


    Return:
        dnds_info_with_pdb_and_auth_info: dictionary storing residues, their dnds values, and their auth residue positions 
    
        The output dictionary will look something like:
        Res type: Targ_prot: res pos: {structure_id: {chain_id: auth_res, …}, …}, res pos
    '''

    dnds_info_with_pdb_and_auth_info = {}
   
    dnds_vals = dnds_split_by_prots[res_type]
    dnds_info_with_pdb_and_auth_info = {}

    for targ_prot in dnds_vals:
        dnds_info_with_pdb_and_auth_info[targ_prot] = {}

        for residue in dnds_vals[targ_prot]:
            dnds = dnds_vals[targ_prot][residue]
            dnds_info_with_pdb_and_auth_info[targ_prot][residue] = {}

            for pdb_chain in auth_to_seq_pos_dict[targ_prot]:

                # pdb_id = pdb_chain.split('_')[0]
                # pdb_chainid = pdb_chain.split('_')[1]

                auth_res = get_auth_from_seq_pos(auth_to_seq_pos_dict[targ_prot][pdb_chain], residue)
                if auth_res == -100: # the current uniprot protein res position does not exist in this pdb chain
                    continue
                else:
                    if pdb_chain not in dnds_info_with_pdb_and_auth_info[targ_prot][residue]:
                        dnds_info_with_pdb_and_auth_info[targ_prot][residue][pdb_chain] = {}
                    dnds_info_with_pdb_and_auth_info[targ_prot][residue][pdb_chain] = auth_res
    return dnds_info_with_pdb_and_auth_info

def store_res_dnds_and_auth_pos_for_mimicked(exo_auth_to_seq_pos_dict, endo_auth_to_seq_pos_dict, res_type, dnds_split_by_prots):
    ''' Make a mapping dictionary to store residues on human target proteins and their corresponding PDB chains and auth residue positions 
        (auth residue positions are needed for computing the distance between residues -- since PDB files are in terms of auth rather than uniprot seq positions)

    Args:
        exo_auth_to_seq_pos_dict: dicitonary storing auth to uniprot sequence position mappings (for exo)
        endo_auth_to_seq_pos_dict: dicitonary storing auth to uniprot sequence position mappings (for endo)
        res_type: the interfacial residue type that we're interested in
        dnds_split_by_prots: dictionary storing dnds values for each residue category


    Return:
        dnds_info_with_pdb_and_auth_info: dictionary storing residues, their dnds values, and their auth residue positions 
    
        The output dictionary will look something like:
        Res type: Targ_prot: res pos: {structure_id: {chain_id: auth_res, …}, …}, res pos
    '''

    dnds_info_with_pdb_and_auth_info = {}
   
    dnds_vals = dnds_split_by_prots[res_type]
    dnds_info_with_pdb_and_auth_info = {}

    for targ_prot in dnds_vals:
        dnds_info_with_pdb_and_auth_info[targ_prot] = {}

        for residue in dnds_vals[targ_prot]:
            dnds = dnds_vals[targ_prot][residue]
            dnds_info_with_pdb_and_auth_info[targ_prot][residue] = {}

            for pdb_chain in exo_auth_to_seq_pos_dict[targ_prot]:

                # pdb_id = pdb_chain.split('_')[0]
                # pdb_chainid = pdb_chain.split('_')[1]

                auth_res = get_auth_from_seq_pos(exo_auth_to_seq_pos_dict[targ_prot][pdb_chain], residue)
                if auth_res == -100: # the current uniprot protein res position does not exist in this pdb chain
                    continue
                else:
                    if pdb_chain not in dnds_info_with_pdb_and_auth_info[targ_prot][residue]:
                        dnds_info_with_pdb_and_auth_info[targ_prot][residue][pdb_chain] = {}
                    dnds_info_with_pdb_and_auth_info[targ_prot][residue][pdb_chain] = auth_res

            for pdb_chain in endo_auth_to_seq_pos_dict[targ_prot]:

                # pdb_id = pdb_chain.split('_')[0]
                # pdb_chainid = pdb_chain.split('_')[1]

                auth_res = get_auth_from_seq_pos(endo_auth_to_seq_pos_dict[targ_prot][pdb_chain], residue)
                if auth_res == -100: # the current uniprot protein res position does not exist in this pdb chain
                    continue
                else:
                    if pdb_chain not in dnds_info_with_pdb_and_auth_info[targ_prot][residue]:
                        dnds_info_with_pdb_and_auth_info[targ_prot][residue][pdb_chain] = {}
                    dnds_info_with_pdb_and_auth_info[targ_prot][residue][pdb_chain] = auth_res
    return dnds_info_with_pdb_and_auth_info


def get_distance_between_residues(memo_distances_fi, pdb_resolution_dict, res_type, dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser):   

    dict_of_distances = {}

    print(f'\n### Calculating distances for {res_type} ###', flush=True)


    for targ_prot in dnds_info_with_pdb_and_auth_info:
        memo_distances_dict = {}
        if osp.exists(memo_distances_fi):
            memo_distances_dict = load_json(memo_distances_fi)
            
        if len(dnds_info_with_pdb_and_auth_info[targ_prot])==1: #only one interfacial residue here, so can't get distance between two residues on this protein
            print(f'\t\t{targ_prot}, skipped (only one interfacial residue)', flush=True)
            continue
        else:
            print(f'\t\t{targ_prot}', flush=True)
            for res_pos1 in dnds_info_with_pdb_and_auth_info[targ_prot]:
                
                for res_pos2 in dnds_info_with_pdb_and_auth_info[targ_prot]:
                    
                    if res_pos1 != res_pos2: # we want to look at distance between two DIFF residues (ensure that r1 and r2 aren't the same)
                        
                        if targ_prot in dict_of_distances:
                            
                            if res_pos1 + '-' + res_pos2 in dict_of_distances[targ_prot] or res_pos2 + '-' + res_pos1 in dict_of_distances[targ_prot]:
                                continue # we've already stored this pair of residues (note: res1-res2 distance is same as res2-res1 distance)
                        
                        # print(res_pos1, res_pos2)
                        # get list of pdb chains that these two residues both appear on
                        pdb_chains_in_common = [pc for pc in dnds_info_with_pdb_and_auth_info[targ_prot][res_pos1].keys() if pc in dnds_info_with_pdb_and_auth_info[targ_prot][res_pos2].keys()]

                        if len(pdb_chains_in_common) == 0: # no chains in common, so we can't find the distance between these two residues
                            continue
                        
                        # now pick the best pdb structure to go with (based on resolution)
                        pdb_ids_to_check_for_res = list(set([pdb_chains.split('_')[0] for pdb_chains in pdb_chains_in_common]))
                        pdb_ids_with_res = {pdbid: pdb_resolution_dict[pdbid] for pdbid in pdb_ids_to_check_for_res}

                        # pdb_with_best_resol = min(pdb_ids_with_res, key=pdb_ids_with_res.get) # get the pdb structure with the smallest (best) resolution value [this could give diff pdb structures each time its run if there is more than one best resolution pdb structure! We'll replace this line with the below two lines]
                        
                        min_res = min(pdb_ids_with_res.values()) # get minimum resolution out of all mapped pdb structures
                        pdb_with_best_resol = sorted([k for k, v in pdb_ids_with_res.items() if v == min_res])[0] # get first pdb structure (alphabetically) with the min resolution


                        pdb_chains_to_get_distances_for = [pc for pc in pdb_chains_in_common if pc.startswith(pdb_with_best_resol)]
                        
                        min_distances_btwn_residues_list = []

                        # load the pdb structure
                        model_loaded = False
                        if len(pdb_chains_to_get_distances_for) > 1: # load structure first b/c will likely need to use more than once
                            structure = parser.get_structure(pdb_with_best_resol, os.path.join(pdb_fis_dir, pdb_with_best_resol.lower() + '.cif'))
                            model = structure[0]
                            model_loaded = True
                        for p_chain in pdb_chains_to_get_distances_for: # go through chain(s) in the pdb structure where the two residues are both present
                            # print(p_chain, res_pos1, res_pos2)
                            # get the min atomic distance between the two residues (res_pos1 and res_pos2) on this chain(p_chain)
                            auth_res_pos1 = dnds_info_with_pdb_and_auth_info[targ_prot][res_pos1][p_chain]
                            auth_res_pos2 = dnds_info_with_pdb_and_auth_info[targ_prot][res_pos2][p_chain]
                            
                            # Get the min atomic distance between the two residues 
                            # (either from the memoized dict of distances if already seen b4, or by calling the get_min_atomic_distance_between_2_residues fn and storing in memoized dict)
                            if p_chain in memo_distances_dict:
                                
                                if str(auth_res_pos1) + '-' + str(auth_res_pos2) in memo_distances_dict[p_chain]:
                                    # print("curr already in memo dict")
                                    min_distance_btwn_residues = memo_distances_dict[p_chain][str(auth_res_pos1)+ '-' + str(auth_res_pos2)]
                               
                                elif str(auth_res_pos2) + '-' + str(auth_res_pos1) in memo_distances_dict[p_chain]:
                                    # print("rev curr already in memo dict")
                                    min_distance_btwn_residues = memo_distances_dict[p_chain][str(auth_res_pos2) + '-' + str(auth_res_pos1)]
                                
                                else:
                                    if model_loaded == False:
                                        structure = parser.get_structure(pdb_with_best_resol, os.path.join(pdb_fis_dir, pdb_with_best_resol.lower() + '.cif'))
                                        model = structure[0]
                                    min_distance_btwn_residues = get_min_atomic_distance_between_2_residues(model, p_chain.split('_')[1], auth_res_pos1, auth_res_pos2)
                                    memo_distances_dict[p_chain][str(auth_res_pos1) + '-' + str(auth_res_pos2)] = min_distance_btwn_residues
                                    
                            else:
                                if model_loaded == False:
                                    structure = parser.get_structure(pdb_with_best_resol, os.path.join(pdb_fis_dir, pdb_with_best_resol.lower() + '.cif'))
                                    model = structure[0]
                                min_distance_btwn_residues = get_min_atomic_distance_between_2_residues(model, p_chain.split('_')[1], auth_res_pos1, auth_res_pos2)
                                memo_distances_dict[p_chain] = {}
                                memo_distances_dict[p_chain][str(auth_res_pos1) + '-' + str(auth_res_pos2)] = min_distance_btwn_residues
                               
                            min_distances_btwn_residues_list.append(min_distance_btwn_residues)
                        
                        avg_min_distance_btwn_residues = np.mean(min_distances_btwn_residues_list)

                        if targ_prot not in dict_of_distances:
                            dict_of_distances[targ_prot] = {}

                        dict_of_distances[targ_prot][res_pos1+ '-' + res_pos2] = avg_min_distance_btwn_residues

            # After running thru each targ prot, replace original memoized distances fi with new one (containing additional residue distances computed here)
            dump_json(memo_distances_dict, mdf)
            

    return dict_of_distances
                         

def get_distance_between_residues2(memo_distances_fi, pdb_resolution_dict, res_type, dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser, tracking_fi, output_fi):   

    dict_of_distances = {}

    print(f'\n### Calculating distances for {res_type} ###', flush=True)

    tracking = ''
    if osp.exists(output_fi):
        with open(output_fi) as out_contents:
            tracking = out_contents.read()

    with open(output_fi, 'a') as o:

        for targ_prot in dnds_info_with_pdb_and_auth_info:
            memo_distances_dict = {}
            if osp.exists(memo_distances_fi):
                memo_distances_dict = load_json(memo_distances_fi)

            if len(dnds_info_with_pdb_and_auth_info[targ_prot])==1: #only one interfacial residue here, so can't get distance between two residues on this protein
                print(f'\t\t{targ_prot}, skipped (only one interfacial residue)', flush=True)
                continue
            else:
                print(f'\t\t{targ_prot}', flush=True)
                for res_pos1 in dnds_info_with_pdb_and_auth_info[targ_prot]:
                    
                    for res_pos2 in dnds_info_with_pdb_and_auth_info[targ_prot]:
                        if targ_prot+ '\t' + res_pos1+ '-' + res_pos2 in tracking:
                            continue
                        if res_pos1 != res_pos2: # we want to look at distance between two DIFF residues (ensure that r1 and r2 aren't the same)
                            
                            if targ_prot in dict_of_distances:
                                
                                if res_pos1 + '-' + res_pos2 in dict_of_distances[targ_prot] or res_pos2 + '-' + res_pos1 in dict_of_distances[targ_prot]:
                                    continue # we've already stored this pair of residues (note: res1-res2 distance is same as res2-res1 distance)
                            
                            # print(res_pos1, res_pos2)
                            # get list of pdb chains that these two residues both appear on
                            pdb_chains_in_common = [pc for pc in dnds_info_with_pdb_and_auth_info[targ_prot][res_pos1].keys() if pc in dnds_info_with_pdb_and_auth_info[targ_prot][res_pos2].keys()]

                            if len(pdb_chains_in_common) == 0: # no chains in common, so we can't find the distance between these two residues
                                continue
                            
                            # now pick the best pdb structure to go with (based on resolution)
                            pdb_ids_to_check_for_res = list(set([pdb_chains.split('_')[0] for pdb_chains in pdb_chains_in_common]))
                            pdb_ids_with_res = {pdbid: pdb_resolution_dict[pdbid] for pdbid in pdb_ids_to_check_for_res}

                            # pdb_with_best_resol = min(pdb_ids_with_res, key=pdb_ids_with_res.get) # get the pdb structure with the smallest (best) resolution value [this could give diff pdb structures each time its run if there is more than one best resolution pdb structure! We'll replace this line with the below two lines]
                            
                            min_res = min(pdb_ids_with_res.values()) # get minimum resolution out of all mapped pdb structures
                            pdb_with_best_resol = sorted([k for k, v in pdb_ids_with_res.items() if v == min_res])[0] # get first pdb structure (alphabetically) with the min resolution


                            pdb_chains_to_get_distances_for = [pc for pc in pdb_chains_in_common if pc.startswith(pdb_with_best_resol)]
                            
                            min_distances_btwn_residues_list = []

                            # load the pdb structure
                            model_loaded = False
                            if len(pdb_chains_to_get_distances_for) > 1: # load structure first b/c will likely need to use more than once
                                structure = parser.get_structure(pdb_with_best_resol, os.path.join(pdb_fis_dir, pdb_with_best_resol.lower() + '.cif'))
                                model = structure[0]
                                model_loaded = True
                            for p_chain in pdb_chains_to_get_distances_for: # go through chain(s) in the pdb structure where the two residues are both present
                                # print(p_chain, res_pos1, res_pos2)
                                # get the min atomic distance between the two residues (res_pos1 and res_pos2) on this chain(p_chain)
                                auth_res_pos1 = dnds_info_with_pdb_and_auth_info[targ_prot][res_pos1][p_chain]
                                auth_res_pos2 = dnds_info_with_pdb_and_auth_info[targ_prot][res_pos2][p_chain]
                                
                                # Get the min atomic distance between the two residues 
                                # (either from the memoized dict of distances if already seen b4, or by calling the get_min_atomic_distance_between_2_residues fn and storing in memoized dict)
                                if p_chain in memo_distances_dict:
                                    
                                    if str(auth_res_pos1) + '-' + str(auth_res_pos2) in memo_distances_dict[p_chain]:
                                        # print("curr already in memo dict")
                                        min_distance_btwn_residues = memo_distances_dict[p_chain][str(auth_res_pos1)+ '-' + str(auth_res_pos2)]
                                   
                                    elif str(auth_res_pos2) + '-' + str(auth_res_pos1) in memo_distances_dict[p_chain]:
                                        # print("rev curr already in memo dict")
                                        min_distance_btwn_residues = memo_distances_dict[p_chain][str(auth_res_pos2) + '-' + str(auth_res_pos1)]
                                    
                                    else:
                                        if model_loaded == False:
                                            structure = parser.get_structure(pdb_with_best_resol, os.path.join(pdb_fis_dir, pdb_with_best_resol.lower() + '.cif'))
                                            model = structure[0]
                                        min_distance_btwn_residues = get_min_atomic_distance_between_2_residues(model, p_chain.split('_')[1], auth_res_pos1, auth_res_pos2)
                                        memo_distances_dict[p_chain][str(auth_res_pos1) + '-' + str(auth_res_pos2)] = min_distance_btwn_residues
                                       
                                else:
                                    if model_loaded == False:
                                        structure = parser.get_structure(pdb_with_best_resol, os.path.join(pdb_fis_dir, pdb_with_best_resol.lower() + '.cif'))
                                        model = structure[0]
                                    min_distance_btwn_residues = get_min_atomic_distance_between_2_residues(model, p_chain.split('_')[1], auth_res_pos1, auth_res_pos2)
                                    memo_distances_dict[p_chain] = {}
                                    memo_distances_dict[p_chain][str(auth_res_pos1) + '-' + str(auth_res_pos2)] = min_distance_btwn_residues
                                   
                                min_distances_btwn_residues_list.append(min_distance_btwn_residues)
                            
                            avg_min_distance_btwn_residues = np.mean(min_distances_btwn_residues_list)

                            if targ_prot not in dict_of_distances:
                                dict_of_distances[targ_prot] = {}

                            dict_of_distances[targ_prot][res_pos1+ '-' + res_pos2] = avg_min_distance_btwn_residues
                            o.write(targ_prot+ '\t' + res_pos1+ '-' + res_pos2 + '\t' + avg_min_distance_btwn_residues + '\n')

                # After running thru each targ prot, replace original memoized distances fi with new one (containing additional residue distances computed here)
                
                dump_json(memo_distances_dict, mdf)

    return dict_of_distances


def get_distance_between_residues_by_populating_from_all_res(all_res_distances_fi, pdb_resolution_dict, res_type, dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser):   
    '''
    This will be for computing endo-spec and exo-spec (just read in results from all_endo and all_exo_respectively as memo distances) 
    Args:
        memo_distances_fi:
        pdb_resolution_dict:
        res_type:
        dnds_info_with_pdb_and_auth_info:
        pdb_fis_dir:
        parser:


    '''

    memo_distances_dict = {} # this is the dictionary storing all residue distances belonging to either all_exo OR all_endo categories.
   
   
    memo_distances_dict = load_json(all_res_distances_fi)

    dict_of_distances = {}

    print(f'\n### Calculating distances for {res_type} ###', flush=True)



    for targ_prot in dnds_info_with_pdb_and_auth_info:
        if len(dnds_info_with_pdb_and_auth_info[targ_prot])==1: #only one interfacial residue here, so can't get distance between two residues on this protein
            print(f'\t\t{targ_prot}, skipped (only one interfacial residue)', flush=True)
            continue
        else:
            print(f'\t\t{targ_prot}', flush=True)
            for res_pos1 in dnds_info_with_pdb_and_auth_info[targ_prot]:
                
                for res_pos2 in dnds_info_with_pdb_and_auth_info[targ_prot]:
                    
                    if res_pos1 != res_pos2: # we want to look at distance between two DIFF residues (ensure that r1 and r2 aren't the same)
                        if targ_prot in dict_of_distances:
                            
                            if res_pos1 + '-' + res_pos2 in dict_of_distances[targ_prot] or res_pos2 + '-' + res_pos1 in dict_of_distances[targ_prot]:
                                continue # we've already stored this pair of residues (note: res1-res2 distance is same as res2-res1 distance)
                        pdb_chains_in_common = [pc for pc in dnds_info_with_pdb_and_auth_info[targ_prot][res_pos1].keys() if pc in dnds_info_with_pdb_and_auth_info[targ_prot][res_pos2].keys()]

                        if len(pdb_chains_in_common) == 0: # no chains in common, so we can't find the distance between these two residues
                            continue
                            
                        if targ_prot not in dict_of_distances:
                            dict_of_distances[targ_prot] = {}

                        dict_of_distances[targ_prot][res_pos1+ '-' + res_pos2] = memo_distances_dict[targ_prot][res_pos1+ '-' + res_pos2]

    
    return dict_of_distances

def get_auth_from_seq_pos(dict_of_auth_to_seq_mappings, seq_res_pos):
    ''' Get the auth position (in terms of the pdb chain) corresponding to the given residue position in the uniprot protein sequence
    '''
    for auth_res, seq_res in dict_of_auth_to_seq_mappings.items():  
        if seq_res == int(seq_res_pos):
            return int(auth_res)
    return -100


def get_min_atomic_distance_between_2_residues(model, chain, auth_res1, auth_res2):
    # try:
    #     structure = parser.get_structure(PDBid, os.path.join(pdb_fis_dir, PDBid.lower() + '.cif'))
    # except:
    #     print(f'{PDBid} PDB file Construction Error')
        
    # model = structure[0]

    residue1 = model[chain][auth_res1]
    residue2 = model[chain][auth_res2]
    
    res1_num = str(residue1.get_full_id()[3][1])                             
    res2_num = str(residue2.get_full_id()[3][1])
    atomic_distances = []
    # compute distance between all atoms on the two residues (take the minimum distance)
    for atom1 in residue1:
        for atom2 in residue2:
            distance = atom1 - atom2
            atomic_distances.append(distance)

    min_dist = 1.0 * min(atomic_distances) # add *1.0 so that it converts to a float64 type (json-compatible) rather than a float32 type (not compatible with json)

    return min_dist

def main():
    script_dir = osp.dirname(__file__)

    # file contianing all dnds values for each residue type
    dnds_split_by_prots_fi = osp.join(script_dir, '..', 'data', 'dnds', 'dnds_for_residue_categories_species_tree', 'targ_prot_res_well_defined_dnds_vals.json')
    
    # Auth to uniprot seq mapping files

    endo_auth_to_seq_fi = osp.join(script_dir, '..', 'data', 'exo_and_endo', 'endo_pdb_auth_to_uniprot_seq.json')
    exo_auth_to_seq_fi = osp.join(script_dir, '..', 'data', 'exo_and_endo', 'exo_pdb_auth_to_uniprot_seq.json')

    
    # PDB information
    pdb_resolution_fi = osp.join(script_dir, '..', 'data', 'PDB', 'resolu.idx')
    pdb_fis_dir = osp.join(script_dir, '..', 'data', 'pdb_files')

    # new dir to output files
    new_dir = osp.join(script_dir, '..', 'data', 'dnds', 'dnds_for_residue_categories_species_tree', 'distances_btwn_residues_in_diff_dnds_categories')

    if not osp.exists(new_dir):
        os.makedirs(new_dir)

    
    dnds_split_by_prots = load_json(dnds_split_by_prots_fi)

    endo_auth_to_seq_dict = load_json(endo_auth_to_seq_fi)

    exo_auth_to_seq_dict = load_json(exo_auth_to_seq_fi)
   


    # output dnds info with pdb chains and auth residues OR load if already there

    all_exo_dnds_with_pdb_auth_info_outfi = osp.join(new_dir, 'all_exo_unsplit_dnds_info_with_pdb_and_auth_info.json')
    all_endo_dnds_with_pdb_auth_info_outfi = osp.join(new_dir, 'all_endo_unsplit_dnds_info_with_pdb_and_auth_info.json')


    exo_specific_dnds_with_pdb_auth_info_outfi = osp.join(new_dir, 'exo_specific_unsplit_dnds_info_with_pdb_and_auth_info.json')
    endo_specific_dnds_with_pdb_auth_info_outfi = osp.join(new_dir, 'endo_specific_unsplit_dnds_info_with_pdb_and_auth_info.json')
    mimicked_dnds_with_pdb_auth_info_outfi = osp.join(new_dir, 'mimicked_unsplit_dnds_info_with_pdb_and_auth_info.json')

    if not osp.exists(all_exo_dnds_with_pdb_auth_info_outfi) or not osp.exists(all_endo_dnds_with_pdb_auth_info_outfi):
        print('Making mapping files for all_exo and all_endo')
        all_exo_dnds_info_with_pdb_and_auth_info = store_res_dnds_and_auth_pos(exo_auth_to_seq_dict, 'all_exo', dnds_split_by_prots)
        all_endo_dnds_info_with_pdb_and_auth_info = store_res_dnds_and_auth_pos(endo_auth_to_seq_dict, 'all_endo', dnds_split_by_prots)
        
        with open(all_exo_dnds_with_pdb_auth_info_outfi, 'w') as aexo, open(all_endo_dnds_with_pdb_auth_info_outfi, 'w') as aeno:
            json.dump(all_exo_dnds_info_with_pdb_and_auth_info, aexo)
            json.dump(all_endo_dnds_info_with_pdb_and_auth_info, aeno)

    else:
        all_exo_dnds_info_with_pdb_and_auth_info = {}
        all_endo_dnds_info_with_pdb_and_auth_info = {}

        with open(all_exo_dnds_with_pdb_auth_info_outfi) as aexo, open(all_endo_dnds_with_pdb_auth_info_outfi) as aeno:
            all_exo_dnds_info_with_pdb_and_auth_info = json.load(aexo)
            all_endo_dnds_info_with_pdb_and_auth_info = json.load(aeno)
    
    

    if not osp.exists(exo_specific_dnds_with_pdb_auth_info_outfi) or not osp.exists(endo_specific_dnds_with_pdb_auth_info_outfi) or not osp.exists(mimicked_dnds_with_pdb_auth_info_outfi):
        print('Making mapping files for exo_specific, mimicked, and endo_specific')
        exo_specific_dnds_info_with_pdb_and_auth_info = store_res_dnds_and_auth_pos(exo_auth_to_seq_dict, 'exo', dnds_split_by_prots)
        endo_specific_dnds_info_with_pdb_and_auth_info = store_res_dnds_and_auth_pos(endo_auth_to_seq_dict, 'endo', dnds_split_by_prots)
        mimicked_dnds_info_with_pdb_and_auth_info = store_res_dnds_and_auth_pos_for_mimicked(exo_auth_to_seq_dict, endo_auth_to_seq_dict, 'mimicry', dnds_split_by_prots)
        
        with open(exo_specific_dnds_with_pdb_auth_info_outfi, 'w') as exosp, open(endo_specific_dnds_with_pdb_auth_info_outfi, 'w') as endosp, open(mimicked_dnds_with_pdb_auth_info_outfi, 'w') as mim:
            json.dump(exo_specific_dnds_info_with_pdb_and_auth_info, exosp)
            json.dump(endo_specific_dnds_info_with_pdb_and_auth_info, endosp)
            json.dump(mimicked_dnds_info_with_pdb_and_auth_info, mim)

    else:
        exo_specific_dnds_info_with_pdb_and_auth_info = {}
        endo_specific_dnds_info_with_pdb_and_auth_info = {}
        mimicked_dnds_info_with_pdb_and_auth_info = {}

        with open(exo_specific_dnds_with_pdb_auth_info_outfi) as exosp, open(endo_specific_dnds_with_pdb_auth_info_outfi) as endosp, open(mimicked_dnds_with_pdb_auth_info_outfi) as mim:
            exo_specific_dnds_info_with_pdb_and_auth_info = json.load(exosp)
            endo_specific_dnds_info_with_pdb_and_auth_info = json.load(endosp)
            mimicked_dnds_info_with_pdb_and_auth_info = json.load(mim)

    


    pdb_resolution_dict = load_pdb_resolution(pdb_resolution_fi)

    parser = MMCIFParser(QUIET=True)


    # initialize memoized dict for storing distances between residues in a pdb structure, so that we don't need to recompute if the same pair of residues are compared again
    memo_distances_fi = osp.join(new_dir, 'memoized_atomic_distances_btwn_residues.json')


    # compute and output distances between residue pairs 
    
    all_exo_distances_outfi = osp.join(new_dir, 'all_exo_res_distances.json')
    all_endo_distances_outfi = osp.join(new_dir, 'all_endo_res_distances.json')

    
    exo_specific_distances_outfi = osp.join(new_dir, 'exo_specific_res_distances.json')
    endo_specific_distances_outfi = osp.join(new_dir, 'endo_specific_res_distances.json')
    mimicked_distances_outfi = osp.join(new_dir, 'mimicked_res_distances.json')
    
    
    all_exo_dict_of_distances = get_distance_between_residues(memo_distances_fi, pdb_resolution_dict, 'all_exo', all_exo_dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser)
    with open(all_exo_distances_outfi, 'w') as aexdo:
        json.dump(all_exo_dict_of_distances, aexdo)

    
    all_endo_dict_of_distances = get_distance_between_residues(memo_distances_fi, pdb_resolution_dict, 'all_endo', all_endo_dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser)
    with open(all_endo_distances_outfi, 'w') as aendo:
        json.dump(all_endo_dict_of_distances, aendo)

    

    mimicked_dict_of_distances = get_distance_between_residues(memo_distances_fi, pdb_resolution_dict, 'mimicry', mimicked_dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser)
    with open(mimicked_distances_outfi, 'w') as mimdo:
        json.dump(mimicked_dict_of_distances, mimdo)
    
    
    exo_specific_dict_of_distances = get_distance_between_residues_by_populating_from_all_res(all_exo_distances_outfi, pdb_resolution_dict, 'exo', exo_specific_dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser)
    with open(exo_specific_distances_outfi, 'w') as exspdo:
        json.dump(exo_specific_dict_of_distances, exspdo)

  
    endo_specific_dict_of_distances = get_distance_between_residues_by_populating_from_all_res(all_endo_distances_outfi, pdb_resolution_dict, 'endo', endo_specific_dnds_info_with_pdb_and_auth_info, pdb_fis_dir, parser)
    with open(endo_specific_distances_outfi, 'w') as enspdo:
        json.dump(endo_specific_dict_of_distances, enspdo)
    
    
   
      

if __name__ == '__main__':
    main()
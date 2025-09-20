import os
import os.path as osp
import json
from dictionary_tools import load_json, dump_json
import numpy as np
import random
from collections import Counter


# Get per-protein o/e ratios -- treat per-protein cluster as a subgraph and compute o/e over 10,000 trials

def get_diff_node_type_counts_per_prot(nodes_anno):
	# store edge counts for each tp separately
	node_counts_all_tp = {}
	for tp in nodes_anno:
		node_counts = {'pos': 0, 'neg': 0, 'other': 0}
		for n in nodes_anno[tp]:
			node_ann = nodes_anno[tp][n]
			node_counts[node_ann] += 1
		node_counts_all_tp[tp] = node_counts
	
	return node_counts_all_tp

def get_diff_edge_type_counts_per_prot(edges_anno):
	# store edge counts for each tp separately
	edge_counts_all_tp = {}
	for tp in edges_anno:
		edge_counts = {'pos': 0, 'neg': 0, 'pos_neg': 0, 'pos_other': 0, 'neg_other': 0, 'other': 0}
		for pair_of_nodes in edges_anno[tp]:
			edge_ann = edges_anno[tp][pair_of_nodes]
			edge_counts[edge_ann] += 1
		edge_counts_all_tp[tp] = edge_counts
	# print(edge_counts)
	# print(total_edges)
	return edge_counts_all_tp

def shuffle_node_annos(nodes_anno): 
	nodes_anno_shuffled_all_prots = {}

	for tp in nodes_anno:
		orig_nodes_anno = nodes_anno[tp]
		orig_annos_counts = Counter(orig_nodes_anno[r] for r in orig_nodes_anno) # count to verify 

		# shuffling values
		temp = list(orig_nodes_anno.values())
		random.shuffle(temp)
		
		# reassigning to keys
		shuffled_nodes_anno = dict(zip(orig_nodes_anno, temp))
		shuffled_annos_counts = Counter(orig_nodes_anno[r] for r in orig_nodes_anno) # count to verify 

		# print(orig_nodes_anno, shuffled_nodes_anno)
		if orig_annos_counts!=shuffled_annos_counts: # count to verify 
			print('False')

		nodes_anno_shuffled_all_prots[tp] = shuffled_nodes_anno
	return nodes_anno_shuffled_all_prots

def relabel_edges_in_graph_based_on_shuffled_annos(nodes_anno_shuffled_all_prots, orig_edges_anno):
	new_edges_anno = {}
	num_edges_all_orig = 0

	for tp in orig_edges_anno:
		new_edges_anno[tp] = {}
		for pair in orig_edges_anno[tp]:
			num_edges_all_orig+=1
			node1 = pair.split('-')[0]
			node2 = pair.split('-')[1]
			# print(nodes_anno_shuffled_all_prots[tp])
			if nodes_anno_shuffled_all_prots[tp][node1] == 'pos' and nodes_anno_shuffled_all_prots[tp][node2] == 'pos': # two +vely selected residues are nearest neighbours
				new_edges_anno[tp][node1 + '-' + node2] = 'pos'
			elif nodes_anno_shuffled_all_prots[tp][node1] == 'neg' and nodes_anno_shuffled_all_prots[tp][node2] == 'neg': # two -vely selected residues are nearest neighbours
				new_edges_anno[tp][node1 + '-' + node2] = 'neg'
			elif nodes_anno_shuffled_all_prots[tp][node1] == 'pos' and nodes_anno_shuffled_all_prots[tp][node2] == 'neg': # one +vely one -vely selected residue are nns
				new_edges_anno[tp][node1 + '-' + node2] = 'pos_neg'
			elif nodes_anno_shuffled_all_prots[tp][node1] == 'neg' and nodes_anno_shuffled_all_prots[tp][node2] == 'pos': # one +vely one -vely selected residue are nns
				new_edges_anno[tp][node1 + '-' + node2] = 'pos_neg'
			elif (nodes_anno_shuffled_all_prots[tp][node1] == 'pos' and nodes_anno_shuffled_all_prots[tp][node2] == 'other') or (nodes_anno_shuffled_all_prots[tp][node1] == 'other' and nodes_anno_shuffled_all_prots[tp][node2] == 'pos'): # one +vely selected residue one other are nns
				new_edges_anno[tp][node1 + '-' + node2] = 'pos_other'
			elif (nodes_anno_shuffled_all_prots[tp][node1] == 'neg' and nodes_anno_shuffled_all_prots[tp][node2] == 'other') or (nodes_anno_shuffled_all_prots[tp][node1] == 'other' and nodes_anno_shuffled_all_prots[tp][node2] == 'neg'): # one -vely selected residue one other are nns
				new_edges_anno[tp][node1 + '-' + node2] = 'neg_other'
			else:
				new_edges_anno[tp][node1 + '-' + node2] = 'other'
	num_edges_new = 0
	for tp in new_edges_anno:
		num_edges_new += len(new_edges_anno[tp])

	
	return new_edges_anno

def get_observed_over_mean_expected_ratio(obs_edge_counts, expected_edge_counts, tps_with_at_least_n_pos_and_n_neg):
	# Get o/mean(e_perm), first get mean of expected edges across the permutations, then divide o by this mean expected edges value
	mean_exp = {}
	obs_over_exp_ratio = {}
	obs_over_exp_ratio_fraction = {}
	pvals = {} # these p-vals indicate whether observed is significantly different from expected
	for tp in obs_edge_counts:
		if tp not in tps_with_at_least_n_pos_and_n_neg: continue
		mean_exp[tp] = {}
		obs_over_exp_ratio[tp] = {}
		obs_over_exp_ratio_fraction[tp] = {}
		pvals[tp] = {}
		for label in obs_edge_counts[tp]:
			exp_edge_counts_over_trials = expected_edge_counts[tp][label] # get the mean expected edge counts over the trials
			mean_exp_edge_counts = np.mean(exp_edge_counts_over_trials)
			mean_exp[tp][label] = mean_exp_edge_counts
			ratio = str(obs_edge_counts[tp][label]) + '/' + str(mean_exp_edge_counts)
			# mean_exp_edge_counts += 0.5
			if mean_exp_edge_counts == 0:
				if obs_edge_counts[tp][label] == 0: # if both denominator and numerator ==0, set o/e to 0
					fraction = 0
					fraction_rounded = 0.0
				else: # denominator is 0, numerator isn't --> undefined
					print('Zero!')
					fraction = -1
			else:
				fraction = obs_edge_counts[tp][label]/mean_exp_edge_counts
				fraction_rounded = round((obs_edge_counts[tp][label]/mean_exp_edge_counts), 2)

			obs_over_exp_ratio[tp][label] = ratio + ' (' + str(fraction_rounded) + ')'
			obs_over_exp_ratio_fraction[tp][label] = fraction
		
			pval = (np.sum(np.array(exp_edge_counts_over_trials) >= obs_edge_counts[tp][label]) + 1) / (len(exp_edge_counts_over_trials) + 1)
			
			pvals[tp][label] = pval
	return mean_exp, obs_over_exp_ratio, obs_over_exp_ratio_fraction, pvals


def get_mean_of_observed_over_expected_ratio(obs_edge_counts, expected_edge_counts, tps_with_at_least_n_pos_and_n_neg):
	# get mean(o/e) ratio 
	# Output mean of these o/e ratios across the permutations
	mean_obs_over_exp_ratio = {}
	for tp in obs_edge_counts:
		if tp not in tps_with_at_least_n_pos_and_n_neg: continue
		mean_obs_over_exp_ratio[tp] = {}
		for label in obs_edge_counts[tp]:
			
				
			exp_edge_counts_over_trials = expected_edge_counts[tp][label] # get the mean expected edge counts over the trials
			o_e_perm_list = []
			for c in exp_edge_counts_over_trials:

				if c == 0: 
					if obs_edge_counts[tp][label] == 0:
						o_e_perm_list.append(0)
					else: # denominator = 0, but numerator isn't zero --> skip (because undefined)
						continue 
				else:
					o_e_perm_list.append(obs_edge_counts[tp][label]/c)
			
			mean_oe = np.mean(o_e_perm_list) 
			median_oe = np.median(o_e_perm_list)
			
			mean_obs_over_exp_ratio[tp][label] = (mean_oe, median_oe)

	return mean_obs_over_exp_ratio





def main():
	script_dir = osp.dirname(__file__)
	dir_of_node_and_edge_annos = osp.join(script_dir, '..', 'data', 'contact_graphs', 'graphviz')
	outdir = osp.join(script_dir, '..', 'data', 'contact_graphs', 'graphviz', 'interrel2_per_prot_all')
	print('Starting')
	if not osp.exists(outdir):
		os.makedirs(outdir)
	num_trials = 10000
	restype = 'whole_interface'
		
	print(f'\nRestype: {restype}')
	nodes_anno_fi = osp.join(dir_of_node_and_edge_annos, restype + '_nodes_anno.json')
	edges_anno_fi = osp.join(dir_of_node_and_edge_annos, restype + '_edges_anno_split_other.json')

	nodes_anno = load_json(nodes_anno_fi)
	edges_anno = load_json(edges_anno_fi)
	
	
	
	orig_edge_counts_fi = osp.join(outdir, restype + '_observed_edge_counts.json')

	if not osp.exists(orig_edge_counts_fi):
		print('\tGetting edge counts (original, observed). . .')
		orig_edge_counts = get_diff_edge_type_counts_per_prot(edges_anno)
		dump_json(orig_edge_counts, orig_edge_counts_fi)
	else:

		orig_edge_counts = load_json(orig_edge_counts_fi)


	edgetypes = ['pos', 'neg', 'pos_neg', 'pos_other', 'neg_other', 'other']
	
	orig_node_counts = get_diff_node_type_counts_per_prot(nodes_anno)
	
	n = 5

	tps_with_at_least_n_pos_and_n_neg = []
	for tp in orig_node_counts:
		if orig_node_counts[tp]['pos'] >=n and orig_node_counts[tp]['neg'] >=n :
			tps_with_at_least_n_pos_and_n_neg.append(tp)
	print('node', len(tps_with_at_least_n_pos_and_n_neg))
	

	expected_edge_counts = {}
	expected_edges_fi = osp.join(outdir, restype + '_shuffled_node_annos_new_edges_repeated_trials.json') # store edge counts for each permutation trial
	
	if not osp.exists(expected_edges_fi):

		# Get null distribution of expected edges 
		for t in range(num_trials): 
			# print('\tShuffling node annotations . . .')
			nodes_anno_shuffled_all_prots = shuffle_node_annos(nodes_anno)

			# print('\tRelabeling edges in graph based on shuffled node annos. . .')
			
			new_edges_anno = relabel_edges_in_graph_based_on_shuffled_annos(nodes_anno_shuffled_all_prots, edges_anno)
			
			# print('\tGetting edge counts for new edge annos (expected). . .')
			new_edge_counts = get_diff_edge_type_counts_per_prot(new_edges_anno)

			# store the expected edge counts for each type of edge in this trial
			for tp in new_edge_counts:
				for label in new_edge_counts[tp]:
					if tp not in expected_edge_counts:
						expected_edge_counts[tp] = {}
					if label not in expected_edge_counts[tp]:
						expected_edge_counts[tp][label] = []
					expected_edge_counts[tp][label].append(new_edge_counts[tp][label])

		dump_json(expected_edge_counts, expected_edges_fi)
	else:
		expected_edge_counts = load_json(expected_edges_fi)


	'''Compute O/E ratios (2 methods -- note that both give pretty much the same thing)
		(Method 1) either we compute the mean E value from the perm trials (E_perm) and then thake O/E_mean
		OR 
		(Method 2) we take the average of all O/E_perm values
	
		Since some target proteins won't have many or any pos and/or neg residues, we will only consider proteins
		that have at least n pos and at least n neg for a specific interface
	'''
	# (Method 1) Get O/E_mean, where E_mean the the mean expected edge count (computed across all permutation trials)
	mean_exp, obs_over_exp_ratio, obs_over_exp_ratio_fraction, pvals= get_observed_over_mean_expected_ratio(orig_edge_counts, expected_edge_counts, tps_with_at_least_n_pos_and_n_neg)
	
	# (Method 2) Get mean(O/E_perm), where E_perm is expected edge count of each permutation trial.
	mean_oe = get_mean_of_observed_over_expected_ratio(orig_edge_counts, expected_edge_counts, tps_with_at_least_n_pos_and_n_neg)

	

	# Output these ratios:
	dict_of_oe_ratios = {'o_e_mean': obs_over_exp_ratio_fraction, 'mean_o_e_permut': mean_oe}
	oe_ratios_fi = osp.join(outdir, restype + '_oe_ratios.json')
	dump_json(dict_of_oe_ratios, oe_ratios_fi)

	edgetypes = ['pos', 'neg', 'pos_neg', 'pos_other', 'neg_other', 'other']
	for et in edgetypes:
		# Take mean O/E across permutations for each protein
		mn = np.mean([dict_of_oe_ratios['mean_o_e_permut'][tp][et][0] for tp in dict_of_oe_ratios['mean_o_e_permut']])
		md = np.median([dict_of_oe_ratios['mean_o_e_permut'][tp][et][0] for tp in dict_of_oe_ratios['mean_o_e_permut']])
		print(et, mn, md)

		# Take median O/E across permutations for each protein
		mn = np.mean([dict_of_oe_ratios['mean_o_e_permut'][tp][et][1] for tp in dict_of_oe_ratios['mean_o_e_permut']])
		md = np.median([dict_of_oe_ratios['mean_o_e_permut'][tp][et][1] for tp in dict_of_oe_ratios['mean_o_e_permut']])
		print(et, mn, md)

		

if __name__ == '__main__':
	main()
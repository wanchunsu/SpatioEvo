import os
import os.path as osp
import json
from dictionary_tools import load_json, dump_json
import numpy as np
import random
from collections import Counter
from scipy import stats

# Compute O/E contact count ratios (across 10,000 inter-protein node-label randomization trials) for different interfaces and sub-interfaces


def get_diff_edge_type_counts(edges_anno):
	total_edges = 0
	edge_counts = {'pos': 0, 'neg': 0, 'pos_neg': 0, 'pos_other': 0, 'neg_other': 0, 'other': 0}
	for tp in edges_anno:
		for pair_of_nodes in edges_anno[tp]:
			edge_ann = edges_anno[tp][pair_of_nodes]
			edge_counts[edge_ann] += 1
			total_edges += 1
	# print(edge_counts)
	# print(total_edges)
	return edge_counts

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

def get_observed_over_mean_expected_ratio(obs_edge_counts, expected_edge_counts):
	# Get o/mean(e_perm), first get mean of expected edges across the permutations, then divide o by this mean expected edges value
	mean_exp = {}
	obs_over_exp_ratio = {}
	pvals = {} # these p-vals indicate whether observed is significantly different from expected
	for label in obs_edge_counts:
		exp_edge_counts_over_trials = expected_edge_counts[label] # get the mean expected edge counts over the trials
		mean_exp_edge_counts = np.mean(exp_edge_counts_over_trials)
		mean_exp[label] = mean_exp_edge_counts
		ratio = str(obs_edge_counts[label]) + '/' + str(mean_exp_edge_counts)
		if mean_exp_edge_counts == 0:
			fraction = 0
		else:
			fraction = round((obs_edge_counts[label]/mean_exp_edge_counts), 2)
		obs_over_exp_ratio[label] = ratio + ' (' + str(fraction) + ')'
		
		# get pval
		pval = perm_test(obs_edge_counts[label], exp_edge_counts_over_trials)
		pvals[label] = pval

	return mean_exp, obs_over_exp_ratio, pvals


def get_mean_of_observed_over_expected_ratio(obs_edge_counts, expected_edge_counts):
	# get o/e_perm ratio for each permutation (note that o stays constant, only e changes in each permutation)
	# Output mean of these o/e_perm ratios across the permutations
	mean_obs_over_exp_ratio = {}
	
	for label in obs_edge_counts:
		print(label)
		exp_edge_counts_over_trials = expected_edge_counts[label] # get the mean expected edge counts over the trials
		o_e_perm_list = []
		for c in exp_edge_counts_over_trials:
			if c == 0:
				print('UNDEFINED!') # Verified that this doesn't happen, i.e., we don't have cases where the expected # of a contact is 0 (undefined)
		
			else:
				o_e_perm_list.append(obs_edge_counts[label]/c)

		
		mean_oe = np.mean(o_e_perm_list)
		
		mean_obs_over_exp_ratio[label] = round(mean_oe, 2)

	return mean_obs_over_exp_ratio


def diff_from_1(data):
    t_stat, p_value = stats.ttest_1samp(data, popmean=1)
    print("p-value:", p_value)
    return p_value

def get_distribution_of_oe(obs_edge_counts, expected_edge_counts):
	for label in obs_edge_counts:
	oe_distr = observed_data[label]/np.array(expected_data[label])
    p = diff_from_1(oe_distr)
    return oe_distr, p


def main():
	script_dir = osp.dirname(__file__)
	dir_of_node_and_edge_annos = osp.join(script_dir, '..', 'data', 'contact_graphs', 'graphviz')
	outdir = osp.join(script_dir, '..', 'data', 'contact_graphs', 'graphviz', 'interrel2')
	print('Starting')
	if not osp.exists(outdir):
		os.makedirs(outdir)
	num_trials = 10000
	for restype in ['exo_specific', 'mimicked', 'endo_specific', 'all_exo', 'all_endo', 'surface', 'whole_interface']:
		
		print(f'\nRestype: {restype}')
		nodes_anno_fi = osp.join(dir_of_node_and_edge_annos, restype + '_nodes_anno.json')
		edges_anno_fi = osp.join(dir_of_node_and_edge_annos, restype + '_edges_anno_split_other.json')

		nodes_anno = load_json(nodes_anno_fi)
		edges_anno = load_json(edges_anno_fi)
		
		
		orig_edge_counts_fi = osp.join(outdir, restype + '_observed_edge_counts.json')

		if not osp.exists(orig_edge_counts_fi):
			print('\tGetting edge counts (original, observed). . .')
			orig_edge_counts = get_diff_edge_type_counts(edges_anno)
			dump_json(orig_edge_counts, orig_edge_counts_fi)
		else:

			orig_edge_counts = load_json(orig_edge_counts_fi)
		
		# print(f'\t\tOriginal (observed) edge counts: {orig_edge_counts}')

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
				new_edge_counts = get_diff_edge_type_counts(new_edges_anno)

				# store the expected edge counts for each type of edge in this trial
				for label in new_edge_counts:
					if label not in expected_edge_counts:
						expected_edge_counts[label] = []
					expected_edge_counts[label].append(new_edge_counts[label])

			dump_json(expected_edge_counts, expected_edges_fi)
		else:
			expected_edge_counts = load_json(expected_edges_fi)


		'''Compute O/E ratios (2 methods -- note that both give pretty much the same thing)
			(Method 1) either we compute the mean E value from the perm trials (E_perm) and then thake O/E_mean
			OR 
			(Method 2) we take the average of all O/E_perm values
		'''

		# (Method 1) Get O/E_mean, where E_mean the the mean expected edge count (computed across all permutation trials)
		mean_exp, obs_over_exp_ratio, pvals= get_observed_over_mean_expected_ratio(orig_edge_counts, expected_edge_counts)
		
		# (Method 2) Get mean(O/E_perm), where E_perm is expected edge count of each permutation trial.
		mean_oe = get_mean_of_observed_over_expected_ratio(orig_edge_counts, expected_edge_counts)
		distr, pval = get_distribution_of_oe(obs_edge_counts, expected_edge_counts)

		print(f'\t O/E_mean: {obs_over_exp_ratio}; mean(O/E_perm): {mean_oe}') # both give pretty much the same ratios! :)

		# Output these ratios:
		dict_of_oe_ratios = {'o_e_mean': obs_over_exp_ratio, 'mean_o_e_permut': mean_oe}
		oe_ratios_fi = osp.join(outdir, restype + '_oe_ratios.json')
		if not osp.exists(oe_ratios_fi):
			dump_json(dict_of_oe_ratios, oe_ratios_fi)


if __name__ == '__main__':
	main()
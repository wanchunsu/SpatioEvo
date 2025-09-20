import os
import os.path as osp
import json
from dictionary_tools import load_json, dump_json
import numpy as np



# label nodes and edges of contact graph

def populate_data(all_distances_dict, distances_to_add):
	diffs = 0
	num_distances = 0
	for tp in distances_to_add:
		if tp not in all_distances_dict:
			all_distances_dict[tp] = {}
		for res_pair in distances_to_add[tp]: # update if necessary with smaller dist
			res_pair_rev = res_pair.split('-')[1] + '-' + res_pair.split('-')[0]

			if res_pair in all_distances_dict[tp]:
				existing_dist = all_distances_dict[tp][res_pair]
				curr_dist = distances_to_add[tp][res_pair]
				# print(existing_dist, curr_dist)

				if curr_dist >= 5 and existing_dist >=5:
					continue
				if curr_dist < 5 and existing_dist <5:
					continue
				else:
					# print(tp, res_pair, existing_dist, curr_dist)
					all_distances_dict[tp][res_pair] = min(existing_dist, curr_dist)
					if curr_dist < existing_dist:
						diffs +=1

			elif res_pair_rev in all_distances_dict[tp]: # update if necessary with smaller dist
				existing_dist = all_distances_dict[tp][res_pair_rev]
				curr_dist = distances_to_add[tp][res_pair]
				# print(existing_dist, curr_dist)

				if curr_dist >= 5 and existing_dist >=5:
					continue
				if curr_dist < 5 and existing_dist <5:
					continue
				else:
					# print(tp, res_pair, existing_dist, curr_dist)
					all_distances_dict[tp][res_pair_rev] = min(existing_dist, curr_dist)
					if curr_dist < existing_dist:
						diffs +=1
			else:
				all_distances_dict[tp][res_pair] = distances_to_add[tp][res_pair]


	return all_distances_dict
	

def combine_interface_distances_dict(all_exo, all_endo, mimicked, btwn_exo_spec_and_endo_spec):
	all_distances_dict = {}

	print('\nAdding all_exo ...')
	all_distances_dict = populate_data(all_distances_dict, all_exo)

	print('\nAdding all_endo ...')
	all_distances_dict = populate_data(all_distances_dict, all_endo)

	print('\nAdding mimicked ...')
	all_distances_dict = populate_data(all_distances_dict, mimicked)

	print('\nAdding btwn exo and endo...')
	all_distances_dict = populate_data(all_distances_dict, btwn_exo_spec_and_endo_spec)
	
	# printing total num RR distances
	num_dist = 0
	for t in all_distances_dict:
		num_dist += len(all_distances_dict[t])

	print(f'Total num distances: {num_dist}, total num proteins: {len(all_distances_dict)}')

	return all_distances_dict



def populate_nodes(combined_nodes_dict, nodes_to_add):

	for tp in nodes_to_add:
		if tp not in combined_nodes_dict:
			combined_nodes_dict[tp] = {}
		for node in nodes_to_add[tp]:
			if node in combined_nodes_dict[tp]:
				if combined_nodes_dict[tp][node] != nodes_to_add[tp][node]:
					print(f'Error! {tp}, {node}, {combined_nodes_dict[tp][node]}, {nodes_to_add[tp][node]}')
				
			else:
				combined_nodes_dict[tp][node] = nodes_to_add[tp][node]
				# print(f'Node added! {tp}_{node}')
		combined_nodes_dict[tp] = dict(sorted(combined_nodes_dict[tp].items(), key=lambda item: int(item[0])))

	return combined_nodes_dict


def combine_interface_nodes(all_exo_nodes, all_endo_nodes, mimicked_nodes, btwn_exo_endo_nodes):

	combined_nodes_dict = {}

	print('\nAdding all_exo ...')
	combined_nodes_dict = populate_nodes(combined_nodes_dict, all_exo_nodes)

	print('\nAdding all_endo ...')
	combined_nodes_dict = populate_nodes(combined_nodes_dict, all_endo_nodes)

	print('\nAdding mimicked ...')
	combined_nodes_dict = populate_nodes(combined_nodes_dict, mimicked_nodes)

	print('\nAdding btwn exo and endo nodes...')
	combined_nodes_dict = populate_nodes(combined_nodes_dict, btwn_exo_endo_nodes)


	num_nodes = 0
	for t in combined_nodes_dict:
		num_nodes += len(combined_nodes_dict[t])
	print(f'Total num nodes: {num_nodes}, total num proteins: {len(combined_nodes_dict)}')


	return combined_nodes_dict


def categorize_edges_in_contact_graph(res_distances_dict, nodes_dnds_categories, outfi):
	# Go through <restype>_res_distances.json (protein: node1-node2: distance) and extract edges (i.e. distances < 5Angstroms)
	# Check the color of the two nodes in contact

	# Output: node1-node2: color

	edges_dnds_categories = {}
	for tp in res_distances_dict:
		if tp not in edges_dnds_categories:
			edges_dnds_categories[tp] = {}
		for node1_node2 in res_distances_dict[tp]:
			node1 = node1_node2.split('-')[0]
			node2 = node1_node2.split('-')[1]
			if (node1 + '-' + node2 not in edges_dnds_categories[tp]) and (node2 + '-' + node1 not in edges_dnds_categories[tp]):
				# store edge if the
				if res_distances_dict[tp][node1_node2] < 5.0: # nearest neighbour cutoff: 5 Angstroms
					if nodes_dnds_categories[tp][node1] == 'pos' and nodes_dnds_categories[tp][node2] == 'pos': # two +vely selected residues are nearest neighbours
						edges_dnds_categories[tp][node1 + '-' + node2] = 'pos'
					elif nodes_dnds_categories[tp][node1] == 'neg' and nodes_dnds_categories[tp][node2] == 'neg': # two -vely selected residues are nearest neighbours
						edges_dnds_categories[tp][node1 + '-' + node2] = 'neg'
					elif nodes_dnds_categories[tp][node1] == 'pos' and nodes_dnds_categories[tp][node2] == 'neg': # one +vely one -vely selected residue are nns
						edges_dnds_categories[tp][node1 + '-' + node2] = 'pos_neg'
					elif nodes_dnds_categories[tp][node1] == 'neg' and nodes_dnds_categories[tp][node2] == 'pos': # one +vely one -vely selected residue are nns
						edges_dnds_categories[tp][node1 + '-' + node2] = 'pos_neg'
					else:
						edges_dnds_categories[tp][node1 + '-' + node2] = 'other'
	# print(edges_dnds_categories)
	dump_json(edges_dnds_categories, outfi)
	return edges_dnds_categories


def categorize_edges_in_contact_graph2(res_distances_dict, nodes_dnds_categories, outfi):
	# Go through <restype>_res_distances.json (protein: node1-node2: distance) and extract edges (i.e. distances < 5Angstroms)
	# Check the color of the two nodes in contact

	# Output: node1-node2: color

	edges_dnds_categories = {}
	for tp in res_distances_dict:
		if tp not in edges_dnds_categories:
			edges_dnds_categories[tp] = {}
		for node1_node2 in res_distances_dict[tp]:
			node1 = node1_node2.split('-')[0]
			node2 = node1_node2.split('-')[1]
			if (node1 + '-' + node2 not in edges_dnds_categories[tp]) and (node2 + '-' + node1 not in edges_dnds_categories[tp]):
				# store edge if the
				if res_distances_dict[tp][node1_node2] < 5.0: # nearest neighbour cutoff: 5 Angstroms
					if nodes_dnds_categories[tp][node1] == 'pos' and nodes_dnds_categories[tp][node2] == 'pos': # two +vely selected residues are nearest neighbours
						edges_dnds_categories[tp][node1 + '-' + node2] = 'pos'
					elif nodes_dnds_categories[tp][node1] == 'neg' and nodes_dnds_categories[tp][node2] == 'neg': # two -vely selected residues are nearest neighbours
						edges_dnds_categories[tp][node1 + '-' + node2] = 'neg'
					elif nodes_dnds_categories[tp][node1] == 'pos' and nodes_dnds_categories[tp][node2] == 'neg': # one +vely one -vely selected residue are nns
						edges_dnds_categories[tp][node1 + '-' + node2] = 'pos_neg'
					elif nodes_dnds_categories[tp][node1] == 'neg' and nodes_dnds_categories[tp][node2] == 'pos': # one +vely one -vely selected residue are nns
						edges_dnds_categories[tp][node1 + '-' + node2] = 'pos_neg'
					elif (nodes_dnds_categories[tp][node1] == 'pos' and nodes_dnds_categories[tp][node2] == 'other') or (nodes_dnds_categories[tp][node1] == 'other' and nodes_dnds_categories[tp][node2] == 'pos'): # one +vely selected residue one other are nns
						edges_dnds_categories[tp][node1 + '-' + node2] = 'pos_other'
					elif (nodes_dnds_categories[tp][node1] == 'neg' and nodes_dnds_categories[tp][node2] == 'other') or (nodes_dnds_categories[tp][node1] == 'other' and nodes_dnds_categories[tp][node2] == 'neg'): # one -vely selected residue one other are nns
						edges_dnds_categories[tp][node1 + '-' + node2] = 'neg_other'
					else: # two other are nns
						edges_dnds_categories[tp][node1 + '-' + node2] = 'other'
	# print(edges_dnds_categories)
	dump_json(edges_dnds_categories, outfi)
	return edges_dnds_categories


# def draw_contact_graph(nodes_dnds_categories, edges_dnds_categories, colors_dict, outfi):
# 	with open(outfi, 'w') as o:
# 		o.write('graph G {\n')
# 		o.write('\tfontname=\"Helvetica,Arial,sans-serif\"\n')
# 		o.write('\tnode [fontname=\"Helvetica,Arial,sans-serif\"]\n')
# 		o.write('\tedge [fontname=\"Helvetica,Arial,sans-serif\"]\n')
# 		o.write('\tlayout=fdp\n\n')
# 		o.write('\tbgcolor="transparent\"\n\n')

# 		for tp in nodes_dnds_categories:

# 			o.write('\tsubgraph ' + 'cluster' + tp + ' {\n')
# 			# o.write('\t\tbgcolor=\"lightgoldenrodyellow\"\n') # set bg color of each cluster (easier to differentiate each protein)

# 			o.write('\t\tperipheries=0\n')

# 			for res in nodes_dnds_categories[tp]:
# 				reslabel = tp + '_' + res # need a unique res label (since there could be the same residue pos on different tps)
# 				color = colors_dict[nodes_dnds_categories[tp][res]]
				
# 				o.write('\t\t' + reslabel + ' [color=\"' + color + '\", fillcolor=\"' + color + '\", style=\"filled\", label=\"\", shape=circle]\n')
				
# 			if tp in edges_dnds_categories: # draw edges (if present)
# 				for node1_node2 in edges_dnds_categories[tp]:
# 					node1 = node1_node2.split('-')[0]
# 					node2 = node1_node2.split('-')[1]
# 					reslabel1 = tp + '_' + node1
# 					reslabel2 = tp + '_' + node2
# 					color_edge = colors_dict[edges_dnds_categories[tp][node1_node2]]
					
# 					o.write('\t\t' + reslabel1 + ' -- ' + reslabel2 + ' [dir=none color=\"' + color_edge + '\", penwidth = 6]\n')
					


# 			o.write('\t}\n') # close bracket

# 		o.write('}')





def main():
	script_dir = osp.dirname(__file__)
	
	colors_dict = {'pos': '#D81B60', 'neg': '#1E88E5', 'other': '#222222', 'pos_neg': '#C79200'} 
	annos_data_dir = osp.join(script_dir, '..', 'data', 'contact_graphs', 'graphviz')
	dist_data_dir = osp.join(script_dir	, '..', 'data', 'dnds', 'dnds_for_residue_categories_species_tree', 'distances_btwn_residues_in_diff_dnds_categories')
	
	all_exo_dist = load_json(osp.join(dist_data_dir,'all_exo' + '_res_distances.json'))
	all_endo_dist = load_json(osp.join(dist_data_dir, 'all_endo' + '_res_distances.json'))
	mimicked_dist = load_json(osp.join(dist_data_dir, 'mimicked' + '_res_distances.json'))
	btwn_exo_spec_and_endo_spec_dist = load_json(osp.join(dist_data_dir, 'btwn_exo_specific_and_endo_specific_res_distances.json'))
	
	combined_interface_distances_dict = combine_interface_distances_dict(all_exo_dist, all_endo_dist, mimicked_dist, btwn_exo_spec_and_endo_spec_dist)

	output_fi_distances = osp.join(dist_data_dir, 'whole_interface_res_distances.json')
	dump_json(combined_interface_distances_dict, output_fi_distances)



	all_exo_nodes = load_json(osp.join(annos_data_dir, 'all_exo' + '_nodes_anno.json'))
	all_endo_nodes = load_json(osp.join(annos_data_dir, 'all_endo' + '_nodes_anno.json'))
	mimicked_nodes = load_json(osp.join(annos_data_dir, 'mimicked' + '_nodes_anno.json'))
	btwn_exo_endo_nodes = load_json(osp.join(annos_data_dir, 'btwn_exo_specific_and_endo_specific_nodes_anno.json'))

	# Combine all nodes in an interface together into a single file
	combined_interface_nodes = combine_interface_nodes(all_exo_nodes, all_endo_nodes, mimicked_nodes, btwn_exo_endo_nodes)
	combined_interface_nodes_out = osp.join(annos_data_dir, 'whole_interface' + '_nodes_anno.json')

	dump_json(combined_interface_nodes, combined_interface_nodes_out)

	# Annotate all edges in an interface (save into a single file)
	out_edges_anno = osp.join(annos_data_dir, 'whole_interface_edges_anno.json') # pos, neg, other categories for edges
	out_edges_anno_split_other = osp.join(annos_data_dir, 'whole_interface_edges_anno_split_other.json') # pos, neg, pos-other, neg-other, other-other categories
	
	combined_interface_edges = categorize_edges_in_contact_graph(combined_interface_distances_dict, combined_interface_nodes, out_edges_anno)
	combined_interface_edges_split_other = categorize_edges_in_contact_graph2(combined_interface_distances_dict, combined_interface_nodes, out_edges_anno_split_other)

	# Make file to draw contact graph
	print('\nMaking dot file draw whole interface contact graph')
	outfi_graph = osp.join(annos_data_dir, 'whole_interface_graph.dot')
	# draw_contact_graph(combined_interface_nodes, combined_interface_edges, colors_dict, outfi_graph)

	print('Done!')
if __name__ == '__main__':
	main()
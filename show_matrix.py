from time import time
import numpy as np
import operator
import itertools
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import scipy.sparse as sps
import networkx as nx
import pygraphviz
from sklearn import manifold


def plot_coo_matrix(matrix, filename):
	if not isinstance(matrix, sps.coo_matrix):
		matrix = sps.coo_matrix(matrix)
	fig = plt.figure()
	ax = fig.add_subplot(111, axisbg='black')
	ax.plot(matrix.col, matrix.row, 's', color = 'white', ms = 1)
	ax.set_xlim(0, matrix.shape[1])
	ax.set_ylim(0, matrix.shape[0])		# maybe invert these 2 lines
	ax.set_aspect('equal')
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.invert_yaxis()		
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.title('File name: %s' %filename)
	plt.show()
	return

def plot_matrix_fig(matrix, filename):
	fig = plt.figure()
	plt.spy(matrix)
	plt.title(filename)
	#plt.savefig('/home/igorpesic/Desktop/filename.png', dpi=600)
	plt.show()
	return

def save_matrix_fig(matrix, path, filename):
	fig = plt.figure()
	plt.spy(matrix)
	#plt.title(filename)
	plt.savefig(path + filename + '.png', dpi=600)
	plt.close(fig)
	return

def save_matrix_file(matrix, path, filename):
	np.savetxt(path + filename + '.txt', matrix)
	return	

# input matrix of size N*N
def viz_graph(matrix, filename):
	desktop = '/home/igorpesic/Desktop/'
	n,m = matrix.shape
	if n <>m:
		print 'Error in viz_graph!!!!'
	dt = [('len', float)]

	matrix = matrix.view(dt)
	G = nx.from_numpy_matrix(matrix)
	G = nx.to_agraph(G)
	G.node_attr.update(color="red", style="filled")
	G.edge_attr.update(color="blue", width="0.001")

	G.draw('/home/igorpesic/Desktop/' + filename + '.png', format='png', prog='dot') 	
	return
# TODO: has to be adjusted to show data according to the given similarity
# visualize the data perserving the euclidian distance between the data points
def viz_2d(matrix, filename):
	desktop = '/home/igorpesic/Desktop/'
	t0 = time()
	print 'Embeeding in 2D....'
	node_position_model = manifold.MDS()
	X = node_position_model.fit_transform(matrix)
	print 'Embeeding done. Time elapsed: ', time() - t0
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)

	plt.figure()
	ax = plt.subplot(111)
	for i in range(X.shape[0]):	
		plt.text(X[i, 0], X[i, 1], 'o',
			color='red',
			fontdict={'weight': 'bold', 'size': 9})
	

	plt.xticks([]), plt.yticks([])
	plt.title(filename)
	plt.savefig(desktop + filename + '_2D' + '.png', dpi=600)    
	return 

# ATTENTION: ONLY WORKS IF DATA IS PERFECTLY SORTED (IT SUPPORTS NON CLUSTERED POINTS)
def sort_matrix_NOT_WORKING(matrix, labels, column_sort = False):
	n = len(matrix)
	m = len(matrix[0])
	n_clusters = len(set(labels))

	# make new, sorted matrix (according to the cluster labels)
	points_in_cluster = [0 for x in range(n_clusters)]
	res_matrix = [[0 for x in range(m)] for x in range(1)]
	# list of sets where element i contains all column indices for cluster i 
	col_index_order = [set() for x in range(n_clusters)]
	for i in range(n_clusters, -2, -1):
		for row in range(n):
			if labels[row] == i and i >= 0:
				points_in_cluster[i] += 1
				res_matrix = np.vstack([res_matrix, matrix[row]])
				col_index_order[i] = col_index_order[i] | set([ind for ind, el in enumerate(matrix[row]) if el <> 0])
			if labels[row] == i and i == -1:
				res_matrix = np.vstack([res_matrix, matrix[row]])

	if column_sort:
		for i in range(len(col_index_order)):
			col_index_order[i] = list(col_index_order[i])			

		col_index_order = list(itertools.chain(*col_index_order))	
		col_indices = np.array(col_index_order)
		sorted_col_matrix = res_matrix[:,col_indices]
		return sorted_col_matrix
	else:
		return res_matrix

# puts non labeled points to the bottom and IF NECESSARY their columns to the right
# WORKS GOOD, CHECKED 2 times
def sort_matrix(data, labels, row_names, sim_mat = ''):
	row_names = np.array(row_names)
	n,m = data.shape
	n1 = labels.shape
	n2 = row_names.shape
	if n <> n1[0] or n <> n2[0]:
		print 'ERROR!!! matrix and labels dont have the same length', n, n1[0], n2[0]
		raise Exception
	if len(n1) > 1:	
		print 'ERROR!!! labels shape is wrong! ', len(n1)
		raise Exception

	# remove empty columns from the data matrix
	col_to_remove = []
	for col_test in range(m):
		tmp_col = data[:,col_test]
		if all(v == 0 for v in tmp_col):
			#print 'error in sorting algorithm, data columns are not filled, ', col_test
			# delete the empy column
			col_to_remove+= [col_test]
	data = np.delete(data, col_to_remove, 1)
	n,m = data.shape

	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)	

	row_sorted_matrix = [[0 for x in range(m)] for x in range(n)]
	sorted_labels = [0 for x in range(n)]
	sorted_names = [0 for x in range(n)]
	if sim_mat <> '':
		sorted_sim_mat = [[0 for x in range(n)] for x in range(n)]

	if -1 in labels:
		temp = set(labels)
		temp.remove(-1)
		temp = list(temp)
		temp.sort()
		order_of_clusters = temp + [-1]
	else:
		order_of_clusters = list(set(labels))

	# sort rows and labels ------------------------------------------------------	
	new_row = 0	
	for cluster_i in order_of_clusters:
		for point_j in range(n):
			if labels[point_j] == cluster_i:
				#row_sorted_matrix = np.vstack([row_sorted_matrix, data[j]])
				row_sorted_matrix[new_row] = data[point_j]
				sorted_labels[new_row] = labels[point_j]
				sorted_names[new_row] = row_names[point_j]
				if sim_mat <> '':
					sorted_sim_mat[new_row] = sim_mat[point_j]
				new_row += 1;
				
	if sim_mat <> '':			
		sim_mat = sorted_sim_mat

	row_sorted_matrix = np.array(row_sorted_matrix)
	sorted_labels = np.array(sorted_labels)
	sorted_names = np.array(sorted_names)
	# sort columns  --------------------------------------------------------------
	col_mapping = {} # col_mapping = {col_index:most_common}	
	# iterate columns
	for col_index in range(m):
		tmp_col = row_sorted_matrix[:,col_index]
		tmp_col = [label if val <> 0 else -100 for val, label in zip(tmp_col,sorted_labels)]
		# remove negative values, because index of column cannot be naegative
		# i.e. we dont consider non clustered elements for column sorting
		tmp_col_nn = [x for x in tmp_col if x >= 0]
		if len(tmp_col_nn) == 0:
			tmp_col = [x for x in tmp_col if x>= -1]	# only negative labels. so sort thier columns also
		else:
			tmp_col = tmp_col_nn

		if most_common(tmp_col) == -1:
			col_mapping[col_index] = 1e100000		#be sure that those with label -1 go at the end of matrix
		else:	
			col_mapping[col_index] = most_common(tmp_col)

	# sort col_maping according to values, 
	# and then take this order of keys as new column order for result matrix	
	sorted_dict = sorted(col_mapping.items(), key= operator.itemgetter(1))
	sorted_col_indices = [col_index_sorted for col_index_sorted, most_comm in sorted_dict]
	# list of labels for each columns e.g. [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,......]
	column_labels = [most_comm if most_comm < 100000 else -1 for col_index_sorted, most_comm in sorted_dict]

	# rearrange colums 
	sorted_col_indices = np.array(sorted_col_indices)		# first are the columns of mostly non clustered data
	result = row_sorted_matrix[:,sorted_col_indices]
	return result, sorted_labels, sorted_names, column_labels

# finds most common element in list
def most_common(lst):
	if len(lst) == 0:
		print 'error in most_common function'
	return max(set(lst), key=lst.count)		
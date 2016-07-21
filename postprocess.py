import numpy as np

# TODO: fix it so it removes the point that is neighboor with most
#		different labels
# NOT USED and propbably NOT WORKING :)
def remove_colision_points(dist_sim_mat, mat_type, labels):
	if mat_type <> 'dist' and mat_type <> 'sim':
		print 'error! mat_type not supported: ', mat_type
		return
	n1,m1 = dist_sim_mat.shape
	n2 = len(labels)
	if n1 <> m1 or n2 <> n1:
		print 'error in remove_col_points'
	else:
		n = n1

	# make a list of adjeacent lists:
	list_of_adj_lists = []
	# fill it
	for i1 in range(n):
		list_i = []
		for j1 in range(i1+1, n):
			# TODO: adjust to accept sim matrix also
			if mat_type == 'dist':
				#if dist_sim_mat[i1][j1] < 1:
				if dist_sim_mat[i1][j1] < 1 and labels[i1] <> labels[j1] and labels[i1] <> -1 and labels[j1] <> -1:
					list_i += [j1]
			else: 
				#if dist_sim_mat[i1][j1] > 0:
				if dist_sim_mat[i1][j1] > 0 and labels[i1] <> labels[j1] and labels[i1] <> -1 and labels[j1] <> -1:
					list_i += [j1]
		list_of_adj_lists += [list_i]

	print 'list_of_adj_lists:'
	print list_of_adj_lists

	labels = labels
	# iterate list_of_adj_lists
	# remove points (e.g. give them label -1)	
	for i in range(n):
		my_label = labels[i]
		list_i = list_of_adj_lists[i]
		#diff_labeled_neighboors_of_i = sum(1 for point in list_i if labels[point]<>my_label and labels[point]<>-1)
		clusters_connected_to_i = [labels[point] for point in list_i if labels[point]<>my_label and labels[point]<>-1]
		clusters_connected_to_i = set(clusters_connected_to_i)
		clusters_connected_to_i = len(clusters_connected_to_i)
		for point_j in list_i:
			# skip points with label -1 and points from the same cluster
			if labels[point_j] <> -1 and labels[point_j] <> my_label:
				# check 
				list_j = list_of_adj_lists[point_j]
				label_j = labels[point_j]
				#diff_labeled_neighboors_of_j = sum(1 for point in list_j if labels[point]<>label_j and labels[point]<>-1)
				clusters_connected_to_j = [labels[point] for point in list_j if labels[point]<>label_j and labels[point]<>-1]
				clusters_connected_to_j = set(clusters_connected_to_j)
				clusters_connected_to_j = len(clusters_connected_to_j)

				#if diff_labeled_neighboors_of_j > diff_labeled_neighboors_of_i:
				if clusters_connected_to_j > clusters_connected_to_i:
					# remove j
					labels[point_j] = -1
				else:
					# remove i
					labels[i] = -1
					my_label = -1
					break
					
	return labels				
	
# check if there is a colision in the SORTED matrix and assign it to -1	
# 	data: 			const. matrix 
#	labels: 		labels for each row obtained by clustering algs. 
#					(-1 if it is already an outlier, this is supported in DBSCAN for example, 
#					and 0,1,2.. are the block labels)
# 	row_names:		constraint names. We need them here so if we reorder the rows in the const mat we must also
#					reorder the names correspondingly 
# 	column_labels:	assignement for each column (this is basically the label that appers most often in each column)
def remove_colision_points2(data, labels, row_names, column_labels):
	data = np.array(data)
	labels = np.array(labels)
	column_labels = np.array(column_labels)
	n,m = data.shape

	# this matrix of same dim as data, but instead of real entries, 
	# we change zero values to -100 (in order not to confuse it with label 0)
	# and non-zero values to label value for specific column (or -1)
	label_matrix = np.zeros((n,m))

	for i in range(m):
		tmp_col = data[:,i]
		col_label = column_labels[i]
		tmp_col = [col_label if val <> 0 else -100 for val in tmp_col]
		label_matrix[:,i] = tmp_col 

	# remove all points that have entries which are not labels[i] or -100	
	for i in range(n):
		row_i = label_matrix[i,:]
		if all(v == -100 or v == labels[i] for v in row_i): 
			pass
		else:	
			'''print labels[i]
			print row_i
			print 'column: ', label_matrix[:,228]
			raise Exception('remove_colision_points2(): Something is not clear here.')'''
			labels[i] = -1

	# sort the data and labels after removal -----------------------------------
	row_sorted_matrix = [[0 for x in range(m)] for x in range(n)]
	sorted_labels = [0 for x in range(n)]
	sorted_names = [0 for x in range(n)]

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
				new_row += 1;

	# after removing some points to -1 from cluster, it can happen that some clusters stay empty, 
	# so that some number might be skipped in new labels order
	# at the end we want set(labels) to be 0,1,2,3...n_new_clusters, -1
	change = False
	last_label = 0
	fixed_labels = [0 for x in range(n)]
	for i in range(n):
		if sorted_labels[i] == -1:
			fixed_labels[i] = -1
			continue
		if i > 0:
			if sorted_labels[i] <> sorted_labels[i-1]:
				change = True
			else:
				change = False
		else:
			change = False

		if change == False:
			fixed_labels[i] = last_label
		else:
			last_label += 1
			fixed_labels[i] = last_label 

	row_sorted_matrix = np.array(row_sorted_matrix)
	sorted_labels = np.array(sorted_labels)
	sorted_names = np.array(sorted_names)
	fixed_labels = np.array(fixed_labels)

	return row_sorted_matrix, fixed_labels, sorted_names	
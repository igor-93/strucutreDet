from __future__ import division
import numpy as np
import gc
import scipy.io
from scipy.sparse import *
from scipy import *
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph._validation import validate_graph

from mst_clustering import MSTClustering

import parse
import preprocess as pp
import postprocess as postp
from show_matrix import sort_matrix #save_matrix_fig, save_matrix_file,
import write_dec as dec


# for reference: https://github.com/jakevdp/mst_clustering
def mst(instance_path, res_folder, strategy = 2):
	instances = instance_path.rsplit('/', 1)[0] + '/'
	file = instance_path.rsplit('/', 1)[1]
	input_type  = '.' + file.rsplit('.', 1)[1]
	file = file.rsplit('.', 1)[0]
	data, row_names = parse.read(instances + file + input_type)
	print 'Size of data matrix: ', data.shape
	if len(data) <> len(row_names):
		print 'MST error: data and row_names have diff. lens', len(data), len(row_names)	
	#save_matrix_fig(data, res_folder, file+'_in')
	dist_matrix = []
	'''OLD try:
		dist_matrix = np.load(res_folder+file+'_dist'+str(strategy)+'.npy')
		print 'Distance matrix %s found.' %(res_folder+file+'_dist'+str(strategy)+'.npy')
	except:
		print 'Distance matrix %s NOT found!!!!' %(res_folder+file+'_dist'+str(strategy)+'.npy')
		dist_matrix = pp.strategy(data, 'distance',strategy)	
		np.save(res_folder+file+'_dist'+str(strategy), dist_matrix)'''
	try:
		dist_matrix = scipy.io.mmread(res_folder+file+'_dist'+str(strategy)).tocsr()
		print 'Distance matrix %s found.' %(res_folder+file+'_dist'+str(strategy))
	except:
		print 'Distance matrix %s NOT found!!!!' %(res_folder+file+'_dist'+str(strategy))
		dist_matrix = pp.strategy(data, 'distance',strategy)	
		scipy.io.mmwrite(res_folder+file+'_dist'+str(strategy),dist_matrix)	

	occupancy = len(dist_matrix.data) / (dist_matrix.shape[0] * dist_matrix.shape[1]) * 100
	q = 10
	dist_percentile = np.percentile(a=dist_matrix.data, q=q, axis=None)
	if dist_percentile == 0: # or strategy == 6:
		q = 1
		print 'Recalculating dist_percentile..'
		#dist_percentile = np.percentile(a=dist_matrix, q=q)
		dist_percentile = np.percentile(a=dist_matrix.data, q=q, axis=None)

	print 'dist_percentile = ', dist_percentile
	old_n_clusters = 0
	old_non_clustered = 0

	# list to save labels from all iterations, so we can later pick the best clustering
	res_from_diff_params = {}
	nr_clusters_from_diff_params = {}
	non_clustered_from_diff_params = {}
	distribution_from_diff_params = {}
	best_iteration = -1
	sec_best_iteration = -1
	n = dist_matrix.shape[0]
	min_non_clusterd = n
	s_min_non_clusterd = n
	min_std_dev = n
	sec_threshold = 0.0001
	n_iterations = 49  		# must be an odd number 
	eps_list = get_eps_list_geom(mid=dist_percentile, length=n_iterations, strategy=strategy)

	pure_diagonal = False
	n_components, p_d_labels = connected_components(dist_matrix, directed=False)
	if n_components > 1:
		print 'MST: pure diagonal found...'
		pure_diagonal = True


	# cluster the data with MST ---------------------------------------------
	for iteration in range(n_iterations):
		gc.collect()
		if dist_percentile == 0:
			print 'dist_percentile = %i, -> we cannot use MST for clustering this instance.' %dist_percentile
			break
		# eps is in range: [dist_percentile - 0.5, dist_percentile + 0.5] but with geometric progression
		eps = eps_list[iteration]

		if eps <= 0:
			continue
		if eps >= 1:
			break	
		# for distance strategy 1: 0.054...
		#eps = 0.1 + (iteration / 10) 
		min_samples = 4
		#print 'DEBUG: eps = ', eps
		labels = []
		print '_______________________________________________________'
		print 'Running MST...'
		print 'iteration= ', iteration
		print 'eps = ', eps
		print 'min_samples = ', min_samples
		
		if pure_diagonal:
			labels = p_d_labels
		else:
			# cutoff_scale is min size of edges to cut
			model = MSTClustering(cutoff_scale=eps, min_cluster_size = min_samples, metric = "precomputed")
			labels = model.fit_predict(dist_matrix)
			
		#print 'labels = ', labels
		#raise Exception('Wait....')

		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		
		num_per_cluster = {}
		for i in range(n_clusters):
			num_per_cluster[i] = 0

		for label in labels:
			for i in range(n_clusters):
				if label == i:
					num_per_cluster[i] += 1; 
		non_clustered = 0;			
		for label in labels:
			if label == -1:
				non_clustered += 1	
		
		# criteria for skiping or breaking the loop ---------------------------------------------
		# skip the iteration if the number of clusters is as before
		if iteration == 0:
			old_n_clusters = n_clusters
			old_non_clustered = non_clustered
		if n_clusters == old_n_clusters and non_clustered == old_non_clustered and iteration > 0:
			print 'Same clustering...'
			continue
		old_n_clusters = n_clusters
		old_non_clustered = non_clustered
		if n_clusters == 1 and non_clustered == 0:
			print 'Stopping because bigger EPS will be the same.'
			break
		# ---------------------------------------------------------------------------------------
		# display some information
		print 'Estimated number of clusters: ',  n_clusters		
		print 'Number of points per cluster: ', num_per_cluster
		print 'Number of non clustered points:', non_clustered
		#draw(A=sim_matrix, colors=labels)
		# ---------------------------------------------------------------------------------------
		sorted_data, sotred_labels, sorted_names, column_labels = sort_matrix(data, labels, row_names)
		#print 'DEBUG:'
		#print 'column_labels = ', column_labels
		#print 'sotred_labels = ', sotred_labels
		#save_matrix_fig(sorted_data, res_folder, file + '_B_dec' +  str(iteration))

		# pull down the points which have non-zero value that colides with points from other clusters
		sorted_data2, sotred_labels2, sorted_names2 = postp.remove_colision_points2(sorted_data, 
																				sotred_labels, sorted_names, column_labels)

		num_per_cluster = {}
		n_clusters = len(set(sotred_labels2)) - (1 if -1 in sotred_labels2 else 0)
		if -1 in sotred_labels2:
			all_clusters_list = range(-1, n_clusters)
		else:
			all_clusters_list = range(n_clusters)

		for i in all_clusters_list:
			num_per_cluster[i] = 0

		for label in sotred_labels2:
			for i in all_clusters_list:
				if label == i:
					num_per_cluster[i] += 1; 
		non_clustered = 0;			
		for label in sotred_labels2:
			if label == -1:
				non_clustered += 1	
		print 'Estimated number of clusters after removal: ',  n_clusters
		print 'Number of points per cluster after removal: ', num_per_cluster
		print 'Number of non clustered points after removal:', non_clustered
		if 0 in num_per_cluster.values():
			print 'TIME TO DEBUG:'
			print 'sotred_labels2 = ', sotred_labels2

		# save picture of end matrix
		#save_matrix_fig(sorted_data2, res_folder, file + '_A_dec' +  str(iteration))
		#if res2_folder <> 'none':
			#save_matrix_fig(sorted_data2, res2_folder, file + '_A_dec' +  str(iteration))
		# find the best iteration, so we only save the best one --------------------------
		label_name_pairs = zip(sotred_labels2, sorted_names2)
		if non_clustered < min_non_clusterd:
			res_from_diff_params[iteration] = label_name_pairs
			nr_clusters_from_diff_params[iteration] = n_clusters
			non_clustered_from_diff_params[iteration] = non_clustered
			distribution_from_diff_params[iteration] = num_per_cluster
			min_non_clusterd = non_clustered
			best_iteration = iteration
			print 'this is best iteration currently'
		# find the best iteration (according variance of cluster sizes), ----------------
		# so we only save the best one 
		temp_num_per_cluster = num_per_cluster.copy()
		if -1 in temp_num_per_cluster.keys():
			del temp_num_per_cluster[-1]
		if len(temp_num_per_cluster.values()) > 1:
			std_dev = np.std(temp_num_per_cluster.values())	
			mean = np.mean(temp_num_per_cluster.values())	
			rel_std_dev = std_dev / mean
			rel_std_dev *= pow(non_clustered/n, 2)
			print 'DEBUG: adjusted rel_std_dev = ', rel_std_dev
			std_dev = rel_std_dev
			# we accept the iteration if adjusted rel_std_dev is smaller, or 
			# if it is within the threshold and number of nonclustered points is smaller
			if (std_dev - min_std_dev) <= sec_threshold and non_clustered < s_min_non_clusterd:
				sec_criteria_fulfiled = True
			else:
				sec_criteria_fulfiled = False
			if std_dev < min_std_dev or sec_criteria_fulfiled:
				res_from_diff_params[iteration] = label_name_pairs
				nr_clusters_from_diff_params[iteration] = n_clusters
				non_clustered_from_diff_params[iteration] = non_clustered
				distribution_from_diff_params[iteration] = num_per_cluster
				min_std_dev = std_dev
				s_min_non_clusterd = non_clustered
				sec_best_iteration = iteration
				print 'this is second best iteration currently'	

		if pure_diagonal:
			break
		# ----------------------------------------------------------------------------------
		print '_______________________________________________________'
				

	best_found = False	
	best_n_clusters = 0
	best_non_clusterd = data.shape[0]
	best_distro = {-1:data.shape[0]}
	best_dec = ''	# name of dec file for best iteration

	s_best_found = False
	s_best_n_clusters = 0
	s_best_non_clusterd = data.shape[0]
	s_best_distro = {-1:data.shape[0]}
	s_dec = ''		# name of dec file for second best iteration

	# save .dec from best iteration
	print 'best_iteration= ', best_iteration
	print 'sec best iteration = ', sec_best_iteration
	if best_iteration >= 0:
		best_found = True
		best_n_clusters = nr_clusters_from_diff_params[best_iteration]
		best_non_clusterd = non_clustered_from_diff_params[best_iteration]
		best_distro = distribution_from_diff_params[best_iteration]
		best_dec = file + '_mst_' + str(best_n_clusters) + '_' + str(best_non_clusterd) +'_dist'+str(strategy)
		dec.write(path = res_folder, filename = best_dec, label_name_pairs = res_from_diff_params[best_iteration])
		print '.dec file %s for iteration %i saved.' %(res_folder+best_dec, best_iteration)
	if sec_best_iteration >= 0:
		if sec_best_iteration <> best_iteration:
			s_best_found = True
		s_best_n_clusters = nr_clusters_from_diff_params[sec_best_iteration]
		s_best_non_clusterd = non_clustered_from_diff_params[sec_best_iteration]
		s_best_distro = distribution_from_diff_params[sec_best_iteration]
		s_dec = file + '_mstSTD_' + str(s_best_n_clusters) + '_' + str(s_best_non_clusterd) +'_dist'+str(strategy)
		dec.write(path = res_folder, filename = s_dec, label_name_pairs = res_from_diff_params[sec_best_iteration])
		print '.dec file %s for iteration %i saved.' %(res_folder+s_dec, sec_best_iteration)	
	print '_______________________________________________________'
	print '_______________________________________________________'	
	gc.collect()
	return best_found, best_n_clusters, best_non_clusterd, best_distro, best_dec, data.shape[0], \
			s_best_found, s_best_n_clusters, s_best_non_clusterd, s_best_distro, s_dec


# returns a geometrix array from  mid-0.5 to mid+0.5 of length = length
def get_eps_list_geom(mid, length, strategy):
	# sould split in N1 and N2 so that 
	if strategy == 2:
		N1 = int((length+1) / 2)
		N2 = N1
	else:
		N2 = int((length+1) / 4)
		N1 = abs(N2 - length) + 1 

	s = mid
	end1 = mid + 0.9      # lower boundary
	end2 = mid + 0.4      # upper boundary

	q1 = pow(end1/s, 1/(N1-1))
	q2 = pow(end2/s, 1/(N2-1))

	geom_seq1 = [2*s-s*pow(q1,i) for i in range(N1-1,0,-1)]
	geom_seq2 = [s*pow(q2,i) for i in range(N2)]

	eps_list = geom_seq1 + geom_seq2
	return eps_list

# returns an aritmetic array from  min(0, mid-0.5) to mid and mid to max(1,mid+0.5) of length = length
def get_eps_list_ar(mid, length):
	# sould split in N1 and N2 so that 

	N1 = int((length+1) / 4)
	N2 = abs(N1 - length) 

	begin = max(0, mid-0.5)
	end = min(1,mid+0.5)

	step_1 = (mid - begin) / N1
	step_2 = abs(mid - end) / N2

	seq1 = [begin+i*step_1 for i in range(N1)]
	seq2 = [mid+i*step_2 for i in range(N2)]

	eps_list = seq1 + seq2
	return eps_list
from __future__ import division
import numpy as np
import gc
from sklearn.cluster import affinity_propagation
from sklearn import metrics

import parse
import preprocess as pp
import postprocess as postp
from show_matrix import save_matrix_fig, save_matrix_file, sort_matrix
import write_dec as dec


def affProp(instance_path, res_folder, strategy = 2):
	instances = instance_path.rsplit('/', 1)[0] + '/'
	file = instance_path.rsplit('/', 1)[1]
	input_type  = '.' + file.rsplit('.', 1)[1]
	file = file.rsplit('.', 1)[0]
	data, row_names = parse.read(instances + file + input_type)
	print 'Size of data matrix: ', data.shape
	if len(data) <> len(row_names):
		print 'Af prop error: data and row_names have diff. lens', len(data), len(row_names)	
	#save_matrix_fig(data, res_folder, file+'_in')
	sim_matrix = []
	# 	
	try:
		sim_matrix = np.load(res_folder+file+'_sim'+str(strategy)+'.npy')
		print 'Sim matrix %s found.' %(res_folder+file+'_sim'+str(strategy)+'.npy')
	except:
		print 'Sim matrix %s NOT found!!!!' %(res_folder+file+'_sim'+str(strategy)+'.npy')
		sim_matrix = pp.strategy(data, 'sim',strategy)	
		np.save(res_folder+file+'_sim'+str(strategy), sim_matrix)

	old_n_clusters = 0
	old_non_clustered = 0

	# list to save labels from all iterations, so we can later pick the best clustering
	res_from_diff_params = {}
	nr_clusters_from_diff_params = {}
	non_clustered_from_diff_params = {}
	distribution_from_diff_params = {}
	best_iteration = -1
	sec_best_iteration = -1
	n = sim_matrix.shape[0]
	min_non_clusterd = n
	s_min_non_clusterd = n
	max_std_dev = n
	sec_threshold = 0.0001
	n_iterations = 20  		# must be an odd number 

	sim_matrix[sim_matrix == 0] = -1e10
	#min_preferance = 0
	#min_preferance *= np.max(sim_matrix[sim_matrix > 0])
	min_preferance = np.min(sim_matrix[sim_matrix > 0]) -10
	max_preferance = np.median(sim_matrix[sim_matrix > 0])
	print 'min_preferance, ', min_preferance
	print 'max_preferance, ', max_preferance
	
	if min_preferance > max_preferance:
		raise Exception('Something is wrong with preferance setting: %d %d', 
			min_preferance, max_preferance)
	elif min_preferance == max_preferance:
		n_iterations = 1
		pref_list = [min_preferance]
	
	pref_step = (max_preferance-min_preferance) / n_iterations

	# cluster the data with DBSCAN ---------------------------------------------
	for iteration in range(n_iterations):
		
		if iteration == 0:
			preference = min_preferance
		else:
			preference += pref_step
		labels = []
		print '_______________________________________________________'
		print 'Aff. Prop. with preferance =', preference
		
		_, labels = affinity_propagation(sim_matrix, preference=preference)
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		
		num_per_cluster = {}
		for i in range(n_clusters):
			num_per_cluster[i] = 0

		for label in labels:
			for i in range(n_clusters):
				if label == i:
					num_per_cluster[i] += 1; 


		# TODO: criteria for skiping or breaking the loop ---------------------------------------------
		# skip the iteration if the number of clusters is as before
		if iteration == 0:
			old_n_clusters = n_clusters
		#elif n_clusters >= old_n_clusters:
		#	break
		old_n_clusters = n_clusters
		# increase the preferance 
		if n_clusters == 1:
			print 'DEBUG: Aff prop. n_clusters == 1, going to next iteration'
			min_preferance = preference
			max_preferance += (max_preferance - min_preferance) / 2
			pref_step = (max_preferance-min_preferance) / (n_iterations-iteration)
			print 'min = %f, max = %f, step = %f' %(min_preferance, max_preferance, pref_step)
			continue
		# lower the preferance	
		if n_clusters >= 0.1*n:
			print 'DEBUG: Aff prop. n_clusters = %i, TOO HIGH!!!' %n_clusters
			max_preferance = preference
			min_preferance = preference - pref_step
			pref_step = (max_preferance-min_preferance) / (n_iterations-iteration)
			print 'min = %f, max = %f, step = %f' %(min_preferance, max_preferance, pref_step)
			continue	
		# ---------------------------------------------------------------------------------------
		# display some information
		print 'Estimated number of clusters: ',  n_clusters		
		print 'Number of points per cluster: ', num_per_cluster
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
			if n_clusters > 1:
				second_best = iteration
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
			if (std_dev - max_std_dev) <= sec_threshold and non_clustered < s_min_non_clusterd:
				sec_criteria_fulfiled = True
			else:
				sec_criteria_fulfiled = False
			if std_dev < max_std_dev or sec_criteria_fulfiled:
				res_from_diff_params[iteration] = label_name_pairs
				nr_clusters_from_diff_params[iteration] = n_clusters
				non_clustered_from_diff_params[iteration] = non_clustered
				distribution_from_diff_params[iteration] = num_per_cluster
				max_std_dev = std_dev
				s_min_non_clusterd = non_clustered
				sec_best_iteration = iteration
				print 'this is second best iteration currently'		
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
		best_dec = file + '_affProp_' + str(best_n_clusters) + '_' + str(best_non_clusterd)
		dec.write(path = res_folder, filename = best_dec, label_name_pairs = res_from_diff_params[best_iteration])
		print '.dec file %s for iteration %i saved.' %(res_folder+best_dec, best_iteration)
	if sec_best_iteration >= 0:
		if sec_best_iteration <> best_iteration:
			s_best_found = True
		s_best_n_clusters = nr_clusters_from_diff_params[sec_best_iteration]
		s_best_non_clusterd = non_clustered_from_diff_params[sec_best_iteration]
		s_best_distro = distribution_from_diff_params[sec_best_iteration]
		s_dec = file + '_affPropSTD_' + str(s_best_n_clusters) + '_' + str(s_best_non_clusterd)
		dec.write(path = res_folder, filename = s_dec, label_name_pairs = res_from_diff_params[sec_best_iteration])
		print '.dec file %s for iteration %i saved.' %(res_folder+s_dec, sec_best_iteration)	
	print '_______________________________________________________'
	print '_______________________________________________________'	
	gc.collect()
	return best_found, best_n_clusters, best_non_clusterd, best_distro, best_dec, data.shape[0], \
			s_best_found, s_best_n_clusters, s_best_non_clusterd, s_best_distro, s_dec
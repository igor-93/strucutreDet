from __future__ import division
import numpy as np
from mcl_impl import mcl_implementation, draw
import scipy.io
from scipy.sparse import *
from scipy import *

import gc
import time

import parse
import preprocess as pp
import postprocess as postp
from show_matrix import sort_matrix #save_matrix_fig, save_matrix_file,
import write_dec as dec

def mcl(instance_path, res_folder, strategy=2):
	instances = instance_path.rsplit('/', 1)[0] + '/'
	file = instance_path.rsplit('/', 1)[1]
	input_type  = '.' + file.rsplit('.', 1)[1]
	file = file.rsplit('.', 1)[0]
	data, row_names = parse.read(instances + file + input_type)
	print 'Size of data matrix: ', data.shape
	if len(data) <> len(row_names):
		print 'MCL error: data and row_names have diff. lens', len(data), len(row_names)
		raise Exception
	#save_matrix_fig(data, res_folder, file+'_in')
	sim_matrix = []
	'''try:
		sim_matrix = np.load(res_folder+file+'_sim'+str(strategy)+'.npy')
		n = sim_matrix.shape[0]
		print 'Sim matrix %s found.' %(res_folder+file+'_sim'+str(strategy)+'.npy')
	except:
		print 'Sim matrix %s NOT found!!!!' %(res_folder+file+'_sim'+str(strategy)+'.npy')
		sim_matrix = pp.strategy(data, 'sim',strategy)	
		np.save(res_folder+file+'_sim'+str(strategy)+'.npy', sim_matrix)'''

	try:
		sim_matrix = scipy.io.mmread(res_folder+file+'_sim'+str(strategy)).tocsr()
		print 'Sim matrix %s found.' %(res_folder+file+'_sim'+str(strategy))
	except:
		print 'Sim matrix %s NOT found!!!!' %(res_folder+file+'_sim'+str(strategy))
		sim_matrix = pp.strategy(data, 'sim',strategy)	
		scipy.io.mmwrite(res_folder+file+'_sim'+str(strategy), sim_matrix)	
 	
	############################################
	# good visualization
	#draw(sim_matrix, colors=[0 for x in range(sim_matrix.shape[0])])
	############################################
	old_n_clusters = 0
	old_non_clustered = 0

	# list to save labels from all iterations, so we can later pick the best clustering
	res_from_diff_params = {}
	nr_clusters_from_diff_params = {}
	non_clustered_from_diff_params = {}
	distribution_from_diff_params = {}
	best_iteration = -1
	sec_best_iteration = -1
	print sim_matrix.shape
	n = sim_matrix.shape[0]
	min_non_clusterd = n
	s_min_non_clusterd = n
	min_std_dev = n
	sec_threshold = 0.0001
	iteration = 0
	perfect_cl_found = False
	
	min_inf_factor = 1.1
	max_inf_factor = 2.2
	inf_l = max_inf_factor - min_inf_factor

	nr_cl_with_diff_params = []
	total_time = 0
	# cluster the data with MCL ---------------------------------------------
	for exp_iter in range(2,3):
	#for exp_iter in range(2,5):
		nr_cl_with_diff_inf = []
		#for save_id, inf_iter in enumerate(np.arange(1.2, 1.85, 0.05)):
		for save_id, inf_iter in enumerate([1.3]):
		#for inf_iter in range(6):	
			gc.collect()
			labels = [-1 for x in range(n)]

			print '#############################################################'
			expand_factor = exp_iter
			#inflate_factor = 1.2 + inf_iter*0.4
			inflate_factor = inf_iter
			print 'DEBUG: iteration = ', iteration
			try:
				labels = np.load(res_folder+file+'_mcl_fix_'+str(save_id)+'_sim'+str(strategy)+'.npy')
				print 'Existing clustering found for expand = %i, inflate = %f' %(expand_factor,inflate_factor) 
				print res_folder+file+'_mcl_fix_'+str(save_id)+'_sim'+str(strategy)+'.npy'

			except:
				print 'NO Existing clustering found for expand = %i, inflate = %f' %(expand_factor,inflate_factor)
				print 'Expand Factor = ', expand_factor
				print 'Infalte Factor = ', inflate_factor
				#print 'NO existing clustering found for expand = %i, inflate = %i.' %(expand_factor,inflate_factor)
				# inflation weakens relations between clusters and strenhtens relations in clusters
				# expansion operator is responsible for allowing flow to connect different regions of the graph.
				# bigger the expand_factor, less clusters (too big, everything is in 1 cluster)
				start_time = time.time()
				clusters = mcl_implementation(sim_matrix, expand_factor = expand_factor, max_loop = 20,
									inflate_factor = inflate_factor)
				curr_time = (time.time() - start_time)
				print 'curr_time: ', curr_time
				total_time += curr_time
				clust_map = {}
				for k, vals in clusters.items():
					for v in vals:
						clust_map[v] = k

				colors = []
				for i in range(n):
					colors.append(clust_map.get(i, 100))

				set_of_colors = set(colors)	
				print 'DEBUG: n clusters = ', len(set_of_colors)

				

				cur_label = 0
				labels = colors
				for cluster in set_of_colors:
					labels = [cur_label if label == cluster else label for label in labels] 
					cur_label += 1
				labels = np.array(labels)

				np.save(res_folder+file+'_mcl_fix_'+str(save_id)+'_sim'+str(strategy)+'.npy', labels)

			# continue if all points in 1 cluster
			if len(set(labels)) == 1:
				print 'skiping this iteration...'
				nr_cl_with_diff_inf += [1]
				continue

			# change expand factor if all points in different clusters
			if len(set(labels)) >= 0.3*n:
				print 'moving to next expand factor...'
				break
			############################################
			# good visualization
			'''if iteration == 3 and file <> 'pp08aCUTS':
				draw(sim_matrix, colors=[label+1 for label in sotred_labels2])
			if iteration == 1 and file == 'pp08aCUTS':
				draw(sim_matrix, colors=[label+1 for label in sotred_labels2])'''
			############################################
			
			n_clusters = len(set(labels)) - (1 if -1 in labels else 0)


			nr_cl_with_diff_inf += [n_clusters]


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
				continue
			old_n_clusters = n_clusters
			old_non_clustered = non_clustered
			
			# ---------------------------------------------------------------------------------------
			# display some information
			print 'Estimated number of clusters: ',  n_clusters		
			print 'Number of points per cluster: ', num_per_cluster
			#print 'REMOVE ME! '
			#continue
			# ---------------------------------------------------------------------------------------
			sorted_data, sotred_labels, sorted_names, column_labels = sort_matrix(data, labels, row_names)
			
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
			print 'Number of points per cluster after removal: ', num_per_cluster
			print 'Number of non clustered points after removal:', non_clustered
			
			# change expand factor if all points in different clusters
			if n_clusters >= 0.1*n:
				print 'moving to next expand factor...'
				break
			

			# save picture of end matrix
			#save_matrix_fig(sorted_data2, res_folder, file + '_A_dec' +  str(iteration))
		
			# find the best iteration (according to # of non clustered points), --------------
			# so we only save the best one 
			label_name_pairs = zip(sotred_labels2, sorted_names2)
			if non_clustered < min_non_clusterd:
				res_from_diff_params[iteration] = label_name_pairs
				nr_clusters_from_diff_params[iteration] = n_clusters
				non_clustered_from_diff_params[iteration] = non_clustered
				distribution_from_diff_params[iteration] = num_per_cluster
				min_non_clusterd = non_clustered
				best_iteration = iteration
				print 'this is best iteration currently'	
			if non_clustered == 0:
				print 'Perfect clustering found!'
				perfect_cl_found = True
				sec_best_iteration = iteration
				break;
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
			# ----------------------------------------------------------------------------------
			print '#############################################################'
			iteration += 1
		nr_cl_with_diff_params += [nr_cl_with_diff_inf]
		if perfect_cl_found:
			break;		
					

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
		best_dec = file + '_mcl_' + str(best_n_clusters) + '_' + str(best_non_clusterd) +'_sim'+str(strategy)
		dec.write(path = res_folder, filename = best_dec, label_name_pairs = res_from_diff_params[best_iteration])
		print '.dec file %s for iteration %i saved.' %(res_folder+best_dec, best_iteration)
	if sec_best_iteration >= 0:
		if sec_best_iteration <> best_iteration:
			s_best_found = True
		s_best_n_clusters = nr_clusters_from_diff_params[sec_best_iteration]
		s_best_non_clusterd = non_clustered_from_diff_params[sec_best_iteration]
		s_best_distro = distribution_from_diff_params[sec_best_iteration]
		s_dec = file + '_mclSTD_' + str(s_best_n_clusters) + '_' + str(s_best_non_clusterd) +'_sim'+str(strategy)
		if sec_best_iteration <> best_iteration:
			dec.write(path = res_folder, filename = s_dec, label_name_pairs = res_from_diff_params[sec_best_iteration])
		print '.dec file %s for iteration %i saved.' %(res_folder+s_dec, sec_best_iteration)	

	print 'Total_time: ', total_time	
	print '_______________________________________________________'
	print '_______________________________________________________'	    
	gc.collect()

	return best_found, best_n_clusters, best_non_clusterd, best_distro, best_dec, data.shape[0], \
			s_best_found, s_best_n_clusters, s_best_non_clusterd, s_best_distro, s_dec
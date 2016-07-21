from __future__ import division
import collections
import ast
import numpy as np 
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from math import log
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import os

'''
		TODO:
			insert the info for 'first lp time' and 'final dual bound'.
			these 2 things can be overlayed or stacked on the same bar as time
'''

# table form: instance name, type, 'gcg', gcg time, alg1 name, alg1 time, agl2 name, agl2 time....
def plot_time(table, n_cl_algs, all_distros, run_number, strategy, result_folder, time_limit):
	print 'Plotting table for strategy ', strategy
	tools.set_credentials_file(username='igor-93', api_key='pblx0ahsl3')
	inst_names = table[:,0]
	inst_names = np.delete(inst_names, 0)
	inst_names = list(inst_names)
	n_points = []
	strategy = str(strategy)
	n_instances = len(table) - 1
	gcg_col_index = 2+n_cl_algs*5	# index of GCG column in the table
	#types = [row[1] for row in table]
	#gcg_times = [row[2] for row in table]

	data = []
	gcg_auto_color = 'rgba(100,100,100,1)'
	gcg_decs_color = 'rgba(100,100,100,0.2)'
	if n_cl_algs == 4:
		cluster_alg_colors = ['rgba(222,45,38,0.8)', 'rgba(74,224,90,1)', 
							  'rgba(74,124,90,1)', 'rgba(38,45,238,0.8)']	
	elif n_cl_algs == 6:
		cluster_alg_colors = ['rgba(224,45,38,0.8)', 'rgba(180,45,38,0.8)', 
							  'rgba(74,224,90,1)', 'rgba(74,180,90,1)', 
							  'rgba(38,45,224,0.8)', 'rgba(38,45,180,0.8)']
	elif n_cl_algs == 8:
		cluster_alg_colors = ['rgba(224,45,38,0.8)', 'rgba(180,45,38,0.8)', 
							  'rgba(74,224,90,1)', 'rgba(74,180,90,1)', 
							  'rgba(54,220,220,1)', 'rgba(54,190,190,1)', 
							  'rgba(38,45,224,1)', 'rgba(38,45,160,1)']						  
	else:
		cluster_alg_colors = n_cl_algs*['rgba(222,45,38,0.8)']

	for row_iter, row in enumerate(table):
		if row_iter == 0:
			continue
		alg_names = []
		alg_times = []
		for i in range(n_cl_algs):
			# add alg names for my clustering algorithms
			alg_names += [table[0,2 + i*5].split()[0]]		
			# add alg times for my clustering algorithms
			alg_times += [float(row[4+i*5])]				
		alg_names += ['GCG']	 				
		alg_times += [row[gcg_col_index]]					# add solving time for gcg auto
		alg_names += [entry for i, entry in enumerate(row) \
						if i > gcg_col_index and (i - gcg_col_index) % 4 == 1 and entry <> '']
		alg_times += [entry for i, entry in enumerate(row) \
						if i > gcg_col_index and (i - gcg_col_index) % 4 == 2 and entry <> '']
		#print 'DEBUG: alg_names = ', alg_names
		#print 'DEBUG: alg_times = ', alg_times
		n_decs = len(alg_names) - n_cl_algs - 1
		# bar for time comparison
		times_bar = go.Bar(
				x=alg_names,
				y=alg_times,
				name='Time for ' +row[0],
				#legendgroup=row[0],
				marker=dict(
					color= cluster_alg_colors +[gcg_auto_color]+ n_decs * [gcg_decs_color])
		)
		data += [times_bar]
		# histograms for cluster distribution of my algs
		for cl_itet in range(n_cl_algs):
			# for cluster i (dbscan, mcl...) get #clusters and their sizes
			cl_distro = all_distros[cl_itet][row_iter-1]
			n_points += [ sum(cl_distro.values()) ]
			if -1 in cl_distro.keys():
				color = (len(cl_distro) -1) *[ cluster_alg_colors[cl_itet] ] + ['rgba(150,150,150,1)']
			else:
				color = len(cl_distro) *[ cluster_alg_colors[cl_itet] ]

			cl_distro = [('CL '+str(label+1),n) 
							if label>-1 else ('NC',n) for label, n in cl_distro.items()]

			
			#ordered_dist = collections.OrderedDict(sorted(cl_distro.items()))	
			distro_i_bar = go.Bar(
					x=zip(*cl_distro)[0],
					y=zip(*cl_distro)[1],
					name=row[0],
					#legendgroup=row[0],
					marker=dict(
						color= color)
			)
			data += [distro_i_bar]

	#n_rows = int(round((n_instances/3)+0.49))
	n_rows = n_instances 	# one row per instace
	n_cols = n_cl_algs + 1 	# in each row we put time comparison sublot + distribution for each clustering alg
	subplot_titles = []
	for inst_name in inst_names:
		subplot_titles += ['Solving time for '+inst_name]
		for i in range(n_cl_algs):
			subplot_titles += ['Cl. distribution by '+table[0,2 + i*5].split()[0]]
	subplot_titles  =tuple(subplot_titles)
	#print 'DEBUG: subplot_titles = ', subplot_titles
	#subplot_titles = tuple((n_cl_algs + 1) * inst_names)
	fig = tools.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
	bar_count = 0
	n_cl_count = 0
	log_time = log(float(time_limit),10)
	for i in range(1, 1+n_rows):
		for j in range(1, 1+n_cols):
			if bar_count < len(data):
				fig.append_trace(data[bar_count], i, j)
				bar_count += 1
				if bar_count % n_cols <> 1:
					n_cl = n_points[n_cl_count]
					n_cl_count+=1
					fig['layout']['yaxis'+str(bar_count)].update(type='log',
									title='# rows (out of %s)'%(n_cl))
				else:
					fig['layout']['yaxis'+str(bar_count)].update(type='log',
									title='Time (sec)', range=[-2, log_time],
									rangemode='tozero')
			else:
				break
	#print 'DEBUG: bar_count = ', bar_count
	fig['layout'].update(height=n_rows*400, width=3300, 
							title='Time Comparison for Similarity '+strategy)
	try:
		py.image.save_as(fig, result_folder+'result_'+run_number+'_str'+strategy+'.png')
		print 'Plot saved.', result_folder+'result_'+run_number+'_str'+strategy+'.png'
	except:
		print 'py.image.save_as FAILED!'
		raise
	#plot_url = py.plot(fig, filename='make-subplots-multiple-with-title')







# read from results/gps folder the files and order them based on:
# instance name, nr. clusters, nr nn points, strategy and cl. algorithm 
# (also makes difference between STD and normal)
def plot_time_and_viz(table, n_cl_algs, run_number, strategy, res_fig_folder, res_gps_folder):
	print 'Plotting time and matrices for strategy ', strategy

	inst_names = table[:,0]
	inst_names = np.delete(inst_names, 0)
	inst_names = list(inst_names)
	n_points = []
	strategy = str(strategy)
	n_instances = len(table) - 1
	gcg_col_index = 2+n_cl_algs*5	# index of GCG column in the table

	# we need this for searching for png files
	alg_names_to_dec = {'DBSCAN':'_dbscan_', 'DBSCAN-STD':'_dbscanSTD_', 
						'MCL':'_mcl_', 'MCL-STD':'_mclSTD_',
						'R-MCL':'_rmcl_', 'R-MCL-STD':'_rmclSTD_',
						'MST':'_mst_', 'MST-STD':'_mstSTD_'}

	# we need this for searching for png files
	alg_names_to_n_cl_column = {'DBSCAN':2, 'DBSCAN-STD':2, 
						'MCL':12, 'MCL-STD':17,
						'R-MCL':22, 'R-MCL-STD':27,
						'MST':32, 'MST-STD':37}					

	alg_names_to_xticks = {'DBSCAN':'DBSCAN', 'DBSCAN-STD':'DBSCAN-2', 
						'MCL':'MCL', 'MCL-STD':'MCL-2',
						'R-MCL':'R-MCL', 'R-MCL-STD':'R-MCL-2',
						'MST':'MST', 'MST-STD':'MST-2'}


	png_files = [png_file for png_file in os.listdir(res_gps_folder) 
					if png_file.endswith('_dist'+strategy+'.png') or png_file.endswith('_sim'+strategy+'.png')]

	fig, axarr = plt.subplots(n_instances, n_cl_algs + 1)
	w = 33
	#h = 1.5*n_instances
	h = (w/16)*9
	fig.set_size_inches(w,h)
	fig.tight_layout()
	# hspace needed so that xticks dont get overwritten, bottom needed so lowest xticks dont get overwritten,
	# top need for title
	fig.subplots_adjust(hspace = 0.5, bottom = 0.05)
	data = []
	gcg_auto_color = (100/255, 100/255, 100/255, 1)
	gcg_decs_color = (100/255, 100/255, 100/255, 0.2)

	if n_cl_algs == 6:
		cluster_alg_colors = [(224,45,38,0.8), (180,45,38,0.8), 
							  (74,224,90,1), (74,180,90,1), 
							  (38,45,224,0.8), (38,45,180,0.8)]
	elif n_cl_algs == 8:
		cluster_alg_colors = [(224,45,38,0.8), (180,45,38,0.8), 
							  (74,224,90,1), (74,180,90,1), 
							  (54,220,220,1), (54,190,190,1), 
							  (38,45,224,1), (38,45,160,1)]						  
	else:
		cluster_alg_colors = n_cl_algs*[(222,45,38,0.8)]

	# normalize the r,g,b values to be in range 0..1 instead in 0..255
	for i, color in enumerate(cluster_alg_colors):
		cluster_alg_colors[i] = (color[0]/255.0, color[1]/255.0, color[2]/255.0, color[3])

	for row_iter, row in enumerate(table):
		if row_iter == 0:
			continue
		alg_names = []	# holds the names of all algorithms (including GCG and all its decs)
		alg_times = []	# holds solv. times of all algorithms (including GCG and all its decs)
		for i in range(n_cl_algs):
			# add alg names for my clustering algorithms
			alg_names += [table[0,2 + i*5].split()[0]]		
			# add alg times for my clustering algorithms
			alg_times += [float(row[4+i*5])]				
		alg_names += ['GCG']	 				
		alg_times += [float(row[gcg_col_index])]					# add solving time for gcg auto
		alg_names += [entry for i, entry in enumerate(row) \
						if i > gcg_col_index and (i - gcg_col_index) % 4 == 1 and (entry <> '' and entry <> ' ')]

		alg_times += [float(entry) for i, entry in enumerate(row) \
						if i > gcg_col_index and (i - gcg_col_index) % 4 == 2 and (entry <> '' and entry <> ' ')]
		#print 'DEBUG: alg_names = ', alg_names
		#print 'DEBUG: alg_times = ', alg_times
		n_decs = len(alg_names) - n_cl_algs - 1
		# bar for time comparison
		width = 1.0
		x = np.arange(len(alg_names))
		#print 'DEBUG plot_time_and_viz(): alg_times = ', alg_times
		axarr[row_iter-1, 0].bar(x, alg_times, align='center', bottom=1e-2, log=True,
							width=width, color=cluster_alg_colors +[gcg_auto_color]+ n_decs * [gcg_decs_color])

		axarr[row_iter-1, 0].set_title('Time for ' + row[0])
		axarr[row_iter-1, 0].set_ylabel('Time (sec)')
		axarr[row_iter-1, 0].set_ylim([1e-2, 3600])
		axarr[row_iter-1, 0].set_xticks(x)
		xTickLabels = [alg_names_to_xticks[item] if item in alg_names_to_xticks.keys() else item 
						for item in alg_names]
		axarr[row_iter-1, 0].set_xticklabels(xTickLabels, rotation='vertical')
		axarr[row_iter-1, 0].set(aspect='auto', adjustable='box-forced')

		# other subplots, i.e. visualizations of matrices
		for i in range(n_cl_algs):
			img_file_names = []
			#print 'png_files: ', png_files
			#print 'row[0]: ', row[0]
			#print 'alg_names_to_dec[alg_names[i-1]]: ', alg_names_to_dec[alg_names[i-1]]
			n_cl_in_dec = row[alg_names_to_n_cl_column[alg_names[i]]]
			n_cl_in_dec = '_'+str(n_cl_in_dec)+'_'

			nn_in_dec = row[alg_names_to_n_cl_column[alg_names[i]]+1]
			nn_in_dec = '_'+nn_in_dec.split('/')[0].split()[0]+'_'


			for png_file in png_files:
				if row[0] in png_file and alg_names_to_dec[alg_names[i]] in png_file:
					img_file_names += [res_gps_folder+png_file]
			
			# DEBUG ONLY
			if row[0] == 'noswot' and alg_names_to_dec[alg_names[i]] == '_dbscanSTD_':
				print 'NEW DEBUG: img_file_names = ', img_file_names


			if len(img_file_names) > 1:
				img_file_names = [img_file_name for img_file_name in img_file_names if 
								 n_cl_in_dec in img_file_name and nn_in_dec in img_file_name]


				if len(img_file_names) > 1:
					print 'img_file_names = ', img_file_names
					raise Exception('We have found more than 1 acceptable png file!')
			
			# DEBUG ONLY
			if row[0] == 'noswot' and alg_names_to_dec[alg_names[i]] == '_dbscanSTD_':
				print 'NEW DEBUG: img_file_names = ', img_file_names

			if len(img_file_names) == 0:
				img_file_name = ''	
			if len(img_file_names) == 1:
				img_file_name = img_file_names[0]

			# case where clustering was not found ot STD was the same as normal
			if img_file_name == '':
				img_file_name = res_fig_folder+'not_found.png'

			image_file = cbook.get_sample_data(img_file_name)
			image = plt.imread(image_file)	
			axarr[row_iter-1, i+1].imshow(image)
			axarr[row_iter-1, i+1].axis('off')
			axarr[row_iter-1, i+1].set_axis_bgcolor(cluster_alg_colors[i])
			axarr[row_iter-1, i+1].set_title(alg_names[i], color = cluster_alg_colors[i])
			



	#fig.suptitle('Time Comparison for Similarity '+strategy)
	fig.savefig(res_fig_folder+'result_'+run_number+'_str'+strategy+'_mat.png')
	print 'Time and matrices saved.', res_fig_folder+'result_'+run_number+'_str'+strategy+'_mat.png'
	plt.close()



def plot_time_for_eval(table, n_cl_algs, run_number, strategy, res_fig_folder, wanted_instances):
	print 'Plotting time and matrices for strategy ', strategy

	n_points = []
	strategy = str(strategy)
	n_instances = len(wanted_instances) 
	gcg_col_index = 2+n_cl_algs*5	# index of GCG column in the table

			
	alg_names_to_xticks = {'DBSCAN':'DBSCAN', 'DBSCAN-STD':'DBSCAN-2', 
						'MCL':'MCL', 'MCL-STD':'MCL-2',
						'R-MCL':'R-MCL', 'R-MCL-STD':'R-MCL-2',
						'MST':'MST', 'MST-STD':'MST-2'}



	fig, axarr = plt.subplots(n_instances)
	w = 33
	#h = 1.5*n_instances
	h = (w/16)*9
	fig.set_size_inches(w/9,h)
	fig.tight_layout()
	# hspace needed so that xticks dont get overwritten, bottom needed so lowest xticks dont get overwritten,
	# top need for title
	fig.subplots_adjust(hspace = 0.5, bottom = 0.05)
	data = []
	gcg_auto_color = (100/255, 100/255, 100/255, 1)
	gcg_decs_color = (100/255, 100/255, 100/255, 0.2)

	if n_cl_algs == 6:
		cluster_alg_colors = [(224,45,38,0.8), (180,45,38,0.8), 
							  (74,224,90,1), (74,180,90,1), 
							  (38,45,224,0.8), (38,45,180,0.8)]
	elif n_cl_algs == 8:
		cluster_alg_colors = [(224,45,38,0.8), (180,45,38,0.8), 
							  (74,224,90,1), (74,180,90,1), 
							  (54,220,220,1), (54,190,190,1), 
							  (38,45,224,1), (38,45,160,1)]						  
	else:
		cluster_alg_colors = n_cl_algs*[(222,45,38,0.8)]

	# normalize the r,g,b values to be in range 0..1 instead in 0..255
	for i, color in enumerate(cluster_alg_colors):
		cluster_alg_colors[i] = (color[0]/255.0, color[1]/255.0, color[2]/255.0, color[3])

	row_iter = 0
	for i, row in enumerate(table):
		if i == 0:
			continue

		if row[0] not in wanted_instances:
			continue

		print 'Adding the instance: ', row[0]

		alg_names = []	# holds the names of all algorithms (including GCG and all its decs)
		alg_times = []	# holds solv. times of all algorithms (including GCG and all its decs)
		for i in range(n_cl_algs):
			# add alg names for my clustering algorithms
			alg_names += [table[0,2 + i*5].split()[0]]		
			# add alg times for my clustering algorithms
			alg_times += [float(row[4+i*5])]				
		alg_names += ['GCG']	 				
		alg_times += [float(row[gcg_col_index])]					# add solving time for gcg auto
		alg_names += [entry for i, entry in enumerate(row) \
						if i > gcg_col_index and (i - gcg_col_index) % 4 == 1 and (entry <> '' and entry <> ' ')]

		alg_times += [float(entry) for i, entry in enumerate(row) \
						if i > gcg_col_index and (i - gcg_col_index) % 4 == 2 and (entry <> '' and entry <> ' ')]
		#print 'DEBUG: alg_names = ', alg_names
		#print 'DEBUG: alg_times = ', alg_times
		n_decs = len(alg_names) - n_cl_algs - 1
		# bar for time comparison
		width = 1.0
		x = np.arange(len(alg_names))
		#print 'DEBUG plot_time_and_viz(): alg_times = ', alg_times
		axarr[row_iter-1].bar(x, alg_times, align='center', bottom=1e-2, log=True,
							width=width, color=cluster_alg_colors +[gcg_auto_color]+ n_decs * [gcg_decs_color])

		axarr[row_iter-1].set_title('Time for ' + row[0])
		axarr[row_iter-1].set_ylabel('Time (sec)')
		axarr[row_iter-1].set_ylim([1e-2, 3600])
		axarr[row_iter-1].set_xticks(x)
		xTickLabels = [alg_names_to_xticks[item] if item in alg_names_to_xticks.keys() else item 
						for item in alg_names]
		axarr[row_iter-1].set_xticklabels(xTickLabels, rotation='vertical')
		axarr[row_iter-1].set(aspect='auto', adjustable='box-forced')

		row_iter += 1

	if row_iter <> 5:
		raise Exception('Error: not all instances are found, but only ', row_iter)


	#fig.suptitle('Time Comparison for Similarity '+strategy)
	fig.savefig(res_fig_folder+'result_'+run_number+'_str'+strategy+'_mat.png')
	print 'Time and matrices saved.', res_fig_folder+'result_'+run_number+'_str'+strategy+'_eval.png'
	plt.close()
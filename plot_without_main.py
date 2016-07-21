import csv
import os
from getpass import getuser
from plot_result import plot_time_and_viz
import numpy as np


username = getuser()
res_folder = '/home/'+username+'/Desktop/results/'
res_fig_folder = '/home/'+username+'/Desktop/adjusted_results_figures/'
res_gps_folder = res_folder + 'gps/'



n_algorithms = 8

for run_number in [7]:

	for strategy in [1,2,5,6]:
		result_csv_name = 'result'+str(run_number)+'_str'+str(strategy)+'.csv'
		csv_file_path = res_folder+result_csv_name
		table = []
		#print 'Reading csv file...'
		if os.path.isfile(csv_file_path):
			csv_file = open(csv_file_path, "rb")
			reader = csv.reader(csv_file, delimiter = ',', quoting = csv.QUOTE_NONE, quotechar='')
			for row in reader:
				table += [row]
				#print row 		# for debuging only!
			csv_file.close()
			print 'We have found and read ', csv_file_path
		else:
			print 'NOT FOUND ', csv_file_path
			continue

		table = np.array(table)
		# plots time with visualizations of matrices
		plot_time_and_viz(table = table, n_cl_algs = n_algorithms, run_number=str(run_number), 
			res_gps_folder=res_gps_folder, strategy=strategy, res_fig_folder=res_fig_folder)
			

			

		

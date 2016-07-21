import csv
import os
from getpass import getuser
from plot_result import plot_time_for_eval
import numpy as np


username = getuser()
res_folder = '/home/'+username+'/Desktop/plot_for_eval/results_tmp/'
res_fig_folder = '/home/'+username+'/Desktop/plot_for_eval/'
res_gps_folder = res_folder + 'gps/'

if not os.path.exists(res_fig_folder):
	os.mkdir(res_fig_folder)

n_algorithms = 8


wanted_instances = [
	'gesa2', 'new46', 'pp08aCUTS', 'neos-820146', 'n4-3'
]

wanted_instances_2 = [
	'neos-826650', 'tanglegram2', 'sp98ir', 'neos-942830', 'noswot'
]


first_row = ['foo', 'bar']

for strategy in [9]:

	table = []

	for run_number in [0,1,2,3,4]:
		result_csv_name = 'result'+str(run_number)+'_str'+str(strategy)+'.csv'
		csv_file_path = res_folder+result_csv_name
		
		#print 'Reading csv file...'
		if os.path.isfile(csv_file_path):
			csv_file = open(csv_file_path, "rb")
			reader = csv.reader(csv_file, delimiter = ',', quoting = csv.QUOTE_NONE, quotechar='')
			for i, row in enumerate(reader):
				if run_number <> 0 and i == 0: 
					continue
				table += [row]
				#print row 		# for debuging only!
			csv_file.close()
			print 'We have found and read ', csv_file_path
		else:
			print 'NOT FOUND ', csv_file_path
			continue

	table = np.array(table)

	# plots time with visualizations of matrices
	plot_time_for_eval(table = table, n_cl_algs = n_algorithms, run_number=str(11), 
		strategy=strategy, res_fig_folder=res_fig_folder, wanted_instances=wanted_instances)			

			

		

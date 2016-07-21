import os
import logging
import numpy as np
from cluster_dbscan import dbscan
from cluster_mcl import mcl
from cluster_r_mcl import r_mcl
from cluster_mst import mst
from plot_result import plot_time, plot_time_and_viz
from preprocess import getStrategies
from detect_and_run import detect_and_run
from shutil import copyfile
import csv
import sys
import socket
from getpass import getuser
from tempfile import NamedTemporaryFile

'''
	NOTE: 
		instaces with names that contain _ or / are not allowed!!!!
		when adding new algorithm: 
			channge: algorithms, 
					 alg_col_map['new alg'] = <starting column>,
					 first_row,
					 make new <alg_name>_dec_name (like d_dec_name or mcl_dec_name),
					 first_row[17], row[18]....
					 add list that contains all cluster distirbutions for that algorithm
					 (something like dbscan_distros or mcl_distros)
					 update all_distros
					 Last-step
					 figure width (in plot_result.py)
					 colors of the subplots (in plot_result.py)

	WARNING:
		This code should be tested more on the orlab computers.
		Change: added R-MCL.			 
'''
def main(run_number, strategy, time_lim = 5):
	
	#run_number = '0'
	username = getuser()
	computer = socket.gethostname()
	computer = computer[0:7] 
	gcg_folder = '/home/'+username+'/gcg/'
	test_folder = gcg_folder+'check/testset/'
	testset_in_name = 'testset_in' + run_number
	testset_out_name = 'testset_out' + run_number + '_str' + str(strategy)
	result_csv_name = 'result'+run_number+'_str'+strategy+'.csv'
	decs_folder = gcg_folder+'check/decs/'+testset_in_name+'.nopresolve/'
	gcg_res_folder = gcg_folder+'check/results/'
	res_folder = '/home/'+username+'/Desktop/results/'
	res_fig_folder = '/home/'+username+'/Desktop/results_figures/'
	csv_file_path = '/home/'+username+'/Desktop/results/'+result_csv_name
	gcg_run_command = 'cd '+gcg_folder+' && make TEST='+testset_out_name+' SETTINGS=nopresolve TIME='+time_lim+' test'
	strategy = int(strategy)
	instance_files = []
	algorithms = ['dbscan', 'dbscanSTD', 'mcl', 'mclSTD', 'rmcl', 'rmclSTD', 'mst', 'mstSTD']
	n_algorithms = len(algorithms)
	dbscan_distros = []		# saves distributions for this algorithm
	dbscanSTD_distros = []
	mcl_distros = [] 		
	mclSTD_distros = [] 
	rmcl_distros = [] 		
	rmclSTD_distros = []		
	mst_distros = []
	mstSTD_distros = []

	n_original_instaces = 0 	# number of instances to solve

	# step 1: copy all .mps files to results folder
	testset_in = open(test_folder+testset_in_name+'.test', 'r')
	for line in testset_in:
		if not '/' in line:
			continue 	# skip empty lines
		n_original_instaces += 1
		line = line.rstrip()
		if '_' in line:
			raise Exception('Underscore is not allowed in instance or path name!')
		instance_files += [line]
		try:
			copyfile(line, res_folder + line.rsplit('/', 1)[1])
		except:
			raise Exception('File not found: ', line)
		# fix the name of the .dec file in the file itself
		dec_name_temp = line.rsplit('/', 1)[1].rsplit('.',1)[0]
		dec_temp = open(line, 'r')			# this part of code we need to check that instance
		dec_data = dec_temp.readlines()		# name is the same in the file itself
		if dec_data[0].startswith('NAME'):
			dec_data[0] = 'NAME          ' + dec_name_temp + '\n'
		dec_temp.close()
		dec_fixed = open(line, 'w')
		dec_fixed.writelines(dec_data)
		dec_fixed.close()		
	testset_in.close()

	# step 2: create all decs by gcg 
	# detected dictionary look like this: 'noswot':[('auto', 1.0), (S_3_0, 1.0), (S_4_0, 2.0)]
	detected = detect_and_run(gcg_folder, res_folder, instance_files, time_lim, run_number)

	instance_files.sort()

	# step 2b: move decs by gcg to results folder
	#os.system('mv '+decs_folder+'*.dec '+res_folder)


	# step 3a: put paths of .dec made by gcg to output testset file
	# step 3b: get the highest number of decompositions per instance (needed for the table size)
	testset_out = open(test_folder+testset_out_name+'.test', 'w')
	cur_nr_of_decs = 0
	change = False
	last_instance_name = ''
	# put all instance files to test_out. this is for automatic gcg detection 
	#(we want to know what dec was chosen by gcg as the best)
	'''for instance_file in instance_files:
		if os.path.isfile(instance_file):
			testset_out.write(instance_file + '\n')
		else:
			raise Exception('ERROR: %s not found.' %(instance_file))
	for dec_file in gcg_dec_files:
		instance_name = dec_file.split('_')[0]
		# check if instance instance in results folder is .mps or .lp
		if os.path.isfile(res_folder+instance_name+'.mps'):
			testset_out.write(res_folder + instance_name + '.mps;')
			testset_out.write(res_folder + dec_file + '\n')
		elif os.path.isfile(res_folder+instance_name+'.lp'):	
			testset_out.write(res_folder + instance_name + '.lp;')
			testset_out.write(res_folder + dec_file + '\n')
		else:
			print 'some error has occured!!!'
			print 'Instance file has wrong ending! ', instance_name
		if instance_name <> last_instance_name:
			cur_nr_of_decs = 1
			last_instance_name = instance_name
			if cur_nr_of_decs > most_decs:
				most_decs = cur_nr_of_decs
		else: 
			cur_nr_of_decs += 1
			if cur_nr_of_decs > most_decs:
				most_decs = cur_nr_of_decs'''
	
	# we fix this because we always show up to 3 best gcg decs
	most_decs = 3

	# step 4 and 5: run clustering algorithms
	#				and save relevant data in the table
	row_len = 3 + n_algorithms*5 + 4*most_decs
	first_row = [' ' for i in range(row_len)]
	first_row[0] = 'INSTANCE'
	first_row[1] = 'TYPE'
	first_row[2] = 'DBSCAN #clusters'
	first_row[3] = 'DBSCAN non-clustered'
	first_row[4] = 'DBSCAN time'
	first_row[5] = 'DBSCAN first lp time'
	first_row[6] = 'DBSCAN final dual bound'
	first_row[7] = 'DBSCAN-STD #clusters'
	first_row[8] = 'DBSCAN-STD non-clustered'
	first_row[9] = 'DBSCAN-STD time'
	first_row[10] = 'DBSCAN-STD first lp time'
	first_row[11] = 'DBSCAN-STD final dual bound'
	first_row[12] = 'MCL #clusters'
	first_row[13] = 'MCL non-clustered'
	first_row[14] = 'MCL time'
	first_row[15] = 'MCL first lp time'
	first_row[16] = 'MCL final dual bound'
	first_row[17] = 'MCL-STD #clusters'
	first_row[18] = 'MCL-STD non-clustered'
	first_row[19] = 'MCL-STD time'
	first_row[20] = 'MCL-STD first lp time'
	first_row[21] = 'MCL-STD final dual bound'
	first_row[22] = 'R-MCL #clusters'
	first_row[23] = 'R-MCL non-clustered'
	first_row[24] = 'R-MCL time'
	first_row[25] = 'R-MCL first lp time'
	first_row[26] = 'R-MCL final dual bound'
	first_row[27] = 'R-MCL-STD #clusters'
	first_row[28] = 'R-MCL-STD non-clustered'
	first_row[29] = 'R-MCL-STD time'
	first_row[30] = 'R-MCL-STD first lp time'
	first_row[31] = 'R-MCL-STD final dual bound'
	first_row[32] = 'MST #clusters'
	first_row[33] = 'MST non-clustered'
	first_row[34] = 'MST time'
	first_row[35] = 'MST first lp time'
	first_row[36] = 'MST final dual bound'
	first_row[37] = 'MST-STD #clusters'
	first_row[38] = 'MST-STD non-clustered'
	first_row[39] = 'MST-STD time'
	first_row[40] = 'MST-STD first lp time'
	first_row[41] = 'MST-STD final dual bound'
	first_row[42] = 'GCG'
	table = [first_row]
	table
	instance_row_map = {}
	# if algorithm is mine, then it looks like 'dbscan:2, mcl:7....'
	# but if algorithm is gcg dec, then it is '<row_index> <dec_name>':12....
	# it basically gives the starting column index for this algorithm or algorithm-row pair
	alg_col_map = {}
	row_index = 1
	for iteration, instance_file in enumerate(instance_files):
		instance = instance_file.rsplit('/', 1)[1]
		instance_type = instance_file.rsplit('/', 2)[1]
		file_type = instance.rsplit('.',1)[1]
		instance = instance.rsplit('.',1)[0]
		# get and write dbscan data -------------------------------------------
		alg_col_map['dbscan'] = 2
		alg_col_map['dbscanSTD'] = 7
		dbscan_found, d_nc1, d_nonc1, d_distr, d_dec, n, \
		dbscanSTD_found, dSTD_nc1, dSTD_nonc1, dSTD_distr, dSTD_dec = dbscan(instance_file, res_folder,strategy)
		dbscan_distros += [d_distr]
		dbscanSTD_distros += [dSTD_distr]

		alg_col_map['mcl'] = 12
		alg_col_map['mclSTD'] = 17
		mcl_found, mcl_nc, mcl_nonc, mcl_distro, mcl_dec, n2, mclSTD_found, \
		mclSTD_nc, mclSTD_nonc, mclSTD_distro, mclSTD_dec = mcl(instance_file, res_folder,strategy)
		mcl_distros += [mcl_distro]
		mclSTD_distros += [mclSTD_distro]

		alg_col_map['rmcl'] = 22
		alg_col_map['rmclSTD'] = 27
		rmcl_found, rmcl_nc, rmcl_nonc, rmcl_distro, rmcl_dec, n3, rmclSTD_found, \
		rmclSTD_nc, rmclSTD_nonc, rmclSTD_distro, rmclSTD_dec = r_mcl(instance_file, res_folder, strategy)
		rmcl_distros += [rmcl_distro]
		rmclSTD_distros += [rmclSTD_distro]

		alg_col_map['mst'] = 32
		alg_col_map['mstSTD'] = 37
		mst_found, mst_nc, mst_nonc, mst_distro, mst_dec, n4, mstSTD_found, \
		mstSTD_nc, mstSTD_nonc, mstSTD_distro, mstSTD_dec = mst(instance_file, res_folder, strategy)
		mst_distros += [mst_distro]
		mstSTD_distros += [mstSTD_distro]

		if n <> n2 or n <> n3 or n <> n4:
			raise Exception('ERROR in the size of input data! %i != %i.' %(n, n2, n3, n4))
		print 'DEBUG row_len= ', row_len
		print 'DEBUG most_decs= ', most_decs
		# these must be the same as those in coresponding clustering functions!!!!
		d_dec_name = d_dec + '.dec'
		dSTD_dec_name = dSTD_dec + '.dec'
		mcl_dec_name = mcl_dec + '.dec'
		mclSTD_dec_name = mclSTD_dec + '.dec'
		rmcl_dec_name = rmcl_dec + '.dec'
		rmclSTD_dec_name = rmclSTD_dec + '.dec'
		mst_dec_name = mst_dec + '.dec'
		mstSTD_dec_name = mstSTD_dec + '.dec'
		if dbscan_found:
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + d_dec_name):
				raise Exception('Cannot find ', res_folder + d_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + d_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + d_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type	
		if dbscanSTD_found:
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + dSTD_dec_name):
				raise Exception('Cannot find ', res_folder + dSTD_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + dSTD_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + dSTD_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type		
		if mcl_found:
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + mcl_dec_name):
				raise Exception('Cannot find ', res_folder + mcl_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + mcl_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + mcl_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type	
		if mclSTD_found:
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + mclSTD_dec_name):
				raise Exception('Cannot find ', res_folder + mclSTD_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + mclSTD_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + mclSTD_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type
		if rmcl_found:
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + rmcl_dec_name):
				raise Exception('Cannot find ', res_folder + rmcl_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + rmcl_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + rmcl_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type	
		if rmclSTD_found:
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + rmclSTD_dec_name):
				raise Exception('Cannot find ', res_folder + rmclSTD_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + rmclSTD_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + rmclSTD_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type				
		if mst_found:			
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + mst_dec_name):
				raise Exception('Cannot find ', res_folder + mst_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + mst_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + mst_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type
		if mstSTD_found:			
			# check if coresponding file exists:
			if not os.path.isfile(res_folder + mstSTD_dec_name):
				raise Exception('Cannot find ', res_folder + mstSTD_dec_name)
			# write instance_path; my_dec_path pair
			if file_type == 'mps':
				testset_out.write(res_folder + instance + '.mps;')
				testset_out.write(res_folder + mstSTD_dec_name + '\n')
			elif file_type == 'lp':	
				testset_out.write(res_folder + instance + '.lp;')
				testset_out.write(res_folder + mstSTD_dec_name + '\n')
			else:
				print 'ERROR: instance is neither .lp nor .mps, but ', file_type			
		gcg_dec_found = False			
		# row must have 2 + n_algorithms*5 + 1 + 4*n_gcg_decompositions
		# for 1 algorithm (dbscan only) it is 12
		row = [' ' for i in range(row_len)]
		row[0] = instance
		instance_row_map[instance] = row_index
		row[1] = instance_type
		row[alg_col_map['dbscan']] = str(d_nc1)
		row[alg_col_map['dbscan']+1] = str(d_nonc1) + '/' + str(n)
		row[alg_col_map['dbscan']+2] = str(-1)
		row[alg_col_map['dbscanSTD']] = str(dSTD_nc1)
		row[alg_col_map['dbscanSTD']+1] = str(dSTD_nonc1) + '/' + str(n)
		row[alg_col_map['dbscanSTD']+2] = str(-1)

		row[alg_col_map['mcl']] = str(mcl_nc)
		row[alg_col_map['mcl']+1] = str(mcl_nonc) + '/' + str(n)
		row[alg_col_map['mcl']+2] = str(-1)
		row[alg_col_map['mclSTD']] = str(mclSTD_nc)
		row[alg_col_map['mclSTD']+1] = str(mclSTD_nonc) + '/' + str(n)
		row[alg_col_map['mclSTD']+2] = str(-1)

		row[alg_col_map['rmcl']] = str(rmcl_nc)
		row[alg_col_map['rmcl']+1] = str(rmcl_nonc) + '/' + str(n)
		row[alg_col_map['rmcl']+2] = str(-1)
		row[alg_col_map['rmclSTD']] = str(rmclSTD_nc)
		row[alg_col_map['rmclSTD']+1] = str(rmclSTD_nonc) + '/' + str(n)
		row[alg_col_map['rmclSTD']+2] = str(-1)

		row[alg_col_map['mst']] = str(mst_nc)
		row[alg_col_map['mst']+1] = str(mst_nonc) + '/' + str(n)
		row[alg_col_map['mst']+2] = str(-1)
		row[alg_col_map['mstSTD']] = str(mstSTD_nc)
		row[alg_col_map['mstSTD']+1] = str(mstSTD_nonc) + '/' + str(n)
		row[alg_col_map['mstSTD']+2] = str(-1)
		
		# get and write gcg data -------------------------------------------
		#if instance in [dec_file.split('_')[0] for dec_file in gcg_dec_files]:
		gcg_dec_found = True
		row[2+n_algorithms*5] = True
		if gcg_dec_found:
			# add information for all decompositions for coresponding instance
			# fill row[8+x*4] where x is redni broj of decomposition
			# this is list of information (e.g. C_3_5) for every decomposition of current instance
			'''this_instance_gcg_decs = ['_'.join(dec_file.rsplit('.',1)[0].split('_')[1:])	
										for dec_file in gcg_dec_files if dec_file.split('_')[0] == instance]'''	
			this_instance_gcg_decs = detected[instance]													
			for x, (this_dec, time) in enumerate(this_instance_gcg_decs):
				if this_dec == 'auto':
					col_index = 2+n_algorithms*5
					row[col_index] = time
				else:
					col_index = 3+n_algorithms*5+(x-1)*4
					alg_col_map[str(row_index)+' '+this_dec] = col_index
					row[col_index] = this_dec
					row[col_index+1] = time

		#writer.writerow(row)
		# instead of writing rows, we save them in list 'table' and then later write them all together
		table += [row]
		row_index += 1 
		for foo in range(2):
			print ' '

	testset_out.close()

	res_gps_folder = res_folder + 'gps/'
	if not os.path.isdir(res_gps_folder):
		os.system('mkdir '+res_gps_folder)

	
	# step 6: run gcg on testset_ou.test
	print 'Running gcg with following command: ', gcg_run_command
	os.system(gcg_run_command)

	# step 6b: correct the error in gp files and change from pdf to png
	# set terminal pdf
	# set output "airland1R1_0_11.pdf"
	gp_files = [gp_file for gp_file in os.listdir(res_gps_folder) if gp_file.endswith('.gp')]
	if [] == gp_files:
		raise Exception('No .gp files found in ', res_gps_folder)
	else:
		for gp_file in gp_files:
			gp_name_pure = gp_file.rsplit('.',1)[0]
			gp_file = res_gps_folder + gp_file
			
			gp_temp = open(gp_file, 'r')			# this part of code we need to check that instance
			gp_data = gp_temp.readlines()		# name is the same in the file itself
			if gp_data[0].startswith('set terminal') and gp_data[1].startswith('set output'):
				gp_data[0] = 'set terminal png\n'
				gp_data[1] = 'set output \"' + gp_name_pure + '.png\"' + '\n'	
			else:
				raise Exception('GP file has unexpected fromat!', gp_file)	
			gp_temp.close()
			gp_fixed = open(gp_file, 'w')
			gp_fixed.writelines(gp_data)
			gp_fixed.close()

	# when running this we can stupid errors in matrix visualizations, so run it alone for each of gps 
	#os.system('cd ' + res_gps_folder + ' && gnuplot *')
	for gp in os.listdir(res_gps_folder):
		os.system('cd ' + res_gps_folder + ' && gnuplot '+ gp)



	# step 7: get the data from gcg statistics:
	# for each instance: time, master programstatistics - root node - first lp time, final dual time
	print 'Reading .out file and saving data in the table....'
	gcg_out_files = [f for f in os.listdir(gcg_res_folder) 
			if f.endswith('.out') and f.startswith('check.'+testset_out_name) and computer in f]
	if len(gcg_out_files) < 1:
		raise Exception('Error: Out file not found!')
	elif len(gcg_out_files) > 1:
		raise Exception('Error: More than 1 out file found!')
	out_file = open(gcg_res_folder+gcg_out_files[0], 'r')
	get_status = False
	inst_name = ' '
	alg_name = ' '
	solved = False
	interrupted = False
	other_status = False
	time = 'time'
	master_stat = False
	root_node_stat = False
	first_lp_time = 'first_lp_time'
	final_dual_bound = 'final_dual_bound'
	row_index = -1
	start_col_index = -1
	tmp_n_inst = n_original_instaces
	time_table = [0 for i in range(len(instance_row_map))]
	for line in out_file:
		if line.startswith('GCG> read ') and '.dec' in line:
			# get the name of dec file and put the following info in coresponding table fields
			dec_file_name = line.rsplit('/',1)[1].rstrip()
			inst_name = dec_file_name.split('_')[0]
			row_index = instance_row_map[inst_name]
			if dec_file_name.split('_')[1] in algorithms:
				alg_name = dec_file_name.split('_')[1]			# gives something like dbscan or mcl
				start_col_index = alg_col_map[alg_name] + 2 	# at this index we write time
			else: 
				# gives something like C_3_0
				alg_name = dec_file_name.split('_',1)[1].split('.')[0] 
				# at this index we write time
				start_col_index = alg_col_map[str(row_index)+' '+alg_name] + 1 
			if start_col_index == 0 or start_col_index == 1:
				raise Exception('start_col_index is wrong')
			get_status = True
			#print 'get_status= ', get_status
			continue
			
		# get the data from auto detection and put it in 'GCG' column
		'''if line.startswith('Solving Time (sec) :') and tmp_n_inst > 0:	
			# write time of auto dec in gcg column
			table[n_original_instaces-tmp_n_inst+1][n_algorithms*5 + 2] =line.rsplit(' ',1)[1].rstrip()
			tmp_n_inst -= 1
			continue'''

		if line.startswith('SCIP Status') and get_status:
			#----------------------------------------------------------------------------------------
			# TODO: check for outher possible outputs!!!!!!!!!!
			#----------------------------------------------------------------------------------------
			if 'problem is solved [optimal solution found]' in line:
				solved = True
			elif 'solving was interrupted [time limit reached]' in line:
				interrupted = True
			else:
				other_status = True 	
			get_status = False	
			#print 'solved, interrupted = ', solved, interrupted
			continue
		if line.startswith('Solving Time (sec) :') and solved:	
			time = line.rsplit(' ',1)[1]
			table[row_index][start_col_index] = str(time).rstrip()
			start_col_index += 1
			#print 'time found'
			continue
		elif line.startswith('Solving Time (sec) :') and interrupted:
			time = line.rsplit(' ',1)[1]
			table[row_index][start_col_index] = str(time).rstrip()
			start_col_index += 1
			#interrupted = False
			#print 'time found'
			continue
		elif line.startswith('Solving Time (sec) :') and other_status:
			time = line.rsplit(' ',1)[1]
			table[row_index][start_col_index] = str(time).rstrip()
			start_col_index += 1
			#other_status = False
			#print 'time found'
			continue
		if line.startswith('Master Program statistics') and (solved or interrupted):	
			master_stat = True
			continue
		if line.startswith('Root Node') and master_stat:	
			root_node_stat = True
			continue
		if 'First LP Time' in line and root_node_stat:
			first_lp_time = line.rsplit(' ',1)[1]
			table[row_index][start_col_index] = str(first_lp_time).rstrip()
			start_col_index += 1
			continue
		if 'Final Dual Bound' in line and root_node_stat:
			table[row_index][start_col_index] = line.rsplit(' ',1)[1].rstrip()
			start_col_index += 1
			solved = False
			interrupted = False
			other_status = False
			master_stat = False
			root_node_stat = False
			continue

	out_file.close()
	table = np.array(table)

	png_files = [png_file for png_file in os.listdir(res_gps_folder) if png_file.endswith('.png')]
	
	# Last-step: this part is for writeing artifical time to those decs where the distros are the same
	# Also, we copy png files for those cases, so we can visualize them also when clustering is the same

	# iterate instances
	strategy = str(strategy)
	iteration = 0
	for d_d, dSTD_d in zip(dbscan_distros, dbscanSTD_distros):
		if d_d.items() == dSTD_d.items():
			table[iteration+1][alg_col_map['dbscanSTD'] + 2] = table[iteration+1][alg_col_map['dbscan'] + 2]
			# copy the png file for STD case
			for png_file in png_files:
				if table[iteration+1][0] in png_file and '_dbscan_' in png_file and '_dist'+strategy in png_file:
					source_f = res_gps_folder+png_file
					dest_f = res_gps_folder+png_file.replace('_dbscan_', '_dbscanSTD_')
					print 'Coping files with command:'
					print '		cp '+source_f+' '+ dest_f	
					os.system('cp '+source_f+' '+ dest_f)
					break
			
		iteration += 1	

	# iterate instances
	iteration = 0
	for mcl_d, mclSTD_d in zip(mcl_distros, mclSTD_distros):
		if mcl_d.items() == mclSTD_d.items():
			table[iteration+1][alg_col_map['mclSTD'] + 2] = table[iteration+1][alg_col_map['mcl'] + 2]
			# copy the png file for STD case
			for png_file in png_files:
				if table[iteration+1][0] in png_file and '_mcl_' in png_file and '_sim'+strategy in png_file:
					source_f = res_gps_folder+png_file
					dest_f = res_gps_folder+png_file.replace('_mcl_', '_mclSTD_')
					print 'Coping files with command:'
					print '		cp '+source_f+' '+ dest_f	
					os.system('cp '+source_f+' '+ dest_f)
					break
			
		iteration += 1	

	# iterate instances
	iteration = 0
	for rmcl_d, rmclSTD_d in zip(rmcl_distros, rmclSTD_distros):
		if rmcl_d.items() == rmclSTD_d.items():
			table[iteration+1][alg_col_map['rmclSTD'] + 2] = table[iteration+1][alg_col_map['rmcl'] + 2]
			# copy the png file for STD case
			for png_file in png_files:
				if table[iteration+1][0] in png_file and '_rmcl_' in png_file and '_sim'+strategy in png_file:
					source_f = res_gps_folder+png_file
					dest_f = res_gps_folder+png_file.replace('_rmcl_', '_rmclSTD_')
					print 'Coping files with command:'
					print '		cp '+source_f+' '+ dest_f	
					os.system('cp '+source_f+' '+ dest_f)
					break
			
		iteration += 1	

	# iterate instances
	iteration = 0
	for mst_d, mstSTD_d in zip(mst_distros, mstSTD_distros):
		if mst_d.items() == mstSTD_d.items():
			table[iteration+1][alg_col_map['mstSTD'] + 2] = table[iteration+1][alg_col_map['mst'] + 2]
			# copy the png file for STD case
			for png_file in png_files:
				if table[iteration+1][0] in png_file and '_mst_' in png_file and '_dist'+strategy in png_file:
					source_f = res_gps_folder+png_file
					dest_f = res_gps_folder+png_file.replace('_mst_', '_mstSTD_')
					print 'Coping files with command:'
					print '		cp '+source_f+' '+ dest_f	
					os.system('cp '+source_f+' '+ dest_f)
					break
			
		iteration += 1		


	print 'Writing csv file...'
	csv_file = open(csv_file_path, "wb")
	writer = csv.writer(csv_file, delimiter = ',', quoting = csv.QUOTE_NONE, quotechar='')
	for row in table:
		writer.writerow(row)
		#print row 		# for debuging only!
	csv_file.close()


	print 'Plotting the result...'
	all_distros = [dbscan_distros, dbscanSTD_distros, mcl_distros, mclSTD_distros, 
						rmcl_distros, rmclSTD_distros, mst_distros, mstSTD_distros]

	# plots time with visualizations of matrices
	plot_time_and_viz(table = table, n_cl_algs = n_algorithms, run_number=run_number, res_gps_folder=res_gps_folder,
		strategy=strategy, res_fig_folder=res_fig_folder)

	try:
		# plots time with clusters distributions
		plot_time(table = table, n_cl_algs = n_algorithms, all_distros=all_distros, run_number=run_number, 
			strategy=strategy, result_folder=res_fig_folder, time_limit=time_lim)
	except:
		print 'Some error occured in plot_time() for strategy', strategy

	print '\n*******************'
	print '*******Done.*******'
	print '*******************'

# Arguments should be of the form <testset_nr> <time>
print 'Arguments passed:', str(sys.argv)
if len(sys.argv) < 2:
	raise Exception('At least 1 argument is needed for this script!')
elif len(sys.argv) == 2:
	time_lim = str(5)
else:
	time_lim = str(sys.argv[2])
run_number = str(sys.argv[1])
#strategy = str(sys.argv[2])
# iterate all strategies
for strategy in [9]:
	strategy = str(strategy)
	print 'RUN_NUMBER = %s, STRATEGY = %s, TIME_LIMIT = %s ' %(run_number,strategy,time_lim)	

	# create (if it doesnt exist already) log file	
	open('error'+run_number+'_'+strategy+'.log', 'a').close()
	logging.basicConfig(level=logging.DEBUG, filename='error'+run_number+'_'+strategy+'.log')

	try:
		main(run_number=run_number, strategy=strategy, time_lim=time_lim)
	except:
		print 'There is an exception...'
		logging.exception("Oops:")
		raise
from __future__ import division
import numpy as np 
import csv 
import os
import socket
import operator

# gcg_folder
# res_folder:			folder where we store our results
# instance_files:		list of instance files in the form of abs. path
# res_gcg_decs:			folder in which we move gcg's decs (should be in results folder)
def detect_and_run(gcg_folder, res_folder, instance_files, time, run_number):
	test_folder = gcg_folder+'check/testset/'
	gcg_res_folder = gcg_folder+'check/results/'
	res_gcg_decs = res_folder + 'gcg_decs/'
	run_number = str(run_number)
	computer = socket.gethostname()
	computer = computer[0:7] 
	decs_folder_detect = gcg_folder+'check/decs/detect'+run_number+'.nopresolve/'

	# if there is a csv file for our dataset, read it, otherwise create it
	if os.path.isfile(res_gcg_decs+'gcg_data_'+run_number+'.csv'):
		csv_file = open(res_gcg_decs+'gcg_data_'+run_number+'.csv', 'rU')
		data = list(list(rec) for rec in csv.reader(csv_file, delimiter=','))
		csv_file.close()

		if len(data) > 1:
			print 'len(data): ', len(data)
			raise Exception('CSV file with detected decs has more than 1 rows.')

		data = data[0]
	else:
		open(res_gcg_decs+'gcg_data_'+run_number+'.csv', 'a').close()
		data = []

	# every third elements is nistance name or its decomposition
	existing_instances = [item for i, item in enumerate(data) if i % 3 == 0]

	# clear decs folder before detection 
	if not os.path.isdir(decs_folder_detect):
		os.system('mkdir '+decs_folder_detect)
	if [] <> os.listdir(decs_folder_detect):
		os.system('rm '+decs_folder_detect+'*.dec')	

	testset_detect = open(test_folder+'detect'+run_number+'.test', 'w')
	to_detect_and_run_inst = []
	new_inst_found = False
	for instance_file in instance_files:
		instance = instance_file.rsplit('/',1)[1]
		if instance not in existing_instances:
			print 'new instance which is not in existing_instances: ', instance
			# put it in new testset which we will use to detect
			testset_detect.write(instance_file+ '\n')
			to_detect_and_run_inst += [instance_file]
			new_inst_found = True
	testset_detect.close()

	if not new_inst_found:
		print 'We did not find any new instances in this test set.'
		print 'NO need for auto_detection.'
		
	else:
		print 'We need to do auto_detection.'
		# run gcg to detect all decompostions for new instances
		os.system('cd '+gcg_folder+' && make TEST=detect'+run_number+' MODE=detectall SETTINGS=nopresolve test')

		# step 2a: save all gcg dec files to a list 
		gcg_dec_files = [f for f in os.listdir(decs_folder_detect) if f.endswith('.dec') and not f.startswith('BLANK_')]
		gcg_dec_files.sort()

		# step 2b: move decs by gcg to results folder
		os.system('mv '+decs_folder_detect+'*.dec '+res_gcg_decs)


		# make new testset with instances alone and instances; all decs
		testset_detect_run = open(test_folder+'detect_run'+run_number+'.test', 'w')

		# make new testset with instances alone
		for instance_file in to_detect_and_run_inst:
			if os.path.isfile(instance_file):
				testset_detect_run.write(instance_file + '\n')
			else:
				raise Exception('ERROR: %s not found.' %(instance_file))

		# for each instance add also all of its decs
		for dec_file in gcg_dec_files:
			instance_name = dec_file.split('_')[0]
			# check if instance instance in results folder is .mps or .lp
			if os.path.isfile(res_folder+instance_name+'.mps'):
				testset_detect_run.write(res_folder + instance_name + '.mps;')
				testset_detect_run.write(res_gcg_decs + dec_file + '\n')
			elif os.path.isfile(res_folder+instance_name+'.lp'):	
				testset_detect_run.write(res_folder + instance_name + '.lp;')
				testset_detect_run.write(res_gcg_decs + dec_file + '\n')
			else:
				raise Exception('Instance file has wrong ending! ', instance_name)

		testset_detect_run.close()

		# run this testset
		os.system('cd '+gcg_folder+' && make TEST=detect_run'+run_number+' SETTINGS=nopresolve TIME='+time+' test')

		# save the data to the existing .csv file but at the end of the file
		# add everything to data list
		data_extention = []

		gcg_out_files = [f for f in os.listdir(gcg_folder+'check/results/') 
				if f.endswith('.out') and f.startswith('check.'+'detect_run'+run_number) and computer in f]
		if len(gcg_out_files) < 1:
			raise Exception('Error: Out file not found!')
		elif len(gcg_out_files) > 1:
			raise Exception('Error: More than 1 out file found!')

		out_file = open(gcg_res_folder+gcg_out_files[0], 'r')

		instance_found = False
		dec_found = False
		master_stat = False
		root_node_stat = False
		name = ''
		time = ''
		final_dual_bound = ''
		for line in out_file:
			if line.startswith('GCG> read ') and ('.mps' in line or '.lp' in line):
				instance_found = True
				name_to_save = line.rsplit('/',1)[1].rstrip()
				continue
				
			if line.startswith('GCG> read ') and '.dec' in line and instance_found:
				dec_found = True
				instance_found = False
				name_to_save = line.rsplit('/',1)[1].rstrip()
				continue	

			# get the data from auto detection and put it in 'GCG' column
			if line.startswith('Solving Time (sec) :') and (dec_found or instance_found):	
				# write time of auto dec in gcg column
				time = line.rsplit(' ',1)[1].rstrip()
				name = name_to_save
				continue

			if line.startswith('Master Program statistics'):	
				master_stat = True
				continue
			if line.startswith('Root Node') and master_stat:	
				root_node_stat = True
				continue
			if 'Final Dual Bound' in line and root_node_stat:
				final_dual_bound = line.rsplit(' ',1)[1].rstrip()
				master_stat = False
				root_node_stat = False
				if name == '' or time == '' or final_dual_bound == '':
					print name, time, final_dual_bound
					raise Exception('Something went wrong!')
				data_extention += [name, time, final_dual_bound] 
				continue

		data += data_extention

		csv_updated_file = open(res_gcg_decs+'gcg_data_'+run_number+'.csv', 'wb')
		spamwriter = csv.writer(csv_updated_file, delimiter=',', quoting = csv.QUOTE_NONE)
		spamwriter.writerow(data)
		csv_updated_file.close()

	#  this dictionry look like this: 'noswot':[('C_3_0', 1.0), (S_3_0, 5.0), (S_4_0, 2.0)]
	inst_to_dec_time = {}

	# this dictionry look like this: 'noswot':[('auto', 1.0), (S_3_0, 1.0), (S_4_0, 2.0)]
	instance_to_best = {}

	for instance_file in instance_files:
		instance_name = instance_file.rsplit('/', 1)[1]		# this is for example 'noswot.mps'
		instance_pure = instance_name.rsplit('.',1)[0] 		# this is for example 'noswot'

		temp_list = []

		for i, item in enumerate(data):
			if i % 3 == 0 and instance_pure in item and '.dec' in item:
				# dec name to add
				to_add = item.rsplit('.',1)[0].split('_',1)[1]
				temp_list += [tuple([to_add, data[i+1]])]

		inst_to_dec_time[instance_pure] = temp_list	


	# for each key sort the lists and take first 2 (sort accoring to tuple element 1) 
	for instance_pure, value in inst_to_dec_time.iteritems():	
		tmp_list = value[:]
		tmp_list.sort(key=lambda x: (float(x[1])))

		auto = ''
		for i, item in enumerate(data):
			if instance_pure in item and ('.mps' in item or '.lp' in item):
				auto = tuple(['auto', data[i+1]])

		if auto == '':
			print 'instance_pure ', instance_pure
			raise Exception('We were not able to find instance %s in the list!!!' %instance_pure)
		if len(tmp_list) > 2:	
			instance_to_best[instance_pure] = [auto, tmp_list[0], tmp_list[1], tmp_list[2]]
		elif len(tmp_list) > 1:
			instance_to_best[instance_pure] = [auto, tmp_list[0], tmp_list[1]]
		elif len(tmp_list) == 1:
			instance_to_best[instance_pure] = [auto, tmp_list[0]]
		else:
			print 'instance_pure: ', instance_pure
			raise Exception('ERROR: it seems that no auto decs found for this instance, whoch is not possible!')
			
	# this dictionry look like this: 'noswot':[('auto', 1.0), (S_3_0, 1.0), (S_4_0, 2.0)]
	#print 'DEBUG: detect_and_run() instance_to_best = ', instance_to_best
	#raise Exception('WAIT')
	return instance_to_best


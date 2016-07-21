from __future__ import division
import numpy as np
from getpass import getuser
from os import listdir
from os.path import isfile, join
import gc
import parse
import preprocess as pp
from show_matrix import save_matrix_fig, save_matrix_file, viz_2d

username = getuser() 
directory = '/home/igorpesic/Desktop/instances/gcg/gap/'
files = []
#files+= ["gap4_2.txt.lp"]
#files = [f for f in listdir(directory) if isfile(join(directory, f))]

files += ['/home/'+username+'/Desktop/instances/toTest/new46.lp']	
files += ['/home/'+username+'/Desktop/instances/toTest/new47.lp']	
files += ['/home/'+username+'/Desktop/instances/miplib/fiber.mps']
files += ['/home/'+username+'/Desktop/instances/miplib/10teams.mps']
files += ['/home/'+username+'/Desktop/instances/miplib/rout.mps']
files += ['/home/'+username+'/Desktop/instances/miplib/noswot.mps']
files += ['/home/'+username+'/Desktop/instances/miplib/modglob.mps']
files += ['/home/'+username+'/Desktop/instances/miplib/gesa2.mps']
files += ['/home/'+username+'/Desktop/instances/miplib/vpm2.mps']
files += ['/home/'+username+'/Desktop/instances/ORLib/airland/airland1R2.mps']

for file in files:
	if not (file.endswith('.mps') or file.endswith('.lp')):
		continue
	#path = directory + file
	path = file
	data, row_names = parse.read(path)  
	sim_matrix = pp.strategy(data, 'sim', 2)
	print 'Occupancy: ', len(sim_matrix.nonzero()[0]) / (sim_matrix.shape[0] * sim_matrix.shape[0])
	#save_matrix_file(dist_matrix, directory, file+'dist_mat')
	#save_matrix_fig(data, directory, file)
	print file, data.shape
	gc.collect()
	#print 'dist_matrix.shape', dist_matrix.shape
	print '-----------------------------------------------'
	
	
	
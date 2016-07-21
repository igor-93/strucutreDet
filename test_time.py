from __future__ import division
from _cluster_dbscan import dbscan
from _cluster_mcl import mcl
from _cluster_r_mcl import r_mcl
from _cluster_mst import mst
from getpass import getuser
import matplotlib.pyplot as plt

username = getuser() 

# where to save the resulting .dec files and figures and .npy
res_folder = ''
inst_folder = ''
if username == 'igorpesic':
	inst_folder = '/home/'+username+'/Desktop/instances/'
	res_folder = '/home/'+username+'/Desktop/time_test/'
elif username == 'pesic':
	inst_folder = '/home/'+username+'/Desktop/results/'
	res_folder = '/home/'+username+'/Desktop/results/'





files = []
# small matrices
#files += [inst_folder+'noswot.mps']
#files += [inst_folder+'fiber.mps']

# mid matrices
#files += [inst_folder+'miplib/pp08aCUTS.mps']	
	

# mid matrices
#files += [inst_folder+'neos-820146.mps']
#files += [inst_folder+'toTest/new46.lp']
#files += [inst_folder+'n4-3.mps']
#files += [inst_folder+'wachplan.mps']

# big matrices
#files += [inst_folder+'neos18.mps']
files += [inst_folder+'rmatr100-p5.mps']
files += [inst_folder+'neos-948126.mps']



for iteration, instance_file in enumerate(files):

	n, time, density = dbscan(instance_file, res_folder, 1)
	print 'DBSCAN: n = %i, time = %f, density = %f' %(n, time, density)
	print ' '
	print ' '
	n, time, density = mcl(instance_file, res_folder, 1)
	print 'MCL: n = %i, time = %f, density = %f' %(n, time, density)
	print ' '
	print ' '
	n, time, density = r_mcl(instance_file, res_folder, 1)
	print 'R-MCL: n = %i, time = %f, density = %f' %(n, time, density)
	print ' '
	print ' '
	n, time, density = mst(instance_file, res_folder, 1)
	print 'MST: n = %i, time = %f, density = %f' %(n, time, density)
	print ' '
	print ' '



print 'TIME TEST FINISHED.'
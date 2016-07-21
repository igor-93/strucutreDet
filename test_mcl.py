from __future__ import division
from cluster_mcl import mcl
from getpass import getuser
import matplotlib.pyplot as plt
import numpy as np

username = getuser() 

# where to save the resulting .dec files and figures and .npy
res_folder = '/home/'+username+'/Desktop/test_results/'

if username == 'igorpesic':
	files = []
	#files += ['/home/'+username+'/Desktop/instances/ORLib/airland/airland1R2.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/fiber.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/pp08aCUTS.mps']		
	#files += ['/home/'+username+'/Desktop/instances/toTest/new46.lp']	
	#files += ['/home/'+username+'/thesis/from_desktop/instances/miplib/gesa2.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/10teams.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/rout.mps']
	files += ['/home/igorpesic/thesis/from_desktop/instances/miplib/noswot.mps']
	#files += ['/home/igorpesic/Desktop/instances/miplib2010/neos-820146.mps']
	#files += ['/home/igorpesic/Desktop/instances/miplib2010/wachplan.mps']
	#files += ['/home/igorpesic/Desktop/instances/miplib2010/neos-911880.mps']
	#files += ['/home/igorpesic/Desktop/instances/miplib2010/n4-3.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/modglob.mps']
	#files += ['/home/'+username+'/Desktop/instances2/neos18.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib2010/wachplan.mps']
	#files += ['/home/igorpesic/Desktop/instances/miplib2010/neos-1224597.mps']
	#
	#files += ['/home/'+username+'/Desktop/instances/miplib/vpm2.mps']
	#files += ['/home/'+username+'/Desktop/instances/ORLib/airland/airland1R2.mps']
	#files += ['/home/'+username+'/Desktop/instances/ORLib/setcover/scp410.mps']
	#files += ['/home/'+username+'/Desktop/instances/ORLib/setcover/scp63.mps']
else: 
	files = []
	#files += ['/home/'+username+'/Desktop/instances/toTest/p1250-10.txt.lp']
	#files += ['/home/'+username+'/Desktop/instances/miplib/pp08aCUTS.mps']		# STD
	#files += ['/home/'+username+'/Desktop/instances/toTest/new46.lp']	
	#files += ['/home/'+username+'/Desktop/instances/toTest/new47.lp']	
	#files += ['/home/'+username+'/Desktop/instances/miplib/fiber.mps']
	files += ['/home/'+username+'/instances/miplib/noswot.mps']


colors = ['xk-','xr-', 'xb-', 'xg-', 'xy-', 'xk-', 'xm-']

for iteration, instance_file in enumerate(files):

	#mcl(instance_file, res_folder, 1)
	mcl(instance_file, res_folder, 1)
	#mcl(instance_file, res_folder, 5)
	#mcl(instance_file, res_folder, 6)


	'''x = to_test.keys()
	x.sort()
	y = [to_test[i] for i in x]
	plt.xlabel('Inflate factor')
	plt.ylabel('Nr. of clusters / nr. of total rows')
	plt.plot(x, y, colors[iteration])'''


	for foo in range(2):
		print ' '

print 'MCL TEST FINISHED.'
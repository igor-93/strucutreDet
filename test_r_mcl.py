from __future__ import division
from cluster_r_mcl import r_mcl
from getpass import getuser
import matplotlib.pyplot as plt

username = getuser() 

# where to save the resulting .dec files and figures and .npy
res_folder = "/home/igorpesic/Desktop/test_results/"

if username == 'igorpesic':
	files = []
	#files += ['/home/'+username+'/Desktop/instances/toTest/p1250-10.txt.lp']
	#files += ['/home/'+username+'/Desktop/instances/miplib/pp08aCUTS.mps']		# STD
	#files += ['/home/'+username+'/Desktop/instances/toTest/new46.lp']	
	#files += ['/home/'+username+'/Desktop/instances/toTest/new47.lp']	
	#files += ['/home/'+username+'/Desktop/instances/miplib/fiber.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/10teams.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/rout.mps']
	files += ['/home/'+username+'/Desktop/instances/miplib/noswot.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/modglob.mps']
	#files += ['/home/'+username+'/Desktop/instances/miplib/gesa2.mps']
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
x = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]

for iteration, instance_file in enumerate(files):

	r_mcl(instance_file, res_folder, 1)
	#r_mcl(instance_file, res_folder, 2)
	#r_mcl(instance_file, res_folder, 5)
	#r_mcl(instance_file, res_folder, 6)



	
	
	'''if len(nr_cl_with_diff_params) < 17: 
		nr_cl_with_diff_params += (17-len(nr_cl_with_diff_params)) * [nr_cl_with_diff_params[-1]]

	nr_cl_with_diff_params = [item / n for item in nr_cl_with_diff_params]
	try:
		plt.plot(x, nr_cl_with_diff_params, colors[iteration], label=str(n)+' rows')
	except:
		print 'x ', x
		print 'y ', nr_cl_with_diff_params'''

	for foo in range(2):
		print ' '

#plt.xlabel('Inflate factor')
#plt.ylabel('Nr. of clusters / nr. of total rows')
#plt.show()
print 'MCL TEST FINISHED.'
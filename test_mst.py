from __future__ import division

from cluster_mst import mst




# where to save the resulting .dec files and figures and .npy
res_folder = "/home/igorpesic/Desktop/test_results/"

files = []
files += ['/home/igorpesic/Desktop/instances/ORLib/airland/airland1R2.mps']
files += ['/home/igorpesic/Desktop/instances/miplib/fiber.mps']
files += ['/home/igorpesic/Desktop/instances/miplib/pp08aCUTS.mps']		
files += ['/home/igorpesic/Desktop/instances/toTest/new46.lp']	
files += ['/home/igorpesic/Desktop/instances/miplib/gesa2.mps']
#files += ['/home/'+username+'/Desktop/instances/miplib/10teams.mps']
#files += ['/home/'+username+'/Desktop/instances/miplib/rout.mps']
#files += ['/home/'+username+'/Desktop/instances/miplib/noswot.mps']
files += ['/home/igorpesic/Desktop/instances/miplib2010/neos-820146.mps']
files += ['/home/igorpesic/Desktop/instances/miplib2010/wachplan.mps']
files += ['/home/igorpesic/Desktop/instances/miplib2010/neos-911880.mps']
files += ['/home/igorpesic/Desktop/instances/miplib2010/n4-3.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/gesa2.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/vpm2.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/airland/airland1R2.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp410.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp63.mps']


for iteration, instance_path in enumerate(files):

	mst(instance_path, res_folder,9)

	for foo in range(2):
		print ' '

print 'DBSCAN TEST FINISHED.'




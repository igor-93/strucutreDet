from __future__ import division
from cluster_Agglomerative import agg


# where to save the resulting .dec files and figures and .npy
res_folder = "/home/igorpesic/Desktop/test_results/"

files = []
#files += ['/home/igorpesic/Desktop/instances/miplib/noswot.mps']
#files += ['/home/igorpesic/Desktop/instances/toTest/new46.lp']	
#files += ['/home/igorpesic/Desktop/instances/toTest/new47.lp']	
#files += ['/home/igorpesic/Desktop/instances/miplib/fiber.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/modglob.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/gesa2.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/vpm2.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/pp08aCUTS.mps']
files += ['/home/igorpesic/Desktop/instances/ORLib/airland/airland1R2.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp410.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp63.mps']


for iteration, instance_path in enumerate(files):

	agg(instance_path, res_folder,2)

	for foo in range(2):
		print ' '

print 'Affinity Propagation TEST FINISHED.'




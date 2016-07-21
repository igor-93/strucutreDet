from __future__ import division
from cluster_dbscan import dbscan
import preprocess as pp 
import numpy as np
import parse
import matplotlib.pyplot as plt
from math import sqrt

username = 'igorpesic'
# where to save the resulting .dec files and figures and .npy
res_folder = "/home/igorpesic/Desktop/test_results/"

files = []
#files += ['/home/igorpesic/Desktop/instances/ORLib/airland/airland1R2.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/fiber.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/pp08aCUTS.mps']		
#files += ['/home/igorpesic/Desktop/instances/toTest/new46.lp']	
files += ['/home/igorpesic/thesis/from_desktop/instances/miplib/gesa2.mps']
#files += ['/home/'+username+'/Desktop/instances/miplib/10teams.mps']
#files += ['/home/'+username+'/Desktop/instances/miplib/rout.mps']
#files += ['/home/igorpesic/thesis/from_desktop/instances/miplib/noswot.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib2010/neos-820146.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib2010/wachplan.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib2010/neos-911880.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib2010/n4-3.mps']
#files += ['/home/igorpesic/Desktop/instances/miplib/vpm2.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/airland/airland1R2.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp410.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp63.mps']


for iteration, instance_path in enumerate(files):

	dbscan(instance_path, res_folder,1)

	for foo in range(2):
		print ' '

print 'DBSCAN TEST FINISHED.'




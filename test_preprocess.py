from __future__ import division
from math import sqrt
#from cluster_dbscan import dbscan
import scipy.io
from scipy.sparse import *
from scipy import *
import numpy as np

import preprocess as pp 
import parse


# where to save the resulting .dec files and figures and .npy
res_folder = "/home/igorpesic/Desktop/test_results/"

files = []
files += ['/home/igorpesic/Desktop/instances/miplib/noswot.mps']
files += ['/home/igorpesic/Desktop/instances/toTest/new46.lp']	
files += ['/home/igorpesic/Desktop/instances/toTest/new47.lp']	
files += ['/home/igorpesic/Desktop/instances/miplib/fiber.mps']
files += ['/home/igorpesic/Desktop/instances/miplib/modglob.mps']
files += ['/home/igorpesic/Desktop/instances/miplib/gesa2.mps']
files += ['/home/igorpesic/Desktop/instances/miplib/vpm2.mps']
files += ['/home/igorpesic/Desktop/instances/ORLib/airland/airland1R2.mps']
files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp410.mps']
#files += ['/home/igorpesic/Desktop/instances/ORLib/setcover/scp63.mps']


for iteration, instance_path in enumerate(files):
	instances = instance_path.rsplit('/', 1)[0] + '/'
	file = instance_path.rsplit('/', 1)[1]
	input_type  = '.' + file.rsplit('.', 1)[1]
	file = file.rsplit('.', 1)[0]
	data, row_names = parse.read(instances + file + input_type)
	print '#################################################################################'
	for strategy in [1,2,5,6]:
		print '###########################'
		try:
			sim_matrix = np.load(res_folder+file+'_sim'+str(strategy)+'.npy')
			print 'sim matrix %s found.' %(res_folder+file+'_sim'+str(strategy)+'.npy')
		except:
			print 'sim matrix %s NOT found!!!!' %(res_folder+file+'_sim'+str(strategy)+'.npy')
			sim_matrix = pp.strategy(data, 'sim',strategy)	
			np.save(res_folder+file+'_sim'+str(strategy), sim_matrix)
		
		print 'For strategy = ', strategy
		occupancy = np.count_nonzero(sim_matrix) / (sim_matrix.shape[0] * sim_matrix.shape[1]) * 100
		#q = 100 - occupancy + 0.01*occupancy
		q = 100 - 0.01*occupancy
		print 'Percentile 0.1 = ', np.percentile(a=sim_matrix, q=q)
		#q = 100 - occupancy + 0.1*occupancy
		q = 100 - 0.1*occupancy
		print 'Percentile 1 = ', np.percentile(a=sim_matrix, q=q)
		#print 'Avg of row perc. = ', np.mean([np.percentile(a=row, q=1) for row in sim_matrix])
		print 'Occupancy = ', occupancy

		try:
			dist_matrix = scipy.io.mmread(res_folder+file+'_dist'+str(strategy)).tocsr()
			print 'Distance matrix %s found.' %(res_folder+file+'_dist'+str(strategy))
		except:
			print 'Distance matrix %s NOT found!!!!' %(res_folder+file+'_dist'+str(strategy))
			dist_matrix = pp.strategy(data, 'distance',strategy)	
			scipy.io.mmwrite(res_folder+file+'_dist'+str(strategy),dist_matrix)
		occupancy2 = len(dist_matrix.data) / (dist_matrix.shape[0] * dist_matrix.shape[1]) * 100
		print 'Occupancy = ', occupancy, occupancy2	
		#q = 5
		#print 'Percentile 1.5 = ', np.percentile(a=dist_matrix.data, q=5, axis=None)
		q = 10
		print 'Percentile 10 = ', np.percentile(a=dist_matrix.data, q=q, axis=None)

	'''
	print '###########################'
	strategy = 2 	# intersection
	try:
		sim_matrix = np.load(res_folder+file+'_sim'+str(strategy)+'.npy')
		print 'sim matrix %s found.' %(res_folder+file+'_sim'+str(strategy)+'.npy')
	except:
		print 'sim matrix %s NOT found!!!!' %(res_folder+file+'_sim'+str(strategy)+'.npy')
		sim_matrix = pp.strategy(data, 'sim',strategy)	
		np.save(res_folder+file+'_sim'+str(strategy), sim_matrix)
	
	print 'For strategy = ', strategy
	occupancy = np.count_nonzero(sim_matrix) / (sim_matrix.shape[0] * sim_matrix.shape[1]) * 100
	q = 100 - occupancy + 0.01*occupancy
	print 'Percentile 0.1 = ', np.percentile(a=sim_matrix, q=q)
	q = 100 - occupancy + 0.1*occupancy
	print 'Percentile 1 = ', np.percentile(a=sim_matrix, q=q)
	#print 'Avg of row perc. = ', np.mean([np.percentile(a=row, q=1) for row in sim_matrix])
	print 'Occupancy = ', occupancy

	try:
		dist_matrix = scipy.io.mmread(res_folder+file+'_dist'+str(strategy)).tocsr()
		print 'Distance matrix %s found.' %(res_folder+file+'_dist'+str(strategy))
	except:
		print 'Distance matrix %s NOT found!!!!' %(res_folder+file+'_dist'+str(strategy))
		dist_matrix = pp.strategy(data, 'distance',strategy)	
		scipy.io.mmwrite(res_folder+file+'_dist'+str(strategy),dist_matrix)
	occupancy = len(dist_matrix.data) / (dist_matrix.shape[0] * dist_matrix.shape[1]) * 100	
	q = 5
	#print 'Percentile 1.5 = ', np.percentile(a=dist_matrix.data, q=5, axis=None)
	q = 10
	print 'Percentile 10 = ', np.percentile(a=dist_matrix.data, q=q, axis=None)


	print '###########################'
	strategy = 5 	# jaccard
	try:
		sim_matrix = np.load(res_folder+file+'_sim'+str(strategy)+'.npy')
		print 'sim matrix %s found.' %(res_folder+file+'_sim'+str(strategy)+'.npy')
	except:
		print 'sim matrix %s NOT found!!!!' %(res_folder+file+'_sim'+str(strategy)+'.npy')
		sim_matrix = pp.strategy(data, 'sim',strategy)	
		np.save(res_folder+file+'_sim'+str(strategy), sim_matrix)
	
	print 'For strategy = ', strategy
	occupancy = np.count_nonzero(sim_matrix) / (sim_matrix.shape[0] * sim_matrix.shape[1]) * 100
	print 'occupancy = ', occupancy
	print 'np.count_nonzero(sim_matrix) = ', np.count_nonzero(sim_matrix)
	q = 100 - occupancy + 0.01*occupancy
	print 'Percentile 0.1 = ', np.percentile(a=sim_matrix, q=q)
	q = 100 - occupancy + 0.1*occupancy
	print 'Percentile 1 = ', np.percentile(a=sim_matrix, q=q)
	#print 'Avg of row perc. = ', np.mean([np.percentile(a=row, q=1) for row in sim_matrix])
	print 'Occupancy = ', occupancy
	
	try:
		dist_matrix = scipy.io.mmread(res_folder+file+'_dist'+str(strategy)).tocsr()
		print 'Distance matrix %s found.' %(res_folder+file+'_dist'+str(strategy))
	except:
		print 'Distance matrix %s NOT found!!!!' %(res_folder+file+'_dist'+str(strategy))
		dist_matrix = pp.strategy(data, 'distance',strategy)	
		scipy.io.mmwrite(res_folder+file+'_dist'+str(strategy),dist_matrix)
	occupancy = len(dist_matrix.data) / (dist_matrix.shape[0] * dist_matrix.shape[1]) * 100	
	q = 5
	#print 'Percentile 1.5 = ', np.percentile(a=dist_matrix.data, q=5, axis=None)
	q = 10
	print 'Percentile 10 = ', np.percentile(a=dist_matrix.data, q=q, axis=None)


	print '###########################'
	strategy = 6 	# cosine
	try:
		sim_matrix = np.load(res_folder+file+'_sim'+str(strategy)+'.npy')
		print 'sim matrix %s found.' %(res_folder+file+'_sim'+str(strategy)+'.npy')
	except:
		print 'sim matrix %s NOT found!!!!' %(res_folder+file+'_sim'+str(strategy)+'.npy')
		sim_matrix = pp.strategy(data, 'sim',strategy)	
		np.save(res_folder+file+'_sim'+str(strategy), sim_matrix)
	
	print 'For strategy = ', strategy
	occupancy = np.count_nonzero(sim_matrix) / (sim_matrix.shape[0] * sim_matrix.shape[1]) * 100
	q = 100 - occupancy + 0.01*occupancy
	print 'Percentile 0.1 = ', np.percentile(a=sim_matrix, q=q)
	q = 100 - occupancy + 0.1*occupancy
	print 'Percentile 1 = ', np.percentile(a=sim_matrix, q=q)
	#print 'Avg of row perc. = ', np.mean([np.percentile(a=row, q=1) for row in sim_matrix])
	print 'Occupancy = ', occupancy

	try:
		dist_matrix = scipy.io.mmread(res_folder+file+'_dist'+str(strategy)).tocsr()
		print 'Distance matrix %s found.' %(res_folder+file+'_dist'+str(strategy))
	except:
		print 'Distance matrix %s NOT found!!!!' %(res_folder+file+'_dist'+str(strategy))
		dist_matrix = pp.strategy(data, 'distance',strategy)	
		scipy.io.mmwrite(res_folder+file+'_dist'+str(strategy),dist_matrix)
	occupancy = len(dist_matrix.data) / (dist_matrix.shape[0] * dist_matrix.shape[1]) * 100	
	q = 5
	#print 'Percentile 1.5 = ', np.percentile(a=dist_matrix.data, q=5, axis=None)
	q = 10
	print 'Percentile 10 = ', np.percentile(a=dist_matrix.data, q=q, axis=None)'''

	print '#################################################################################'
	

print 'Pre-process TEST FINISHED.'




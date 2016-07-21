from __future__ import division
from decimal import *
import numpy as np
import operator
import functools
import itertools
from math import log

# E-step
# Expectation step of the EM Algorithm
#
# INPUT:
# means          : Mean for each Gaussian KxD
# weights        : Weight vector 1xK for K Gaussians
# X              : Input data NxD
#
# N is number of data points
# D is the dimension of the data points
# K is number of gaussians

# OUTPUT:
# logLikelihood  : Log-likelihood (a scalar).
# gamma          : NxK matrix of responsibilities for N datapoints and K gaussians.
def e_step(means, weights, X):
	#print 'Doing e_step...'
	K, D = means.shape
	N = X.shape[0]
	gamma = np.ones((N,K))
	bernoulli_mat = np.ones((N,K))
	weights = np.log(weights) # we will use log weights
	logLikelihood = 0
	logLikelihood_failed = False

	min_value = 1e-8
	# build the matrix of bernoulli distribution
	for n in range(N):
		for k in range(K):
			x = X[n,:]
			mean = means[k,:]	
			'''res = np.zeros(x.shape[0])
			res[x==1] = mean[x==1]
			res[x==0] = 1 - mean[x==0]
			res[res<1e-8] = 1e-8
			res[res > 1 - 1e-8] = 1 - 1e-8
			bernoulli_mat[n,k] = np.sum(np.log(res))'''
			#bernoulli_mat[n,k] = reduce(operator.mul, 
			#						[mean[i] if x[i] == 1 else 1-mean[i] for i in range(len(x))], 1)
			
			bernoulli_mat[n,k] = np.sum([log(mean[i]) if x[i] == 1 else log(1-mean[i]) for i in range(len(x))])

		gamma[n,:] = weights + bernoulli_mat[n,:]
		gamma[n,:] -= np.max(gamma[n,:] ) 
		gamma[n,gamma[n,:] < -10 ] = -10
		gamma[n,:] = np.exp(gamma[n,:])  
		gamma[n,:] /= np.sum(gamma[n,:]) 

		if not logLikelihood_failed:
			try:
				logLikelihood += log(np.sum(np.exp(weights + bernoulli_mat[n,:])))	
			except ValueError:
				logLikelihood_failed = True
				logLikelihood = -float("inf")
		

			

	'''# is column vector of len N
	down_part = np.dot(bernoulli_mat, weights)
	for n in range(N):
		# elementwise mulitiplication of weights and rows of bernoulli_mat
		gamma[n,:] = np.multiply(weights, bernoulli_mat[n,:])

		# divide each row if matrix with the down part
		gamma[n,:] /= down_part[n]'''

	'''
	for n in range(N):
		gamma[n,:] = weights + bernoulli_mat[n,:]

		gamma[n,:] -= np.max(gamma[n,:] ) 

		gamma[n,gamma[n,:] < -10 ] = -10

		gamma[n,:] = np.exp(gamma[n,:]) 
		    
		gamma[n,:] /= np.sum(gamma[n,:]) 	'''

	gamma[gamma < min_value] = min_value
	gamma[gamma > 1 - min_value] = 1 - min_value

	return gamma, logLikelihood


# M-step
# Maximization step of the EM Algorithm
#
# INPUT:
# gamma          : NxK matrix of responsibilities for N datapoints and K gaussians.
# X              : Input data (NxD matrix for N datapoints of dimension D).
#
# N is number of data points
# D is the dimension of the data points
# K is number of gaussians
#
# OUTPUT:
# means          : Mean for each gaussian (KxD).
# weights        : Vector of weights of each gaussian (1xK).
def m_step(gamma, X):
	#print 'Doing m_step...'
	N, K = gamma.shape
	D = X.shape[1]

	Nk = np.sum(gamma, axis=0)

	# update means
	rev_Nk_diag = [[1/Nk[j] if j == i else 0 for j in range(K)] for i in range(K)]
	means = np.dot(rev_Nk_diag, np.dot(gamma.T, X))
	means[means < 1e-8] = 1e-8
	means[means > 1 - 1e-8] = 1 - 1e-8


	# update weights
	weights = Nk / N
	weights /= np.sum(weights)

	return weights, means


# Log Likelihood estimation
#
# INPUT:
# means          : Mean for each Gaussian KxD
# weights        : Weight vector 1xK for K Gaussians
# X              : Input data NxD
#
# where N is number of data points
# D is the dimension of the data points
# K is number of gaussians
#
# OUTPUT:
# logLikelihood  : log-likelihood
def getLogLikelihood(means, weights, X):
	#print 'DEBUG: Calculating log likelihood...'
	K, D = means.shape
	N = X.shape[0]
	logLikelihood = 0
	weights = np.log(weights)
	logLikelihood = 0

	bernoulli_mat = np.ones((N,K))
	# build the matrix of bernoulli distribution
	for n in range(N):
		for k in range(K):
			x = X[n,:]
			mean = means[k,:]
			bernoulli_mat[n,k] = np.sum([log(mean[i]) if x[i] == 1 else log(1-mean[i]) for i in range(len(x))])
			#bernoulli_mat[n,k] = reduce(operator.mul, 
			#						[mean[i] if x[i] == 1 else 1-mean[i] for i in range(len(x))], 1)
		
		try:
			logLikelihood += log(np.sum(np.exp(weights + bernoulli_mat[n,:])))	
		except:
			print 'bernoulli_mat[n,:] ', bernoulli_mat[n,:]
			print 'weights ', weights
			print 'np.sum(np.exp(weights + bernoulli_mat[n,:])) ', np.sum(np.exp(weights + bernoulli_mat[n,:]))
			raise
	
	'''for n in range(N): 	
		logLikelihood += log(np.sum(np.exp(weights + bernoulli_mat[n,:])))'''

	return logLikelihood

# EM algorithm for estimation Bernoulli mixture model
#
# INPUT:
# data           : input data, N observations, D dimensional (NxD)
# K              : number of mixture components (modes)
#
# OUTPUT:
# means          : Mean for each gaussian (KxD).
# weights        : Vector of weights of each gaussian (1xK).
def em_implementation(data, K, n_iters = 25):
	weights = np.array([1/K for i in range(K)])
	# make the data binary
	data[data<>0] = 1
	N, D = data.shape
	n_ones = int(N / K)
	means = np.ones((K,D))

	oldLogLik = -float("inf")
	# init means
	for k in range(K):
		mean = np.random.uniform(0.35,0.65,D)
		means[k] = mean

	for stepNr in range(n_iters):
		# E Step
		#oldLogLik, gamma = e_step(means, weights, data)
		gamma, logLik = e_step(means, weights, data)
		# M Step
		weights, means = m_step(gamma, data)
		gc.collect()
		#print 'logLik = ', logLik
		# termination criteria
		if abs(oldLogLik - logLik) <= 0.001 and logLik > -float("inf"):
			#print 'means = ', means
			print 'DEBUG: condition fulfiled at iteration = ', stepNr
			break

		oldLogLik = logLik	
		

	# label is tha column index of the max value in the row
	labels = np.argmax(gamma, axis = 1)
	skipped = []

	# fil the empty clusters
	for k in range(K):
		if k not in labels:
			skipped += [k]
	while skipped <> []:
		ind = np.argwhere(labels == np.amax(labels))
		ind = list(itertools.chain(*ind))
		labels[ind] = skipped[0]
		del skipped[0]

	return labels	
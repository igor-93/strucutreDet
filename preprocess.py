from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from math import sqrt

def getStrategies():
	#return [1,2,3,4,5,6,7]
	return [1,2,5,6,7]

# main function
def strategy(matrix, str_type, str_nr):
	if str_nr not in [1,2,3,4,5,6,7,8,9]:
		print 'strategy error: strategy number is wrong: %s ' %(str_nr)
		raise Exception('strategy error: strategy number is wrong: %s ' %(str_nr))
	if str_type not in ['sim', 'distance']:
		print 'strategy error: strategy type is wrong: %s ' %(str_type)
		raise Exception('strategy error: type parameter is wrong: %s ' %(str_type))
	if str_nr == 1:
		return strategy_1(matrix, str_type)
	elif str_nr == 2:	
		return strategy_2(matrix, str_type)
	elif str_nr == 3:	
		raise Exception('Strategy 3 not in use anymore!!!')	
	elif str_nr == 4:	
		raise Exception('Strategy 4 not in use anymore!!!')		
	elif str_nr == 5:	
		return strategy_5(matrix, str_type)		
	elif str_nr == 6:	
		return strategy_6(matrix, str_type)	
	elif str_nr == 7:	
		raise Exception('Strategy 7 not in use anymore!!!')			
	elif str_nr == 8:	
		return strategy_8(matrix, str_type)
	elif str_nr == 9:	
		return strategy_9(matrix, str_type)				

# input: N*D matrix
# strategy: Johnson
#				a/2 * (1/(a+b) + 1/(a+c)), with a = len(AND), a+b = non-zero count of row1 and 
#				a+c = non-zero count of row2
# output: symmetrical N*N matrix
# example:
#		row1	0 1 1 1 0 0 
#		row2    1 0 0 1 0 0
#		score	0+0+0+9+1+1 = 11 	-> result[row1][row2] = 11 = result[row2][row1]
def strategy_1(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 1'
	for i in range(n):
		for j in range(i + 1,n):
			# size of intersection
			a = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			if a == 0:
				score = 0
			else:
				# a+b and a+c
				ab = np.count_nonzero(matrix[i,:])
				ac = np.count_nonzero(matrix[j,:])
				if a == ab and ab == ac: 	# this is only for numerical reasons
					score = 1
				else:
					score = (a/2.0) * (1/ab + 1/ac)
			res_matrix[i,j] = score	
			res_matrix[j,i] = score

	res_matrix *= 0.999999
	# each point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)		
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 1'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 1'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix	


# input: N*D matrix
# strategy: Intersection:
#				 score++ for each (1,1) pair
# output: symmetrical N*N matrix
# example:
#		row1	0 1 1 1 0 0 
#		row2    1 0 0 1 0 0
#		score	0+0+0+1+0+0 = 1 	-> result[row1][row2] = 1 = result[row2][row1]
def strategy_2(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 2'
	for i in range(n):
		for j in range(i + 1,n):
			# +1 for each (1,1) pair
			score = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			res_matrix[i,j] = score	
			res_matrix[j,i] = score
	# scale the matrix in range 0-1
	max_val =  np.amax(res_matrix)
	scaler = 1 / max_val
	res_matrix *= scaler * 0.999999
	# each point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)	
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 2'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 2'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix


# input: N*D matrix
# strategy: NAND:
# 				score++ for each (0,0) pair
# output: symmetrical N*N matrix
# example:
#		row1	0 1 1 1 0 0 
#		row2    1 0 0 1 0 0
#		score	0+0+0+1+0+0 = 2 	-> result[row1][row2] = 2 = result[row2][row1]
def strategy_3(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 3'
	for i in range(n):
		for j in range(i + 1,n):
			# +1 for each (0,0) pair
			score = np.sum(np.logical_not(np.logical_or(matrix[i,:], matrix[j,:])))
			res_matrix[i,j] = score	
			res_matrix[j,i] = score
	# scale the matrix in range 0-1
	max_val =  np.amax(res_matrix)
	scaler = 1 / max_val
	res_matrix *= scaler * 0.999999
	# teach point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)	
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 3'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 3'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix


# input: N*D matrix
# strategy: similarity for each (1,1) pair is =  n * (similarity for each (0,0) pair)
#			where n = % of avg. of not-null entries in 2 coresponding rows 
#			(showed bad results because (1,1) gets too little score)
# output: symmetrical N*N matrix
# example:
#		row1	0 1 1 1 0 0 
#		row2    1 0 0 1 0 0
#		score	0+0+0+1+0+0 = x 	-> result[row1][row2] = x = result[row2][row1]
def strategy_4(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 4'
	for i in range(n):
		for j in range(i + 1,n):
			# +1 for each (0,0) pair
			score = np.sum(np.logical_not(np.logical_or(matrix[i,:], matrix[j,:])))
			# +n for each (1,1) pair
			score2 = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			# n = % of avg. of not-null entries in 2 coresponding rows
			score_weight = (np.count_nonzero(matrix[i,:]) + np.count_nonzero(matrix[j,:])) / (2*n)
			score2 *= score_weight
			score += score2

			res_matrix[i,j] = score	
			res_matrix[j,i] = score
	# scale the matrix in range 0-1
	max_val =  np.amax(res_matrix)
	scaler = 1 / max_val
	res_matrix *= scaler * 0.999999
	# teach point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)	
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 4'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 4'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix


# input: N*D matrix
# strategy: Jaccard measure:
#				size of intersection (AND)/ size of union (OR)
# output: symmetrical N*N matrix
# example:
#		row1	0 1 1 1 0 0 
#		row2    1 0 0 1 0 0
#		score	1/4			-> result[row1][row2] = 0.25 = result[row2][row1]
# note: this strategy emphesizes the (1,1) similarity if more such pairs exist
def strategy_5(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 5'
	for i in range(n):
		for j in range(i + 1,n):
			# size of intersection
			intersection = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			if intersection == 0:
				score = 0
			else:
				# size of union
				union = np.sum(np.logical_or(matrix[i,:],matrix[j,:]))
				score = intersection / union

			res_matrix[i,j] = score	
			res_matrix[j,i] = score

	res_matrix *= 0.999999
	# each point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)		
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 5'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 5'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix	

# input: N*D matrix
# strategy: Cosine:
#				score = cos(theta) = sum(a_i, b_i) / sqrt(sum(a))sqrt(sum(b))
#				with a,b rows in input matrix
# output: symmetrical N*N matrix
# example:
#		row1	0 1 1 1 0 0 
#		row2    1 0 0 1 0 0
#		score	1 / sqrt(3)sqrt(2) = 0.408		-> result[row1][row2] = 0.408 = result[row2][row1]
# note: this strategy emphesizes the (1,1) similarity if more such pairs exist
def strategy_6(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 6'
	for i in range(n):
		for j in range(i + 1,n):
			score = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			if score <> 0:
				score = score / ( sqrt(np.count_nonzero(matrix[i,:])) * sqrt(np.count_nonzero(matrix[j,:])) )
			res_matrix[i,j] = score	
			res_matrix[j,i] = score
	res_matrix[res_matrix > 1] = 1
	res_matrix[res_matrix < 0] = 0

	res_matrix *= 0.999999
	# each point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 6'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 6'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix		


# input: N*D matrix
# strategy: Inverse of strategy 4:
#				similarity for each (0,0) pair is =  n * (similarity for each (1,1) pair)
#				where n = % of avg. of not-null entries in 2 coresponding rows 
# output: symmetrical N*N matrix
def strategy_7(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 7'
	for i in range(n):
		for j in range(i + 1,n):
			# +n for each (0,0) pair
			score = np.sum(np.logical_not(np.logical_or(matrix[i,:], matrix[j,:])))
			# +1 for each (1,1) pair
			score2 = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			# n = % of avg. of not-null entries in 2 coresponding rows
			score_weight = (np.count_nonzero(matrix[i,:]) + np.count_nonzero(matrix[j,:])) / (2*n)
			score *= score_weight
			score += score2

			res_matrix[i,j] = score	
			res_matrix[j,i] = score
	# scale the matrix in range 0-1
	max_val =  np.amax(res_matrix)
	scaler = 1 / max_val
	res_matrix *= scaler * 0.999999
	# teach point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)	
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 7'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 7'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix	

# input: N*D matrix
# strategy: 3-time-weighted Jaccard measure:
#				3a / (3a + b + c)
# output: symmetrical N*N matrix
# example:
#		row1	0 1 1 1 0 0 
#		row2    1 0 0 1 0 0
#		score	1/4			-> result[row1][row2] = 0.25 = result[row2][row1]
# note: this strategy emphesizes the (1,1) similarity if more such pairs exist
def strategy_8(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 8'
	for i in range(n):
		for j in range(i + 1,n):
			# size of intersection
			a = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			if a == 0:
				score = 0
			else:
				# TODO
				b = np.sum(np.logical_and(matrix[i,:], np.logical_not(matrix[j,:])))
				c = np.sum(np.logical_and(matrix[j,:], np.logical_not(matrix[i,:])))
				score = 3*a / (3*a + b + c)

			res_matrix[i,j] = score	
			res_matrix[j,i] = score

	res_matrix *= 0.999999
	# each point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)		
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 8'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 8'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix	

# input: N*D matrix
# strategy: Simpson measure:
#				a / min(a+b, a+c)
# output: symmetrical N*N matrix
def strategy_9(matrix, str_type):
	n = len(matrix)
	m = len(matrix[0])

	res_matrix = np.zeros((n,n))
	print 'This is strategy 9'
	for i in range(n):
		for j in range(i + 1,n):
			# size of intersection
			a = np.sum(np.logical_and(matrix[i,:],matrix[j,:]))
			if a == 0:
				score = 0
			else:
				# a+b and a+c
				ab = np.count_nonzero(matrix[i,:])
				ac = np.count_nonzero(matrix[j,:])
				score = a / min(ab,ac)

			res_matrix[i,j] = score	
			res_matrix[j,i] = score

	res_matrix *= 0.999999
	# each point has similarity 1 (e.g. 100%) with itself	
	res_matrix += np.identity(n)		
	# if we need distance, we just invert it
	if str_type == 'distance':
		res_matrix = 1 - res_matrix
		res_matrix[res_matrix==1] = 0 	# 0 is now for not connected points and ID
		res_matrix = csr_matrix(res_matrix)
		print 'This was distance strategy 9'
	elif str_type == 'sim':
		res_matrix = csr_matrix(res_matrix)
		print 'This was similarity strategy 9'
	else:
		print 'Error: str_type must be "distance" or "sim". Default is "sim". '	
	return res_matrix	
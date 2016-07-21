# this script should write .dec file for the given matrix with labels
'''
PRESOLVED
0
NBLOCKS
9
BLOCK 1
constraint1
constraint2
....
BLOCK 2
constraintk
...
....
..
BLOCK 9
'''
def write(path, filename, label_name_pairs):
	labels, names = zip(*label_name_pairs)
	#print 'DEBUG write_func'
	#print labels
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	file = open(path + filename + '.dec', 'w')
	file.write('PRESOLVED\n0\nNBLOCKS\n')
	file.write(str(n_clusters) + '\n')

	if -1 in labels:
		cluster_order = range(n_clusters) + [-1]
	else:
		cluster_order = range(n_clusters)

	for cluster in cluster_order:
		if cluster <> -1:
			file.write('BLOCK %i' %(cluster+1))
			file.write('\n')
		else:
			file.write('MASTERCONSS\n')		
		# iterate all labels and take the names with label block
		for label, name in label_name_pairs:
			if label == cluster:
				file.write(name)
				file.write('\n')
		

	
	





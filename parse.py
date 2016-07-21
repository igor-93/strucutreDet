import re
import numpy as np

# Function for reading MIP files
# it automatically detects if it is MPS or LP format and parses it correspondigally
# it returns const. matrix AND const. (i.e. row) names
def read(file_name):
	if file_name.endswith('.lp'):
		res = read_lp(file_name)
	elif file_name.endswith('.mps'):
		res = read_mps(file_name)
	else:
		print 'File format %s not supported.' %(file_name)
		res = np.zeros((1,1))
	return res

# parse LP files (its bit more complicated than it should be, but it works good)
def read_lp(file_name):
	read_line = False
	file = open(file_name, 'r')
	# 1.1 count the subto attributes and get their names
	contents = file.read()
	subto_nr = contents.count('\ subto')
	#diff_var_nr = contents.count('\ var')
	subto_list = []		# holds a list of all constraints
	row_names = []
	file.seek(0)
	constraints_predef = False
	# case where we have all constrains defined in the header of the file	
	if subto_nr > 0:
		constraints_predef = True
		for line in file:
			if line.startswith('\ subto'):
				subto_match = re.match(r'\\ subto +\w+', line)
				subto = subto_match.group(0)
				subto = subto[len('\ subto '):]
				#row_names += [subto]
				subto = subto + '_'
				subto_list.append(subto)
			if len(subto_list) == subto_nr:
				break	
		file.seek(0)	
	# case where we have no defined list of constraints in the header	
	else:	
		#print 'Constraints are not defined in the header.'
		pass
	# this part is needed in order to initialize the matrix before filling it
	n = 0	# number of constraints (i.e rows)
	m = 0	# number of variables (i.e. columns)
	start_var_count = False
	index = 0
	var_index_mapping = {}
	# count the variables in the file and if possible, contraints
	for line in file:
		# case where we have all constrains defined in the header of the file	
		if any(subto in line for subto in subto_list):
			n +=1 
		if line.startswith('General') or line.startswith('Binaries'):
			start_var_count = True
			continue
		if line.startswith('End'):
			start_var_count = False	
			break
		if start_var_count:		# start counting all variables (m) and map them (i.e. 'x#1#1' : 0... 'y#1' : 1200)
			#var_list = re.findall(r'\w#*[0-9]*#*[0-9]', line)
			var_list = line.split()
			for var in var_list:
				var_index_mapping[var] = index
				index += 1	
	m = index 
	#print 'DEBUG: print_lp m= ', m
	# 1.2 initialize the matrix with -1. size of the matrix is NxM
	if constraints_predef:
		matrix = [[0 for x in range(m)] for x in range(n)]
		#print 'Starting matrix size: %i x %i' % (n,m) 
	else:
		matrix = []
		#print 'Starting with an empty matrix'

	# 1.3 reset the iterator of the file
	file.seek(0)

	# 2. read all the lines and put the values in the matrix
	begin = False
	read_line = False
	mat_row = -1;
	for line in file:
		if 'Bounds' in line:
			begin = False
			read_line = False
			break	
		if 'Subject to' in line or 'Subject To' in line:
			begin = True
			continue
		# case where we have all constrains defined in the header of the file	
		if begin and any(subto in line for subto in subto_list):
			read_line = True
			mat_row +=1
			#row_name_match = re.match(r' \w*:', line)
			#row_name = row_name_match.group(0)
			row_name = line.split(':')[0]
			row_name = row_name[1:]
			row_names += [row_name]
			continue
		# case where we have no defined list of constraints in the header
		if ':' in line and begin:
			read_line = True
			mat_row += 1
			matrix.append([0 for x in range(m)])
			#row_name_match = re.match(r' \w*:', line)
			#row_name = row_name_match.group(0)
			row_name = line.split(':')[0]
			row_name = row_name[1:]
			row_names += [row_name]
		if begin and read_line:
			# now add the data from the important lines to the matrix.
			matches = re.findall(r'[+-]?[0-9]* [xyz]#*[0-9]+#?[0-9]*', line)
			for match in matches:
				# find the coefficient and save it in int_coeff variable
				coeff_match = re.match(r'[+-]?[0-9]*', match)
				coeff = coeff_match.group(0)
				int_coeff = 0
				if len(coeff) < 1:
					int_coeff = 1
				elif len(coeff) == 1:
					if coeff[0] == '-':
						int_coeff = -1
					else:
						int_coeff = 1
				else:
					if coeff[0] == '-':
						coeff = coeff[1:]
						int_coeff = -int(coeff)
					else:
						coeff = coeff[1:]
						int_coeff = int(coeff)

				str_index_match = re.search(r'[xyz]#*[0-9]+#?[0-9]*', match)
				str_index = str_index_match.group(0)
				mat_col = var_index_mapping[str_index]
				matrix[mat_row][mat_col] = int_coeff
	matrix = np.array(matrix)
	row_names = np.array(row_names)
	if row_names.shape[0] == 0:	
		print 'ERROR in read_lp with row_names'	
	if matrix.shape[0] == 0 or matrix.shape[1] == 0:	
		print 'ERROR: read_lp() reading failed!'				
	return matrix, row_names

# reads MPS files
def read_mps(file_name):
	file = open(file_name, 'r')
	read_line = False
	n = 0	# number of constraints (i.e rows)
	m = 0	# number of variables (i.e. columns)
	start_row_count = False
	start_col_count = False  
	col_index = 0
	row_index = 0
	row_index_mapping = {}
	col_index_mapping = {}
	row_names = []
	objective_row_index = 0
	objective_row_found = False
	# count the rows and columns in the file
	for line in file:
		if 'MARKER' in line:
			continue
		if len(line.split()) == 0:
			continue
		if line.startswith('ROWS'):
			start_row_count = True
			continue
		if line.startswith('COLUMNS'):
			start_row_count = False	
			start_col_count = True
			continue
		if line.startswith('RHS'):
			start_row_count = False	
			start_col_count = False
			break	
		if start_row_count:		# start indexing rows (i.e. 'CAP00101' : 0... )
			# skip the objective function
			if line.split()[0] == 'N':
				objective_row_found = True
				objective_row_index = row_index
				#print 'OBJROW found! objective_row_index = ', objective_row_index
			row_name = line.split()[1]
			row_names += [row_name] 
			row_index_mapping[row_name] = row_index
			row_index += 1
		if start_col_count:		# start indexing columns (i.e. 'Y0040103' : 0... )
			col_name = line.split()[0]
			# check if col_name already in col_index_mapping keys
			if col_name not in col_index_mapping.keys() and col_name <> '':
				col_index_mapping[col_name] = col_index
				col_index += 1		
	n = row_index	
	m = col_index 

	# we need this because of the memory overflow (this limits to 1.8 GB)
	limit = 90000000
	if n * m > limit:
		print 'Matrix is too big to be read. Size of matrix is: (%i x %i)' %(n,m)
		print 'In total %i out of %i allowed entries.' %(n*m, limit)
		return np.zeros((1,1)), np.zeros((1,1))

	# 1.2 initialize the matrix with -1. size of the matrix is NxM
	matrix = [[0 for x in range(m)] for x in range(n)]
	
	# 1.3 reset the iterator of the file
	file.seek(0)

	# 2. read all the lines and put the values in the matrix
	begin = False

	for line in file:
		if 'MARKER' in line:
			continue
		if 'RHS' in line:
			begin = False
			break	
		if line.startswith('COLUMNS'):
			begin = True
			continue
		if len(line.split()) == 0 and begin:
			continue
		if begin:
			# line_list is of the form:
			# ['Y0040103', 'OBJECTIV', '-30.042', 'CAP00502', '.01343']
			#   col 		row1 		value1 		row2 		value2
			line_list = line.split()
			mat_row1 = row_index_mapping[line_list[1]]
			mat_col = col_index_mapping[line_list[0]]
			coeff1 = float(line_list[2])
			matrix[mat_row1][mat_col] = coeff1
			# if more then 3 in list...
			if len(line_list) > 3:
				mat_row2 = row_index_mapping[line_list[3]]
				coeff2 = float(line_list[4])
				matrix[mat_row2][mat_col] = coeff2
				
	if objective_row_found:
		n = n-1
		del matrix[objective_row_index]
		del row_names[objective_row_index]
	matrix = np.array(matrix)
	return matrix, row_names
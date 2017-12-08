import numpy as np


def pfile(file):
	dic = {}
	
	with open(file) as f:
		lines = f.readlines()
	
	for line in lines:
		line = line.strip().split()
		if len(line) > 3:
			n = int(line[0])
			if '(' in line[2]: 
				idx = line[2].index('(')
				key = line[2][0:idx]
				units = line[2][idx+1:-1]
			else:
				idx = 0
				key = line[2]
				units = 'None'
			dic[key] = {}
			dic[key]['units'] = units
			dic[key]['psi'] = np.zeros(n)
			dic[key]['y'] = np.zeros(n)
			dic[key]['dydpsi'] = np.zeros(n)
			i = 0
			continue
		
		dic[key]['psi'][i] = np.float64(line[0])
		dic[key]['y'][i] = np.float64(line[1])
		dic[key]['dydpsi'][i] = np.float64(line[2])
		i += 1
	
	
	return dic








import numpy as np

class K_File:
	def __init__(self):
		self.nml = {}
		self.nml['LPNAME'] =['PSF1A',  'PSF2A', 'PSF3A', 'PSF4A', 'PSF5A',
							 'PSF6NA', 'PSF7NA','PSF8A', 'PSF9A', 'PSF1B',
							 'PSF2B',  'PSF3B', 'PSF4B', 'PSF5B', 'PSF6NB',
							 'PSF7NB', 'PSF8B', 'PSF9B', 'PSI11M','PSI12A',
							 'PSI23A', 'PSI34A','PSI45A','PSI58A','PSI9A',
							 'PSF7FA', 'PSI7A', 'PSF6FA','PSI6A', 'PSI12B',
							 'PSI23B', 'PSI34B','PSI45B','PSI58B','PSI9B',
							 'PSF7FB', 'PSI7B', 'PSF6FB','PSI6B', 'PSI89FB',
							 'PSI89NB','PSI1L', 'PSI2L', 'PSI3L']


		self.nml['MPNAM2'] =['MPI11M067','MPI1A067', 'MPI2A067', 'MPI3A067',  'MPI4A067  ',
							 'MPI5A067', 'MPI8A067', 'MPI9A067', 'MPI79A067', 'MPI7FA067 ',
							 'MPI7NA067','MPI67A067','MPI6FA067','MPI6NA067', 'MPI66M067 ',
							 'MPI1B067', 'MPI2B067', 'MPI3B067', 'MPI4B067',  'MPI5B067  ',
							 'MPI8B067', 'MPI89B067','MPI9B067', 'MPI79B067', 'MPI7FB067 ',
							 'MPI7NB067','MPI67B067','MPI6FB067','MPI6NB067',
							 'MPI8A322', 'MPI89A322','MPI9A322', 'MPI79FA322','MPI79NA322',
							 'MPI7FA322','MPI7NA322','MPI67A322','MPI6FA322', 'MPI6NA322 ',
							 'MPI66M322','MPI6NB322','MPI6FB322','MPI67B322', 'MPI7NB322 ',
							 'MPI7FB322','MPI79B322','MPI9B322', 'MPI89B322', 'MPI8B322  ',
							 'MPI5B322', 'MPI4B322', 'MPI3B322', 'MPI2B322',  'MPI1B322  ',
							 'MPI11M322','MPI1A322', 'MPI2A322', 'MPI3A322',  'MPI4A322  ',
							 'MPI5A322', 'MPI1U157', 'MPI2U157', 'MPI3U157',  'MPI4U157',
							 'DSL1U180', 'DSL2U180', 'DSL3U180', 'DSL4U157',
							 'MPI5U157', 'MPI6U157', 'MPI7U157', 'DSL5U157',  'DSL6U157',
							 'MPI1L180', 'MPI2L180', 'MPI3L180']
		
		# 'DSL1U180', 'DSL2U180', 'DSL3U180', 'DSL4U157',
		# 'DSL5U157',  'DSL6U157'					 
		self.DSL_areas = [0.148739625718, 0.146993802189, 0.189430809952, 0.0385235076433, 
						  0.203298492472, 0.218204272396]
	
	def read_Fnml(self, filename):
		"""Reads the namelist file and returns its contents in a dictionary"""
	
		lines = open(filename,'r').readlines()
	
		first_entry_found = False
		for line in lines:
			line = line.strip()
		
			# remove Comments
			if '!' in line:
				idx = line.index('!')
				line = line[0:idx]
				if len(line) < 1: continue
		
			# check for beginning and end of namelist
			if line[0] == '&': continue
			if line[0] == '/': first_entry_found = False
		
			# remove possible comma at the end
			if line[-1] == ',': line = line[0,-1]
		
			# find lines with keys
			if '=' in line:
				if not first_entry_found: first_entry_found = True
			
				# store as many (key, value) pairs as in a single line
				num = line.count('=')
				keys = list(np.arange(num))
				values = list(np.arange(num))
				_, keys[0], line = self._split_line(line)	
				for i in xrange(1,num):
					values[i-1], keys[i], line = self._split_line(line)
				values[-1] = line.split()
			
				# convert values from string
				for i in xrange(num):
					if len(values[i]) == 1:
						values[i] = self._convert_value(values[i][0])
					else:
						values[i] = np.array([self._convert_value(item) for item in values[i]])
				
					self.nml[keys[i]] = values[i]
			else:	# just add values to last known key
				if not first_entry_found: continue
				values = line.strip().split()
				for item in values: self.nml[keys[-1]] = np.append(self.nml[keys[-1]],float(item))
		
		
	def make_v3fit_input(self, filename, type = 'all'):
		"""
		read the output from v3mags.py and make a temporary file with the formated v3fit
		input to use copy/paste. This routine will become part of a v3fit input generator 
		script in the future.
		"""
		import h5py
		with h5py.File(filename, 'r') as f:
			dset = f['name']; name = dset[...]
			dset = f['signal']; signal = dset[...]
			dset = f['weight']; weight = dset[...]
			dset = f['sigma']; sigma = dset[...]

		# replace with EFIT k-file data
		name = list(name)
		N = len(name)
		idxs = np.ones(N, dtype = bool)
		for i,probe in enumerate(self.nml['MPNAM2']):
			if probe in name:
				idx = name.index(probe)
				if 'DSL' in probe:
					n = int(probe[3]) - 1
					factor = self.DSL_areas[n]
				else: factor = 1
				signal[idx] = self.nml['EXPMP2'][i] * factor
				weight[idx] = self.nml['FWTMP2'][i]
				idxs[idx] = False
		for i,probe in enumerate(self.nml['LPNAME']):
			if probe in name:
				idx = name.index(probe)
				signal[idx] = (self.nml['COILS'][i] + self.nml['SIREF']) * 2*np.pi
				weight[idx] = self.nml['FWTSI'][i]
				idxs[idx] = False
			
		if type == 'EFIT':	# set weight of all other probes to zero
			weight[idxs] *= 0.0
		elif type == '2D':	# set weight of all 3D probes to zero
			weight[295::] *= 0.0
		elif type == '3D':	# set weight of all 2D probes to zero
			weight[0:295] *= 0.0
		
		with open('v3fit_temp.in','w') as f:
			for i in np.arange(N):
				f.write('sdo_data_a('+format(i+1,'3d')+') = ' + format(signal[i],' 13.7e') + '  ')
				f.write('sdo_sigma_a('+format(i+1,'3d')+') = ' + format(sigma[i],' 13.7e') + '  ')
				f.write('sdo_weight_a('+format(i+1,'3d')+') = ' + str(int(weight[i])) + '  ')
				f.write('! ' + name[i] + '\n')


	def _split_line(self, line):
		"""Splits the line at the first = and finds the variable name"""
		A, B = line.split('=', 1)	# do one split only at the first occurance of '='
		A = A.strip()
		B = B.strip()
		A = A.split()
		key = A[-1]
		if len(A) > 1: A = A[0:-1]
		else: A = []
		return A, key, B
	
	
	def _convert_value(self, invalue):
		"""Formats the values"""
		if 'T' in invalue: value = True
		elif 'F' in invalue: value = False
		else:
			try:
				value = float(invalue)
				if not '.' in invalue: value = int(value)
			except:
				value = invalue
				if (invalue[0]== "'") & (value[0]== "'"): value = invalue[1:-1]

		return value
	
	
	
	 
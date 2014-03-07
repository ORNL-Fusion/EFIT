import numpy as np

# --- read_g_file -------------------------------------------
# reads g-file and stores data in output dictionary
# specify shot, time (both as int) and ...
#	gpath (string)		->	path where to find g-file, default = current working dir

def read_g_file(shot, time, gpath = '.'):
	if not (gpath[-1] == '/'): gpath += '/'
	with open(gpath + 'g' + format(shot,'06d') + '.' + format(time,'05d'), 'r') as f:
		head = f.readline().split()
		NR = int(head[-2])
		NZ = int(head[-1])
		
		data = []
		for i in range(0,4):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			data.append([float(x) for x in line])
		data = np.array(data).flatten(0)
			
		Xdim = data[0]		# Distance from inner edge to outer edge covered by EFIT, in [m]
		Zdim = data[1]		# Distance from bottom to top (Z_axis) covered by EFIT, equally spaced around midplane, in [m]
		R0 = data[2]		# Major Radius of torus in [m]
		R1 = data[3]		# Position of inner edge on radial scale in [m]
		Zmid = data[4]		# Position of midplane on z-scale in [m]
		RmAxis = data[5]	# R-position of magnetic Axis in [m]
		ZmAxis = data[6]	# Z-position of magnetic Axis in [m]
		psiAxis = data[7]	# poloidal Flux at magnetic Axis
		psiSep = data[8]	# poloidal Flux at Separatrix
		Bt0 = data[9]		# toroidal magnetic field at Major Radius in [T]
		Ip = data[10]		# Plasma current in [A]
		#  9 more Unused parameters
	
		Fpol = []
		for i in range(0,int(np.ceil(NR/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			Fpol.append([float(x) for x in line])
		Fpol = [num for elem in Fpol for num in elem]
		Fpol = np.array(Fpol)
	
		Pres = []
		for i in range(0,int(np.ceil(NR/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			Pres.append([float(x) for x in line])
		Pres = [num for elem in Pres for num in elem]
		Pres = np.array(Pres)
		
		FFprime = []
		for i in range(0,int(np.ceil(NR/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			FFprime.append([float(x) for x in line])
		FFprime = [num for elem in FFprime for num in elem]
		FFprime = np.array(FFprime)
		
		Pprime = []
		for i in range(0,int(np.ceil(NR/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			Pprime.append([float(x) for x in line])
		Pprime = [num for elem in Pprime for num in elem]
		Pprime = np.array(Pprime)
		
		psiRZ = []
		for i in range(0,int(np.ceil(NR*NZ/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			psiRZ.append([float(x) for x in line])
		psiRZ = [num for elem in psiRZ for num in elem]
		psiRZ = np.array(psiRZ).reshape(NR,NZ)
		
		qpsi = []
		for i in range(0,int(np.ceil(NR/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			qpsi.append([float(x) for x in line])
		qpsi = [num for elem in qpsi for num in elem]
		qpsi = np.array(qpsi)
				
		head = f.readline().split()
		Nlcfs = int(head[0])
		Nwall = int(head[1])
	
		lcfs = []
		for i in range(0,int(np.ceil(2*Nlcfs/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			lcfs.append([float(x) for x in line])
		lcfs = [num for elem in lcfs for num in elem]
		lcfs = np.array(lcfs).reshape(Nlcfs,2)
	
		wall = []
		for i in range(0,int(np.ceil(2*Nwall/5.0))):
			line = f.readline()
			line = split_data(line)					# g-file does not always have blank spaces between data points 
			wall.append([float(x) for x in line])
		wall = [num for elem in wall for num in elem]
		wall = np.array(wall).reshape(Nwall,2)
		
	f.close
	
	# Construct (R,Z) grid for psiRZ
	dR = Xdim/(NR - 1)
	R = R1 + np.arange(NR)*dR
	
	dZ = Zdim/(NZ - 1)
	NZ2 = int(np.floor(0.5*NZ))
	Z = Zmid + np.arange(-NZ2,NZ2+1)*dZ
	
	# normalize psiRZ
	psiRZn = (psiRZ - psiAxis) / (psiSep - psiAxis)

	g = {'shot':shot, 'time':time, 'NR':NR, 'NZ':NZ,
		 'Xdim':Xdim, 'Zdim':Zdim, 'R0':R0, 'R1':R1, 'Zmid':Zmid,
		 'RmAxis':RmAxis, 'ZmAxis':ZmAxis, 'psiAxis':psiAxis, 'psiSep':psiSep, 
		 'Bt0':Bt0, 'Ip':Ip,
		 'Fpol':Fpol, 'Pres':Pres, 'FFprime':FFprime, 'Pprime':Pprime, 'qpsi':qpsi,
		 'psiRZ':psiRZ, 'R':R, 'Z':Z, 'dR':dR, 'dZ':dZ, 'psiRZn':psiRZn,
		 'Nlcfs':Nlcfs, 'Nwall':Nwall, 'lcfs':lcfs, 'wall':wall}
	return g


# --- read_g_file_mds -------------------------------------------
# reads g-file from MDS+ and stores data in output dictionary
# Note: MDS+ data is only single precision!
# specify shot, time (both as int) and ...
#	tree (string)		->	EFIT tree name, default = 'EFIT01'
# further keywords:
#	exact (bool)		->	True: time must match time in EFIT tree, otherwise abort
#							False: EFIT time closest to time is used (default)
#	Server (string)		->	MDS+ server name or IP, default = 'atlas.gat.com' (for DIII-D)
# optionally the g-file is written to disk
#	write2file (bool)	->	True: save g-file (default), False: do not write file
#	gpath (string)		->	path where to save g-file, default = current working dir

def read_g_file_mds(shot, time, tree = 'EFIT01', exact = False, Server = 'atlas.gat.com',
				    write2file = True, gpath = '.'):
	import MDSplus

	# in case those are passed in as strings
	shot = int(shot)
	time = int(time)

	# Connect to server, open tree and go to g-file
	MDS = MDSplus.Connection(Server)
	MDS.openTree(tree,shot)
	base = 'RESULTS:GEQDSK:'

	# get time slice
	signal = 'GTIME'
	k = np.argmin(np.abs(MDS.get(base + signal).data() - time))
	time0 = int(MDS.get(base + signal).data()[k])

	if (time != time0):
		if exact:
			raise RuntimeError(tree + ' does not exactly contain time ' + str(time) + '  ->  Abort')
		else:
			print 'Warning: ' + tree + ' does not exactly contain time ' + str(time) + ' the closest time is ' + str(time0)
			print 'Fetching time slice ' + str(time0)
			time = time0

	# store data in dictionary
	g = {'shot':shot, 'time':time}
	
	# get header line
	header = MDS.get(base + 'ECASE').data()[k]

	# get all signals, use same names as in read_g_file
	translate={'MW':'NR', 'MH':'NZ', 'XDIM':'Xdim', 'ZDIM':'Zdim', 'RZERO':'R0',
			   'RMAXIS':'RmAxis', 'ZMAXIS':'ZmAxis', 'SSIMAG':'psiAxis', 'SSIBRY':'psiSep',
			   'BCENTR':'Bt0', 'CPASMA':'Ip', 'FPOL':'Fpol', 'PRES':'Pres', 
			   'FFPRIM':'FFprime', 'PPRIME':'Pprime', 'PSIRZ':'psiRZ', 'QPSI':'qpsi',
			   'NBBBS':'Nlcfs', 'LIMITR':'Nwall'}
	for signal in translate:
		g[translate[signal]] = MDS.get(base + signal).data()[k]

	g['R1'] = MDS.get(base + 'RGRID').data()[0]
	g['Zmid'] = 0.0
	
	RLIM = MDS.get(base + 'LIM').data()[:,0]
	ZLIM = MDS.get(base + 'LIM').data()[:,1]
	g['wall'] = np.vstack((RLIM, ZLIM)).T
	
	RBBBS = MDS.get(base + 'RBBBS').data()[k][:g['Nlcfs']]
	ZBBBS = MDS.get(base + 'ZBBBS').data()[k][:g['Nlcfs']]
	g['lcfs'] = np.vstack((RBBBS,ZBBBS)).T
	
	KVTOR = 0
	RVTOR = 1.7
	NMASS = 0
	RHOVN = MDS.get(base + 'RHOVN').data()[k]

	# convert floats to integers
	for item in ['NR', 'NZ', 'Nlcfs', 'Nwall']: g[item] = int(g[item])
	
	# convert single (float32) to double (float64) and round
	for item in ['Xdim', 'Zdim', 'R0', 'R1', 'RmAxis', 'ZmAxis', 'psiAxis', 'psiSep', 'Bt0', 'Ip']: 
		g[item] = np.round(np.float64(g[item]),7)
	
	# convert single arrays (float32) to double arrays (float64)
	for item in ['Fpol', 'Pres', 'FFprime', 'Pprime', 'psiRZ', 'qpsi', 'lcfs', 'wall']:
		g[item] = np.array(g[item], dtype = np.float64)

	# write g-file to disk
	if write2file:
		if not (gpath[-1] == '/'): gpath += '/'	
		with open(gpath + 'g' + format(shot,'06d') + '.' + format(time,'05d'), 'w') as f:
			if ('EFITD' in header[0]) and (len(header) == 6): 
				for item in header: f.write(item)
			else: 
				f.write('  EFITD    xx/xx/xxxx    #' + str(shot) + '  ' + str(time) + 'ms        ')
				
			f.write('   3 ' + str(g['NR']) + ' ' + str(g['NZ']) + '\n')
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(g['Xdim'], g['Zdim'], g['R0'], g['R1'], g['Zmid']))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(g['RmAxis'], g['ZmAxis'], g['psiAxis'], g['psiSep'], g['Bt0']))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(g['Ip'], 0, 0, 0, 0))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(0,0,0,0,0))
			write_array(g['Fpol'], f)
			write_array(g['Pres'], f)
			write_array(g['FFprime'], f)
			write_array(g['Pprime'], f)
			write_array(g['psiRZ'].flatten(), f)
			write_array(g['qpsi'], f)
			f.write(str(g['Nlcfs']) + ' ' + str(g['Nwall']) + '\n')
			write_array(g['lcfs'].flatten(), f)
			write_array(g['wall'].flatten(), f)		
			f.write(str(KVTOR) + ' ' + format(RVTOR,' .9E') + ' ' + str(NMASS) + '\n')
			write_array(RHOVN, f)

	# Construct (R,Z) grid for psiRZ
	g['dR'] = g['Xdim']/(g['NR'] - 1)
	g['R'] = g['R1'] + np.arange(g['NR'])*g['dR']
	
	g['dZ'] = g['Zdim']/(g['NZ'] - 1)
	NZ2 = int(np.floor(0.5*g['NZ']))
	g['Z'] = g['Zmid'] + np.arange(-NZ2,NZ2+1)*g['dZ']
	
	# normalize psiRZ
	g['psiRZn'] = (g['psiRZ'] - g['psiAxis']) / (g['psiSep'] - g['psiAxis'])
	
	return g

	
# --- write_array -----------------------
# write numpy array in format used in g-file:
# 5 columns, 9 digit float with exponents and no spaces in front of negative numbers
def write_array(x, f):
	N = len(x)
	rows = N/5	# integer division
	rest = N - 5*rows
	
	for i in xrange(rows):
		for j in xrange(5): f.write('% .9E'%(x[i*5 + j]))
		f.write('\n')
	
	if(rest > 0):
		for j in xrange(rest): f.write('% .9E'%(x[rows*5 + j]))
		f.write('\n')


# --- split_data --------------------------------------------
# In the g-file there is typically no space between data points,
# if the value is negative (space is occupied by the '-' sign).
# Each row in the file is read in as a single string.
# This routine searches for the 'e' in each floating point value
# and splits the string 'line' accordingly
def split_data(line):
    length = len(line)
    index = []
    
    # find all the 'E' in the string, each number is in floating representation and therefore has one
    for j in range(0,length):
        if ((line[j] == 'E') | (line[j] == 'e')) & (j < length-4):
            index.append(j+4)
            
    # Split up line into fragments using the positions of 'E'
    line3 = []
    line3.append(line[0:index[0]].strip())	# first data# .strip() removes any blanks in front 
    for k in range(0,len(index)-1):
        line3.append(line[index[k]:index[k+1]].strip())	# omitt line[index[-1]:length] = '\n', so stop before 
    return line3
    
    
# --- compare_them --------------------------
# compares two g files and print maximum error fo each entry
def compare_them(g1, g2):
	# ints
	integers = ['shot', 'time', 'NR', 'NZ', 'Nlcfs', 'Nwall']
	print 'Integers'
	for item in integers:
		print 'Error in', item, ':', (g1[item] - g2[item])
		
	# floats
	floats = ['Xdim', 'Zdim', 'R0', 'R1', 'Zmid', 'RmAxis', 'ZmAxis', 'psiAxis', 'psiSep',
			  'Bt0', 'Ip', 'dR', 'dZ']
	print 'Doubles'
	for item in floats:
		if (g2[item] == 0.0):
			print 'rel. Error in', item, ':', (g1[item] - g2[item])
		else:
			print 'rel. Error in', item, ':', (g1[item] - g2[item])/g2[item]
			
	# arrays
	#arrays = [item for item in g2 if ((item not in integers) and (item not in floats))]
	arrays = ['R', 'Z', 'psiRZ', 'psiRZn', 'Fpol', 'Pres', 'Pprime', 'FFprime', 'qpsi', 'lcfs', 'wall']
	print 'Arrays'
	for item in arrays:
		x = g2[item].copy().flatten()
		x[np.where(x == 0.0)] = 1e-50
		x = x.reshape(g2[item].shape)
		print 'Max. rel. Error in', item, ':', np.max((g1[item] - g2[item])/x)
	



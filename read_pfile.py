import numpy as np


def pfile(file):
	"""
	Parse the contents of the p-file to a dictionary
	"""
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


def getProfile(file, key, save = False, show = True, N = 301, type = 'tanh0'):
	"""
	Read a profile from p-file, extend it to psi = 1.2, save and plot 
	Input:
	  file = pathname for p-file, or .sav IDL file with raw Ti data points from EFITVIEWER
	  key = which profile to work on, e.g. 'ti', or 'ne'
	  save = bool, True: save to file, default is False
	  show = bool, make figures, default is True, save figures as eps for save = True as well
	  N = points in profile
	  type = options for profile fitting of raw data: 'tanh','tanh0','tanhflat'
	Return:
	  psi, pro
	"""
	from Misc.optimize_profiles import fit_profile
	if '.sav' in file:
		from scipy.io import readsav
		rawdata = readsav(file)
		x,y = rawdata['x'],rawdata['y']
		units = 'a.u.'
	else:
		p = pfile(file)
		x,y,units = p[key]['psi'],p[key]['y'],p[key]['units']
		rawdata = {'px':x, 'py':y}
	rawdata['key'],rawdata['units'] = key,units
	if type in ['tanh', 'tanh0', 'tanhflat']: psi,pro,_ = fit_profile(x,y,type = type,xlim = [0,1.2])
	else: psi,pro = x,y
	if save:
		idx = file[::-1].find('/')
		if(idx == -1): tag = file[1::] 
		else: tag = file[-idx+1::]
		tag = tag.replace('.','_')
	else: tag = None
	if show: plotProfile(psi,pro, tag = tag, rawdata = rawdata)
	if save:
		with open(key + '_' + tag + '.dat','w') as f:
			f.write('# ' + key + ' profile in ' + units + ', based of p-file ' + file + ' \n')
			f.write('# psi          ' + key + ' [' + units + '] \n')
			for i in xrange(len(psi)):
				f.write(format(psi[i],' 13.7e') + ' \t' + format(pro[i],' 13.7e') + '\n')
	return psi, pro


def plotProfile(psi, pro, save  = False, tag = None, rawdata = None):
	"""
	Make figure of profile
	save = True: save figure as eps
	add tag to figure file names; save is True, for tag not None
	"""
	import matplotlib.pyplot as plt
	if tag is None: tag = ''
	else:
		tag = '_' + tag
		save  = True
	
	plt.figure()
	if rawdata is not None:
		if rawdata.has_key('x'): plt.plot(rawdata['x'],rawdata['y'],'ro')
		if rawdata.has_key('px'): plt.plot(rawdata['px'],rawdata['py'],'r--')
	plt.plot(psi,pro,'k-',lw = 2)
	plt.xlabel('$\\psi$')
	plt.ylabel(rawdata['key'] + ' [' + rawdata['units'] + ']')
	if save: plt.gcf().savefig(rawdata['key'] + 'Profile' + tag + '.eps', dpi = (300), bbox_inches = 'tight')





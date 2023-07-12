#!/usr/bin/env python3
import numpy as np
import scipy.interpolate as scinter


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


def getProfile(file, key, save = False, show = True, N = 301, type = 'tanh0', xmin = 0):
	"""
	Read a profile from p-file, extend it to psi = 1.2, save and plot 
	Input:
	  file = pathname for p-file, or .sav IDL file with raw Ti data points from EFITVIEWER
	  key = which profile to work on, e.g. 'ti', or 'ne'
	  save = bool, True: save to file, default is False
	  show = bool, make figures, default is True, save figures as eps for save = True as well
	  N = points in profile
	  type = options for profile fitting of raw data: 'tanh','tanh0','tanhflat','spline'
	  xmin = use type fit for x > xmin. For x < xmin, splines are used. The full profile is then combined
	Return:
	  psi, pro
	"""
	if '.sav' in file:
		from scipy.io import readsav
		rawdata = readsav(file)
		x,y = rawdata['x'],rawdata['y']
		units = 'a.u.'
	else:
		p = pfile(file)
		if key not in p:
			print (key, 'not found. Available keys are:')
			for key in np.sort(list(p.keys())):
				print (key)
			return 0,0
		x,y,units = p[key]['psi'],p[key]['y'],p[key]['units']
		rawdata = {'px':x, 'py':y}
	rawdata['key'],rawdata['units'] = key,units
	if type in ['tanh', 'tanh0', 'tanhflat']:
		from Misc.optimize_profiles import fit_profile
		if xmin > 0:
			idx = x > xmin
			x0 = x[idx]
			y0 = y[idx]
			psi0,pro0,_ = fit_profile(x0,y0,type = type,xlim = [xmin,1.2])
			x1 = np.append(x[-idx][0:-10],psi0[10::])
			y1 = np.append(y[-idx][0:-10],pro0[10::])
			f = scinter.UnivariateSpline(x1, y1, s = 0)
			psi = np.linspace(0,1.2,N)
			pro = f(psi)
		else:
			psi,pro,_ = fit_profile(x,y,type = type,xlim = [0,1.2],points = N)
	elif type in ['spline']:
		f = scinter.UnivariateSpline(x,y,s = 0)
		psi = np.linspace(x.min(),x.max(),N)
		pro = f(psi)
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
			for i in range(len(psi)):
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
	plt.plot(psi,pro,'k-',lw = 2)
	if rawdata is not None:
		if 'x' in rawdata: plt.plot(rawdata['x'],rawdata['y'],'ro')
		if 'px' in rawdata: plt.plot(rawdata['px'],rawdata['py'],'r--')
	plt.xlabel('$\\psi$')
	plt.ylabel(rawdata['key'] + ' [' + rawdata['units'] + ']')
	if save: plt.gcf().savefig(rawdata['key'] + 'Profile' + tag + '.eps', dpi = (300), bbox_inches = 'tight')



def write(p, filename):
	"""
	"""
	fmt = '.6f'
	keys = ['ne', 'te', 'ni', 'ti', 'nb', 'pb', 'ptot', 'omeg', 'omegp', 'omgvb', 'omgpp', 'omgeb', 'er', 'ommvb', 'ommpp', 'omevb', 'omepp', 'kpol', 'omghb', 'nz1', 'vtor1', 'vpol1']
	#units= ['10^20/m^3','KeV','10^20/m^3','KeV','10^20/m^3','KPa','KPa','kRad/s','kRad/s','kRad/s','kRad/s','kRad/s','kV/m','','','','','km/s/T','','10^20/m^3','km/s','km/s']
	
	with open(filename,'w') as f:
		for key in keys:
			N = len(p[key]['psi'])
			if(p[key]['units'] == 'None'): f.write(str(N) + ' psinorm ' + key + '() ' + 'd' + key + '/dpsiN' + '\n')
			else: f.write(str(N) + ' psinorm ' + key + '(' + p[key]['units'] + ') ' + 'd' + key + '/dpsiN' + '\n')
			for i in range(N): f.write(' ' + format(p[key]['psi'][i],fmt) + '   ' + format(p[key]['y'][i],fmt) + '   ' + format(p[key]['dydpsi'][i],fmt) + '\n')
		f.write('3 N Z A of ION SPECIES\n')
		for i in range(3): f.write(' ' + format(p['Z']['psi'][i],fmt) + '   ' + format(p['Z']['y'][i],fmt) + '   ' + format(p['Z']['dydpsi'][i],fmt) + '\n')



# ----------------------------------------------------------------------------------------
# --- Launch main() ----------------------------------------------------------------------
if __name__ == '__main__':
	import argparse
	import textwrap
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser(description = 'Plot the ne,ni,Te,Ti profiles from p-file', 
				formatter_class = argparse.RawDescriptionHelpFormatter,
				epilog = textwrap.dedent('''\
                Examples: read_pfile.py p148712.04101'''))

	parser.add_argument('pfile', help = 'Profile file name or (full or rel.) pathname', type = str)
	parser.add_argument('-k', '--keys', help = 'List of additional keys to plot, separated by comma, no spaces, e.g. er,nz', type = str, default = None)
	args = parser.parse_args()
	
	keys = ['ne','ni','te','ti']
	if args.keys is not None:
		newkeys = args.keys.split(',')
		for key in newkeys: keys.append(key)
	
	for key in keys:
		_ = getProfile(args.pfile, key, type = 'spline')
		
	plt.show()



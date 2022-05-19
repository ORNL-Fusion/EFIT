#!/usr/bin/env python3
import numpy as np
import scipy.interpolate as scinter


def pfile(filename):
	with open(filename) as f: lines = f.readlines()
	readVals = False
	pfile = {}
	
	for line in lines:
		line = line.strip()
		if len(line) < 1: continue
		if ('parameters' in line) & (not readVals): 
			readVals = True
			continue
		if ' : ' in line:
			keys = line.split()
			key = keys[0]
			for i in range(1,len(keys)):
				if not ':' in keys[i]: key = key + keys[i]
				else: 
					idx = i+1
					break
			pfile[key] = []
			pfile[key + '_unit'] = ' '.join(keys[idx::])
			continue
		if readVals:
			vals = line.split()
			for val in vals: 
				val = val.replace('D','e')
				pfile[key].append(np.float64(val))
			continue
			
	for key in pfile:
		if (len(pfile[key]) < 2): 
			pfile[key] = pfile[key][0]
			if pfile[key] == int(pfile[key]): pfile[key] = int(pfile[key])
		elif '_unit' in key: continue
		else: pfile[key] = np.array(pfile[key])
	
	pfile['PsiN'] = (pfile['Psi']-pfile['Psi'][0])/(pfile['Psi'][-1]-pfile['Psi'][0])
	pfile['PsiN_unit'] = 'normalized poloidal flux'
	return pfile


def getProfile(file, key, save = False, show = True, N = 301, type = 'tanh0', xmin = 0):
	"""
	Read a profile from p-file, extend it to psi = 1.2, save and plot 
	Input:
	  file = pathname for p-file
	  key = which profile to work on, e.g. 'ti', or 'ne'
	  save = bool, True: save to file, default is False
	  show = bool, make figures, default is True, save figures as eps for save = True as well
	  N = points in profile
	  type = options for profile fitting of raw data: 'tanh','tanh0','tanhflat','spline'
	  xmin = use type fit for x > xmin. For x < xmin, splines are used. The full profile is then combined
	Return:
	  psi, pro
	"""
	p = pfile(file)
	if key not in p:
		print (key, 'not found. Available keys are:')
		for key in np.sort(list(p.keys())):
			print (key)
		return 0,0
	x,y,units = p['PsiN'],p[key],p[key + '_unit']
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



# ----------------------------------------------------------------------------------------
# --- Launch main() ----------------------------------------------------------------------
if __name__ == '__main__':
	import argparse
	import textwrap
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser(description = 'Plot the ne,ni,Te,Ti profiles from p-file', 
				formatter_class = argparse.RawDescriptionHelpFormatter,
				epilog = textwrap.dedent('''\
                Examples: read_iter_pfile.py P_ITER_xxx.TXT'''))

	parser.add_argument('pfile', help = 'Profile file name or (full or rel.) pathname', type = str)
	parser.add_argument('-k', '--keys', help = 'List of additional keys to plot, separated by comma, no spaces, e.g. er,nz', type = str, default = None)
	args = parser.parse_args()
	
	keys = ['Ne','Ni','Te','Ti']
	if args.keys is not None:
		newkeys = args.keys.split(',')
		for key in newkeys: keys.append(key)
	
	for key in keys:
		_ = getProfile(args.pfile, key, type = 'spline')
		
	plt.show()



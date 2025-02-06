import sys,os
import numpy as np
import scipy.interpolate as scinter

import Misc.optimize_profiles as op

def make_profile(x, y, key, asymptote = None, save = False, show = True, 
				xmin = 0, xmax = 1.2, dx = 0.005, xout = None,
				matchpoint = None, dxfit = None, SOLdxfit = None):
	"""
	Take profile data x,y, extend it to psi = xmax using a tanh with given asymptotic value
	Optional save and plot 
	Input:
	  x = psi
	  y = data, e.g. 'te', or 'ne'
	  key = string that identifies what kind of data is fitted, like 'ne' or 'Te'
	  asymptote = asymptotic value for psi -> inf; for asymptote = 0, the fit is the same as type = 'tanh0'
	  save = bool, True: save to file, default is False
	  show = bool, make figures, default is True, save figures as eps for save = True as well
	  xmin = min psi for profile fit; the profile fit takes data only from x > xmin; For x < xmin splines are used.
	  xmax = max psi for profile fit; this is the extrapolation limit
	  dx = final resolution of profile
	  xout = optional grid points for the final profile; if given, ignores dx; default is equidistant grid with dx
	  matchpoint = psi where x < xmin spline and x > xmin fit are matched together. This can be different than xmin, default is xmin
	  dxfit = number of grid points on the left and right of matchpoint to ignore, so that the match can be made smoother with splines, typically single digit
	  SOLdxfit = extra number of grid points to ignore on top of dxfit on the psi > matchpoint side; typically 0
	Return:
	  psi, pro
	"""
	if x.min() > 0:
		x1, y1 = x[0],y[0]
		x2, y2 = x[1],y[1]
		a = (y2-y1)/(x2-x1)
		b = y2 - a*x2
		x = np.append(0,x)
		y = np.append(b,y)
		
	if key in ['ne','Ne','ni','Ni','density']: 
		units = '10$^{20}$/m$^3$'
		if asymptote is None: asymptote = 0.02
		if matchpoint is None: matchpoint = 0.99
		if dxfit is None: dxfit = 3
	elif key in ['te','Te','ti','Ti','temperature']: 
		units = 'keV'
		if asymptote is None: asymptote = 1e-4
		if matchpoint is None: matchpoint = 0.99
		if dxfit is None: dxfit = 3
	elif key in ['omega','omgeb','rotation']: 
		units = 'kRad/s'
		if asymptote is None: asymptote = 0.0
		if matchpoint is None: matchpoint = xmin
		if dxfit is None: dxfit = 9
	else: 
		units = 'a.u.'
		if asymptote is None: asymptote = 0.0
		if matchpoint is None: matchpoint = xmin
		if dxfit is None: dxfit = 0
	
	if SOLdxfit is None: SOLdxfit = 0
	
	rawdata = {'px':x, 'py':y, 'units':units, 'key':key}
	if xmin > 0:
		# fit a profile within [xmin,xmax] using dx step size
		idx = x > xmin
		x0 = x[idx]
		y0 = y[idx]
		psi0,pro0,_ = fit_profile(x0, y0, asymptote, xlim = [xmin,xmax], dx = dx)
		
		# attach the original profile for x < xmin with the fitted profile for x > xmin
		idx1 = np.abs(x - matchpoint).argmin()
		idx2 = np.abs(psi0 - matchpoint).argmin()
		x1 = np.append(x[0:idx1-dxfit],psi0[idx2+dxfit+SOLdxfit::])
		y1 = np.append(y[0:idx1-dxfit],pro0[idx2+dxfit+SOLdxfit::])
		
		# spline the combined profile for a smooth curve everywhere
		# uses the points xout if given, or a equidistant grid with dx otherwise
		f = scinter.UnivariateSpline(x1, y1, s = 0)
		if xout is None: psi = np.arange(0,xmax+dx,dx)
		else: psi = xout
		pro = f(psi)
	else:	# fit the entire profile, This is usually not a good idea, as the core and edge won't both fit well with the same tanh; this will ignore xout
		psi,pro,_ = fit_profile(x, y, asymptote, xlim = [0,xmax], dx = dx)
		
	if show: 
		plotProfile(psi, pro, rawdata = rawdata)
		print('Minimum: ', pro.min(), '  at psi = ', psi[pro.argmin()])
	if save:
		with open('profile_' + str.lower(key),'w') as f:
			#f.write('# ' + key + ' profile in ' + units + ' \n')
			#f.write('# psi          ' + key + ' [' + units + '] \n')
			for i in range(len(psi)):
				f.write(format(psi[i],' 13.7e') + ' \t' + format(pro[i],' 13.7e') + '\n')
	return psi, pro


def fit_profile(xin, y, asymptote, xlim = None, truncate = None, dx = 0.005):
	"""
	Calls op.fit_profile(...), but for type = tanhfix only
	tanhfix is a tanh with a linear slope on the left, a fixed monotonic asymptote on the right 
	fit values in popt are: 	SYMMETRY POINT, FULL WIDTH, HEIGHT, SLOPE INNER
	"""
	if xlim is None: points = int((xin.max() - xin.min())/dx) + 1
	else: points = int((xlim[1] - xlim[0])/dx) + 1

	# fit profile
	x1,y1,popt = op.fit_profile(xin, y, type = 'tanhfix', xlim = xlim, truncate = truncate, points = points, asymptote = asymptote)
	#print(popt)
	return x1,y1,popt
	

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








































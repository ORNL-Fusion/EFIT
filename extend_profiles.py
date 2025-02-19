import sys,os
import numpy as np
import scipy.interpolate as scinter

import Misc.optimize_profiles as op

def make_profile(x, y, key, asymptote = None, save = False, show = True, 
				xmin = 0, xmax = 1.2, dx = 0.005, xout = None, preservePoints = True,
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
	  xout = optional grid points for the final profile; if given, ignores dx; default is equidistant grid with dx (see also preservePoints)
	  preservePoints = Bool, whether to preserve the original x grid points along the inter- and extrapolated profile, or to replace with a linspace grid
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

	# If you want to keep all original points in the profile, but fill in the gaps
	# divide intervals until number of grid points exceed threshold, given by suggested dx
	# this assumes non-equidistant grid, as original profiles are likely equidistant in rho, not in psi
	psi = x.copy()
	Npsi = len(psi)		# this is the current number of grid points
	
	if Npsi < 100: 			# use equidistant temporary upscaling for smooth profile fits 
		Nup = 101
		upscale = True		# this is very important in making a better tanh curve fit 
	else: upscale = False
	
	# this sets the output grid
	if xout is None: 
		if preservePoints:
			Nmax = int(1.0/dx) + 1	# this is the threshold
			while Npsi < Nmax:
				Npsi = 2*Npsi - 1
				xout = np.zeros(Npsi)
				xout[0::2] = psi
				xout[1::2] = psi[0:-1] + 0.5*np.diff(psi)
				psi = xout
			# now psi is on higher resolution, but still only [0,1], now extend to xmax
			psiSol = np.arange(1+dx,xmax+dx,dx)
			psi = np.append(psi,psiSol)
			Npsi = len(psi)
			#print(Npsi,psi)
		else: psi = np.arange(0,xmax+dx,dx)
	else: psi = xout
	
	if upscale:
		f = scinter.PchipInterpolator(x,y)		# this gets overwritten by a smooth profile fit later
		x = np.linspace(x.min(),x.max(),Nup)
		y = f(x)
	
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
		pro = f(psi)
	else:	# fit the entire profile, This is usually not a good idea, as the core and edge won't both fit well with the same tanh; this will ignore xout
		psi0,pro0,_ = fit_profile(x, y, asymptote, xlim = [0,xmax], dx = dx)
		f = scinter.UnivariateSpline(psi0,pro0, s = 0)
		pro = f(psi)
		
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


def correct_ne(psi0, p, ne, psiEx, Te, Ti = None, asymptote = 0, correctionMargin = None):
	"""		
	Correct ne, so that p - sum(n*T) >= 0 for all points in p
	This ignores any original ni profile and assumes ni = ne, as required in M3D-C1, also ignore any impurities
	Then extrapolate ne using exponential decay and interpolate ne on an extended psi grid
	psi0 = original psi grid for p and ne
	p = pressure profile on psi0 grid
	ne = electron density on psi0 grid
	psiEx = inter- and extrapolated (aka extended) psi grid for temperature only
	Te = electron temperature on extended psi grid
	Ti = optional ion temperature on extended psi grid, if None: Ti = p/ne - Te, but then force Ti >= Te.min() and continue correcting ne with this assumed Ti
	asymptote = ne in far SOL
	correctionMargin = value < 1, but close to 1, default is 0.99, to multiply sum(n*T) so that sum(n*T) < p even for interpolated values.
	"""
	e = 1.60217663e-19
	if correctionMargin is None: correctionMargin = 0.99
	
	# Spline te and ti for the original psi
	fte = scinter.UnivariateSpline(psiEx, Te, s = 0)
	tex = fte(psi0)*e
	if Ti is None: 
		Ti = (p/ne - tex)/e
		idx = np.where(Ti < Te.min())[0]
		if len(idx) > 0: Ti[idx] = Te.min()
		tix = Ti*e
	else: 
		fti = scinter.UnivariateSpline(psiEx, Ti, s = 0)	
		tix = fti(psi0)*e
	
	# with original ne get d = p - sum(n*T), with original p, and extended/splined T
	d = p - ne * (tex + tix)
	
	# find points where d < 0
	idx = np.where(d < 0)[0]
	if len(idx) == 0: 
		print('ne okay, no correction needed')
		return	# done, d >=0 everywhere already
	
	# where d < 0 replace ne with new value so that  p - sum(n*T) >= 0 -> ne patched
	ne0 = p / (tex+tix)		# this ne would make d = 0 everywhere
	nePatched = ne.copy()
	nePatched[idx] = ne0[idx] * correctionMargin	# give it a tiny margin
	
	# Use a monotonic interpolation for ni_patched -> upscale ni to extended psi grid
	# !!!!!!! This interpolator does not overshoot like UnivariateSpline, but instead maintains a monotonic curve !!!!!!!!!!!!
	# However, this is not as smooth as a regular spline. The first derivatives are guaranteed to be continuous, but the second derivatives may jump
	f = scinter.PchipInterpolator(psi0, nePatched)
	sol = psiEx > 1
	psiCore = psiEx[~sol]
	y = f(psiCore)
	
	# Point and derivative at separatrix using the extended psi grid
	x1 = psiCore[-1]	# this should be  = 1
	y1 = y[-1]
	dx = psiCore[-1] - psiCore[-2]	# NOT equidistant
	dy1 = (-y[-2] + y[-1])/dx		# 1st order only due to non-equidistant grid
	
	# Fit exponential decay f(x) = a*exp(b*x) + c; c = asymptote is given as input
	c = asymptote
	b = dy1/(y1 - c)
	a = dy1/b * np.exp(-b*x1)
	
	# extend ni
	f = lambda x: a*np.exp(b*x) + c
	neNew = np.append(y, f(psiEx[sol]))
	
	# verify & update
	fne = scinter.UnivariateSpline(psiEx, neNew, s = 0)
	p2 = fne(psi0) * (tex + tix)  	# Use original psi grid
	d2 = p - p2
	if any(d2 < 0): print('Sum of n*T exceeds thermal pressure inside the separatrix. Check extended profiles.')
	else: print('ne correction okay')
	return neNew


def checkExtension(psiEx, ne, Te, ni, Ti, pEx):
	"""
	Verify that p - sum(n*T) >= 0
	Using the extended psi grid and original grid
	"""
	e = 1.60217663e-19
	p = ne * Te*e + ni * Ti*e
	d = pEx - p
	if any(d < -1e-14):	# exclude machine rounding errors 
		idx = np.where(d < 0)[0]
		print('Sum of n*T exceeds thermal pressure inside the separatrix. Check extended profiles.')
	else: 
		print('Extension okay')
	
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(psiEx, pEx*1e-3, 'k-', lw = 2, label = 'thermal pressure')
	plt.plot(psiEx, p*1e-3, 'b-', lw = 2, label = 'sum(n*T)')
	plt.plot(psiEx, d*1e-3, 'r-', lw = 2, label = 'difference')
	if any(d < -1e-14): plt.plot(psiEx[idx], d[idx]*1e-3, 'rx')
	plt.plot([0,psiEx.max()], [0,0], 'k--', lw = 1)
	plt.xlim(0,psiEx.max())
	plt.xlabel('$\\psi$')
	plt.ylabel('p$_{th}$ [kPa]')
	plt.legend()
	return


def extendPressure(psiEx, ne, Te, ni, Ti, psi0, p0):
	"""
	Extend the pressure using sum(n*T) for psi > 1
	"""
	e = 1.60217663e-19
	p = ne * Te*e + ni * Ti*e
	idx = np.where(psiEx > 1)[0]
	psiNew = np.append(psi0, psiEx[idx])
	pNew = np.append(p0, p[idx])
	f = scinter.PchipInterpolator(psiNew,pNew)
	pEx = f(psiEx)
	checkExtension(psiEx, ne, Te, ni, Ti, pEx)
	return pEx


def writeProfile(x, y, key, tag = None, norm = None):
	if tag is None: tag = ''
	else: tag = '_' + tag
	if norm is None:
		if 'n' in key: norm = 1e20
		else: norm = 1e3
	with open('profile' + tag + '_' + str.lower(key),'w') as f:
		#f.write('# ' + key + ' profile in ' + units + ' \n')
		#f.write('# psi          ' + key + ' [' + units + '] \n')
		for i in range(len(x)):
			f.write(format(x[i],' 13.7e') + ' \t' + format(y[i]/norm,' 13.7e') + '\n')































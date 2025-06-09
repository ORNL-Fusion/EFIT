import numpy as np
from scipy.optimize import minimize
import scipy.interpolate as scinter
from scipy.integrate import simpson
from scipy.optimize import curve_fit

def residual(x, y, x0, y0):
	x = np.sort(x)
	idx = np.abs(np.diff(x)) > 0
	idx = np.append(idx,True)
	x[-idx] -= (np.arange(len(x)) +1)[-idx][::-1] * 1e-12
	f = scinter.UnivariateSpline(x, y, s = 0)
	res = (f(x0) - y0)**2
	return np.sum(res)
	

def get_optimal_knots(N, s, Y, method = 'SLSQP', quiet = False, smin = 0, smax = 1):
	# initial guess and bounds for each knot
	x0 = np.linspace(smin,smax,N)
	if smin == 0: bounds = [(0,0)]		# keep first knot fixed at 0
	else: bounds = [(smin,smax)]
	for i in range(1,N-1):
		bounds.append((smin,smax))
	if smax == 1: bounds.append((1,1))	# keep last knot fixed at 1
	else: bounds.append((smin,smax))
	
	# function to minimize
	f_wout = scinter.UnivariateSpline(s, Y, s = 0)
	res_wout = lambda x: residual(x, f_wout(x), s[(s>=smin) & (s<=smax)], Y[(s>=smin) & (s<=smax)])

	result = minimize(res_wout, x0, bounds = bounds, method = method, options = {'maxiter':500})
	if not quiet: print ('Success:', result.success, ',  N_iter =', result.nit)
	if not quiet: print (result.message)
	
	xres = result.x
	if smin > 0.6: xres = np.append(0.45,xres)
	if smin > 0.4: xres = np.append(0.2,xres)
	if smin > 0: xres = np.append(0,xres)
	if smax < 1: xres = np.append(xres,1)
	return xres, res_wout(xres)	


def get_linear_knots(N,s,y, add_knots = None):
	x0 = np.linspace(0,1,N)
	if add_knots is not None: x0 = add_in_knots(x0, add_knots)
	f_wout = scinter.UnivariateSpline(s, y, s = 0)
	f = scinter.UnivariateSpline(x0, f_wout(x0), s = 0)
	return np.sum((f(s) - y)**2)
	
	
def add_in_knots(x0, add_knots):
	for knot in add_knots:
		idx = abs(x0 - knot).argmin()
		x0_list = list(x0)
		if knot in x0_list: continue
		if x0[idx] < knot: x0_list.insert(idx+1,knot)
		else: x0_list.insert(idx,knot)
		x0 = np.array(x0_list)
	return x0


def fit_profile(xin,y, N = None, type = 'spline', param = None, xlim = None, truncate = None, points = 200, guess = None, asymptote = 0):
	if xlim is None: x1 = np.linspace(xin.min(),xin.max(),points)
	else: x1 = np.linspace(xlim[0],xlim[1],points)
	
	if truncate is not None:
		idx = xin < truncate
		xin = xin[idx].copy()
		y = y[idx].copy()
	
	if type == 'tanh':
		f = scinter.UnivariateSpline(xin, y, s = 0)
		
		# initial guess
		if guess is None: p0 = [0.97,0.06,f(0.85),0,0.06*(f(0)-f(0.85))/f(0.85),0,0,0,0]
		else: p0 = guess
	
		# fit profile
		f2 = lambda x,c0,c1,c2,c3,c4,c5,c6,c7,c8: tanh_multi(x,c0,c1,c2,c3,c4,c5,c6,c7,c8,param)
		popt,_ = curve_fit(f2, xin, y, p0 = p0)	
		y1 = tanh_multi(x1, *popt)
		return x1,y1,popt
		
	elif type == 'tanhflat':
		f = scinter.UnivariateSpline(xin, y, s = 0)
		
		# initial guess
		if guess is None: p0 = [0.97,0.06,f(0.85),0,0.06*(f(0)-f(0.85))/f(0.85),0,0]
		else: p0 = guess
	
		# fit profile
		f2 = lambda x,c0,c1,c2,c3,c4,c5,c6: tanh_flatout(x,c0,c1,c2,c3,c4,c5,c6,param)
		popt,_ = curve_fit(f2, xin, y, p0 = p0)
		y1 = tanh_flatout(x1, *popt)
		return x1,y1,popt
		
	elif type == 'tanh0':
		f = scinter.UnivariateSpline(xin, y, s = 0)
		
		# initial guess
		if guess is None: p0 = [0.97,0.06,f(0.85),0.06*(f(0)-f(0.85))/f(0.85),0,0]
		else: p0 = guess
	
		# fit profile
		f2 = lambda x,c0,c1,c2,c3,c4,c5: tanh_0out(x,c0,c1,c2,c3,c4,c5,param)
		popt,_ = curve_fit(f2, xin, y, p0 = p0)
		y1 = tanh_0out(x1, *popt)
		return x1,y1,popt
		
	elif type == 'tanhlin':
		f = scinter.UnivariateSpline(xin, y, s = 0)
		
		# initial guess
		if guess is None: p0 = [0.97,0.06,f(0.85),0.06*(f(0)-f(0.85))/f(0.85),0,0]
		else: p0 = guess
	
		# fit profile
		f2 = lambda x,c0,c1,c2,c3,c4,c5: tanh_lin(x,c0,c1,c2,c3,c4,c5)
		popt,_ = curve_fit(f2, xin, y, p0 = p0)
		y1 = tanh_lin(x1, *popt)
		return x1,y1,popt

	elif type == 'tanhfix':
		# fixed value for x -> inf
		f = scinter.UnivariateSpline(xin, y, s = 0)
		
		# initial guess
		if guess is None: p0 = [0.97,0.06,f(0.85),0.06*(f(0)-f(0.85))/f(0.85),0,0]
		else: p0 = guess
		
		# fit profile
		f2 = lambda x,c0,c1,c2,c4,c5,c6: tanh_flatout(x,c0,c1,c2,asymptote,c4,c5,c6)
		popt,_ = curve_fit(f2, xin, y, p0 = p0)
		y1 = tanh_flatout(x1, popt[0],popt[1],popt[2],asymptote,popt[3],popt[4],popt[5])
		return x1,y1,popt
		
	elif type == 'exp':		
		# This uses a simple exponential decay with the derivative at the last point of xin to match slopes and a fixes asymptote
		x0 = xin[-1]					# this should be  = 1
		y0 = y[-1]
		dx = xin[-1] - xin[-2]			# NOT equidistant
		dy = (-y[-2] + y[-1])/dx		# 1st order only due to non-equidistant grid
		
		if dy >= 0: 
			print('Exponential extension failed: ',x0,y0,dx,dy)
		
		# Fit exponential decay f(x) = a*exp(b*x) + c; c = asymptote is given as input
		c = asymptote
		b = dy/(y0 - c)
		a = dy/b * np.exp(-b*x0)	
		fexp = lambda x: a*np.exp(b*x) + c
		
		idx = np.where(x1 > x0)[0]
		y1 = np.append(y, fexp(x1[idx]))
		x1 = np.append(xin, x1[idx])
		return x1,y1,[a,b,c]
		
	else:
		x = xin - xin[0]
		norm = x[-1]
		x /= norm  # now from 0 -> 1
		f = scinter.UnivariateSpline(x, y, s = 0)
		
		if N is None: N = np.arange(4, 30)
		elif isinstance(N,int): N = [N]
		resOld = 1e+12
		for n in N: 
			xn0,res = get_optimal_knots(n,x,y, quiet = True)
			if res < resOld:
				resOld = res
				xn = xn0.copy()
			else: 
				print ('Optimum N:', n-1)
				break
	
		yn = f(xn)
		fn = scinter.UnivariateSpline(xn,yn, s = 0)

		y1 = fn(x1)
		x1 = x1*norm + xin[0]
		xn = xn*norm + xin[0]
		return x1,y1,xn,yn


def eichFit(xin,y, p0 = None, xlim = None, bg = False, fx = 1):
	if xlim is None: x1 = xin
	else: x1 = np.linspace(xlim[0],xlim[1],200)

	f = scinter.UnivariateSpline(xin, y, s = 0)
	
	# initial guess
	if p0 is None: 
		if bg: p0 = [50,10,0,2,0]
		else: p0 = [50,10,0,2]

	# fit profile
	if bg: f2 = lambda x,lq,S,s0,q0,qBG: eich_profile(x,lq,S,s0,q0,qBG,fx)
	else: f2 = lambda x,lq,S,s0,q0: eich_profile(x,lq,S,s0,q0,0,fx)
	popt,_ = curve_fit(f2, xin, y, p0 = p0)	
	y1 = eich_profile(x1, *popt, fx = fx)
	return x1,y1,popt


def optimize_profiles(infile, Nmax = 20, quiet = True, lin = False, smin = 0, smax = 1, add_knots_p = None, add_knots_c = None, add_knots_i = None):	
	import VMEC.Python.wout_class as WC
	import Misc.Fnml as Fnml

	if ('wout' in infile) & ('.nc' in infile):
		wout = WC.Wout(infile)
		nml = {'am_aux_s':wout.data['s'],'ac_aux_s':wout.data['s'],'ai_aux_s':wout.data['s'],
		'am_aux_f':wout.data['presf'],'ac_aux_f':wout.data['jcurv']*2*np.pi,'ai_aux_f':wout.data['iotaf']}
	else:
		nml = Fnml.read_Fnml(infile)
	
	N = np.arange(4, Nmax)
	res_p = np.zeros(N.shape)
	res_c = np.zeros(N.shape)
	res_i = np.zeros(N.shape)
	for i,n in enumerate(N):
		# Pressure
		norm = 1e+3
		if lin: res_p[i] = get_linear_knots(n, nml['am_aux_s'], nml['am_aux_f']/norm, add_knots_p)
		else: _,res_p[i] = get_optimal_knots(n, nml['am_aux_s'], nml['am_aux_f']/norm, method = 'SLSQP', quiet = quiet)
		# Current
		norm = 1e+6
		if lin: res_c[i] = get_linear_knots(n, nml['ac_aux_s'], nml['ac_aux_f']/norm, add_knots_c)
		else: _,res_c[i] = get_optimal_knots(n, nml['ac_aux_s'], nml['ac_aux_f']/norm, method = 'SLSQP', quiet = quiet, smin = smin, smax = smax)
		# Iota
		norm = 1
		if lin: res_i[i] = get_linear_knots(n, nml['ai_aux_s'], nml['ai_aux_f']/norm, add_knots_i)
		else: _,res_i[i] = get_optimal_knots(n, nml['ai_aux_s'], nml['ai_aux_f']/norm, method = 'SLSQP', quiet = quiet)
		
	import pylab as plt
	plt.figure()
	plt.semilogy(N, res_p, label = 'Pressure')
	plt.semilogy(N, res_c, label = 'Current')
	plt.semilogy(N, res_i, label = 'Iota')
	plt.legend()
	
	return N, res_p, res_c, res_i


def get_optimized_profiles(infile, n_p, n_c, n_i, lin = False, smin = 0, smax = 1, add_knots_p = None, add_knots_c = None, add_knots_i = None):
	import VMEC.Python.wout_class as WC
	import Misc.Fnml as Fnml

	if ('wout' in infile) & ('.nc' in infile):
		wout = WC.Wout(infile)
		nml = {'am_aux_s':wout.data['s'],'ac_aux_s':wout.data['s'],'ai_aux_s':wout.data['s'],
		'am_aux_f':wout.data['presf'],'ac_aux_f':wout.data['jcurv']*2*np.pi,'ai_aux_f':wout.data['iotaf'],
		'pcurr_type':wout.data['pcurr_type'],'CURTOR':wout.data['ctor']}
	else:
		nml = Fnml.read_Fnml(infile)
	
	# Pressure
	norm = 1e+3
	if lin: 
		x_p = np.linspace(0,1,n_p)
		if add_knots_p is not None: x_p = add_in_knots(x_p, add_knots_p)
	else: x_p,_ = get_optimal_knots(n_p, nml['am_aux_s'], nml['am_aux_f']/norm, method = 'SLSQP')
	# Current
	norm = 1e+6
	if lin: 
		x_c = np.linspace(0,1,n_c)
		if add_knots_c is not None: x_c = add_in_knots(x_c, add_knots_c)
	else: x_c,_ = get_optimal_knots(n_c, nml['ac_aux_s'], nml['ac_aux_f']/norm, method = 'SLSQP', smin = smin, smax = smax)
	# Iota
	norm = 1
	if lin: 
		x_i = np.linspace(0,1,n_i)
		if add_knots_i is not None: x_i = add_in_knots(x_i, add_knots_i)
	else: x_i,_ = get_optimal_knots(n_i, nml['ai_aux_s'], nml['ai_aux_f']/norm, method = 'SLSQP')
	
	f = scinter.UnivariateSpline(nml['am_aux_s'], nml['am_aux_f'], s = 0)
	y_p = f(x_p)
	f = scinter.UnivariateSpline(nml['ac_aux_s'], nml['ac_aux_f'], s = 0)
	y_c = f(x_c)
	I = simpson(f(np.linspace(0,1,300)))
	f = scinter.UnivariateSpline(nml['ai_aux_s'], nml['ai_aux_f'], s = 0)
	y_i = f(x_i)

	print ("  pmass_type = 'Akima_spline'")
	output = write_array('am_aux_s', x_p)
	for line in output: print (line)
	output = write_array('am_aux_f', y_p)
	for line in output: print (line)

	print ("  pcurr_type = '" + nml['pcurr_type'] + "'")
	output = write_array('ac_aux_s', x_c)
	for line in output: print (line)
	output = write_array('ac_aux_f', y_c)
	for line in output: print (line)

	print ("  piota_type = 'Akima_spline'")
	output = write_array('ai_aux_s', x_i)
	for line in output: print (line)
	output = write_array('ai_aux_f', y_i)
	for line in output: print (line)

	import pylab as plt
	plt.figure()
	f2 = scinter.UnivariateSpline(x_p, y_p, s = 0)
	plt.plot(nml['am_aux_s'], f2(nml['am_aux_s'])/1e+3,'r-', nml['am_aux_s'], nml['am_aux_f']/1e+3, 'k--', x_p, y_p/1e+3, 'rx')
	plt.xlabel('s'); plt.ylabel('p [kPa]')
	plt.figure()
	f2 = scinter.UnivariateSpline(x_c, y_c, s = 0)
	I2 = simpson(f2(np.linspace(0,1,300)))
	plt.plot(nml['ac_aux_s'], f2(nml['ac_aux_s'])/1e+6,'r-', nml['ac_aux_s'], nml['ac_aux_f']/1e+6, 'k--', x_c, y_c/1e+6, 'rx')
	plt.xlabel('s'); plt.ylabel('j$_{\\phi}$ [10$^6$ A/m$^2$]')
	plt.figure()
	f2 = scinter.UnivariateSpline(x_i, y_i, s = 0)
	plt.plot(nml['ai_aux_s'], f2(nml['ai_aux_s']),'r-', nml['ai_aux_s'], nml['ai_aux_f'], 'k--', x_i, y_i, 'rx')
	plt.xlabel('s'); plt.ylabel('$\\iota$')
	
	print ('\n New total current: CURTOR =', I2/I*nml['CURTOR'])
	print ('Remember to set NCURR = 1')
	return I2/I


# --- write_array ------------------------------------------------------------------------
# puts array into a formated string list, suitable for writing to file
def write_array(name, s, cols = 5):
	N = len(s)
	output = []
	output.append("  " + name + "  =  " + format(s[0],' 13.7E'))# + "\n")
	idx = 1
	while (idx < N):
		line =  '    ' + format(s[idx],' 13.7E')
		idx += 1
		for k in range(cols -1):
			if(idx < N): 
				line += '  ' + format(s[idx],' 13.7E')
			idx += 1
		output.append(line)# + '\n')
	return output


def tanh_0out(x, c0, c1, c2, c3, c4, c5, param=None):
	"""
	tanh function with cubic or quartic inner and constant outer leg with value=0
	and derivative=0 at param
	  c0 = SYMMETRY POINT
	  c1 = FULL WIDTH
	  c2 = HEIGHT
	  c3 = SLOPE OR QUADRATIC (IF ZERO DER) INNER
	  c4 = QUADRADIC OR CUBIC (IF ZERO DER) INNER
	  c5 = CUBIC OR QUARTIC (IF ZERO DER) INNER
	"""
	c = [c0,c1,c2,c3,c4,c5]
	z = 2.*(c[0]-x)/c[1]
	if param is None:
		pz1 = 1.+ c[3]*z + c[4]*z**2 + c[5]*z**3
	else:
		z0 = 2.*(c[0] - param)/c[1]
		cder = -(2.0*c[3]*z0 + 3.0*c[4]*z0**2 + 4.0*c[5]*z0**3)
		pz1 = 1 + cder*z + c[3]*z**2 + c[4]*z**3 + c[5]*z**4

	f = 0.5*c[2]* (pz1*np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z)) + 0.5*c[2]
	return f


def tanh_multi(x, c0,c1,c2,c3,c4,c5,c6,c7,c8, param = None):
	"""
	tanh function with cubic or quartic inner and linear
	to quadratic outer extensions and derivative=0 at param
	c0 = SYMMETRY POINT
	c1 = FULL WIDTH
	c2 = HEIGHT
	c3 = OFFSET
	c4 = SLOPE OR QUADRATIC (IF ZERO DER) INNER
	c5 = QUADRADIC OR CUBIC (IF ZERO DER) INNER
	c6 = CUBIC OR QUARTIC (IF ZERO DER) INNER
	c7 = SLOPE OUTER
	c8 = QUADRATIC OUTER
	"""
	c = [c0,c1,c2,c3,c4,c5,c6,c7,c8]
	z = 2.*(c[0]-x)/c[1]
	if param is None:
		pz1 = 1.+ c[4]*z + c[5]*z**2 + c[6]*z**3
	else:
		z0 = 2.*(c[0] - param)/c[1]
		cder = -(2.0*c[4]*z0 + 3.0*c[5]*z0**2 + 4.0*c[6]*z0**3)
		pz1 = 1 + cder*z + c[4]*z**2 + c[5]*z**3 + c[6]*z**4
		
	pz2 = 1 + (c[7]*z + c[8]*z**2)

	f = 0.5*(c[2]-c[3]) * (pz1*np.exp(z) - pz2*np.exp(-z))/(np.exp(z) + np.exp(-z)) + 0.5*(c[2]+c[3])
	return f


def tanh_lin(x, c0,c1,c2,c3,c4,c5):
	"""
	tanh function with linear inner and linear
	outer extensions
	c0 = SYMMETRY POINT
	c1 = FULL WIDTH
	c2 = HEIGHT
	c3 = OFFSET
	c4 = SLOPE INNER
	c5 = SLOPE OUTER
	"""
	c = [c0,c1,c2,c3,c4,c5]
	z = 2.*(c[0]-x)/c[1]
	pz1 = 1. + c[4]*z
	pz2 = 1. + c[5]*z
	f = 0.5*(c[2]-c[3]) * (pz1*np.exp(z) - pz2*np.exp(-z))/(np.exp(z) + np.exp(-z)) + 0.5*(c[2]+c[3])
	return f


def tanh_flatout(x, c0,c1,c2,c3,c4,c5,c6, param = None):
	"""
	tanh function with cubic or quartic inner and constant outer extensions
	and derivative=0 at param
	c0 = SYMMETRY POINT
	c1 = FULL WIDTH
	c2 = HEIGHT
	c3 = OFFSET
	c4 = SLOPE OR QUADRATIC (IF ZERO DER) INNER
	c5 = QUADRADIC OR CUBIC (IF ZERO DER) INNER
	c6 = CUBIC OR QUARTIC (IF ZERO DER) INNER
	"""
	c = [c0,c1,c2,c3,c4,c5,c6]
	z = 2.*(c[0]-x)/c[1]
	if param is None:
		pz1 = 1.+ c[4]*z + c[5]*z**2 + c[6]*z**3
	else:
		z0 = 2.*(c[0] - param)/c[1]
		cder = -(2.0*c[4]*z0 + 3.0*c[5]*z0**2 + 4.0*c[6]*z0**3)
		pz1 = 1 + cder*z + c[4]*z**2 + c[5]*z**3 + c[6]*z**4
		
	f = 0.5*(c[2]-c[3]) * (pz1*np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z)) + 0.5*(c[2]+c[3])
	return f
	
	
def eich_profile(x, lq, S, s0, q0, qBG = 0, fx = 1):
	"""
	Based on the paper: T.Eich et al.,PRL 107, 215001 (2011)
	lq is heat flux width at midplane in mm
	S is the private flux region spreading in mm
	s0 is the separatrix location at Z = 0 in m
	q0 is the amplitude
	qBG is the background heat flux
	fx is the flux expansion between outer midplane and target plate
	x is in m
	return function q(x)
	
	in Eich paper: s (here x) and s0 are distances along target, mapped from midplane using flux expansion,
	so: s = s_midplane * fx; same for s0, with s0 the position of strikeline on target
	Here, use s_midplane directly, so set fx = 1 and identify s = s_midplane = R and s0 = Rsep
	"""
	from scipy.special import erfc
	lq *= 1e-3		# in m now
	S *= 1e-3		# in m now
	a = lq*fx
	b = 0.5*S/lq
	c = S*fx
	q = 0.5 * q0 * np.exp(b**2 - (x-s0)/a) * erfc(b - (x-s0)/c) + qBG
	return q
	
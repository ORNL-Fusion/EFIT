# Bsquare_class.py
# Author: A. Wingen
# Date: Apr. 10. 2013
# ----------------------------------------------------------------------------------------
# Calculates flux surface averaged B**2 = B_pol**2(psi) + B_tor**2(psi), and
# calculates flux surface averaged parallel current density: < J dot B >(psi)
# psi can either be a 1D-array or a float
#
# Initialize by:
#	import Bsquare_class
#	Bsq = Bsquare_class.Bsquare(shot, time, gpath)
# with:
#	shot (int)  	->  shot number (default is 148712)
#	time (int)  	->  time in ms  (default is 4101)
# 	gpath (string)	->	path to g-file	(default is path to g148712.04101)
#
# For B**2 call: 		B = Bsq.ev(psi)
# For j_parallel call: 	j = Bsq.jpar(psi)
# with:
#	psi (1D-array or float)	->	normalized flux
#		e.g.:  psi = linspace(0.001, 0.999, 100)
#
# ----------------------------------------------------------------------------------------

from numpy import *
import scipy.interpolate as inter

import Misc.convertRZ as RZ
import EFIT.load_gfile_d3d as load_gfile_d3d

class Bsquare:

	def __init__(self, shot = 148712, time = 4101, gpath = '/Users/wingen/c++/d3d/gfiles/kinetic'):
		g = load_gfile_d3d.read_g_file(shot, time, gpath)
		dpsidR, dpsidZ = self.derive_psi(g['R'], g['Z'], g['psiRZ'], g['dR'], g['dZ']) # not normalized
		
		# Member variables
		self.g = g
		self.get_Fpol = inter.UnivariateSpline(linspace(0.0,1.0,g['NR']), g['Fpol'], s = 0)
		self.get_FFprime = inter.UnivariateSpline(linspace(0.0,1.0,g['NR']), g['FFprime'], s = 0)
		self.get_Pprime = inter.UnivariateSpline(linspace(0.0,1.0,g['NR']), g['Pprime'], s = 0)
		self.get_psi = inter.RectBivariateSpline(g['R'], g['Z'], g['psiRZn'].T)		# normalized
		self.get_dpsidR = inter.RectBivariateSpline(g['R'], g['Z'], dpsidR.T)		# not normalized
		self.get_dpsidZ = inter.RectBivariateSpline(g['R'], g['Z'], dpsidZ.T)		# not normalized

	#-------------------------------------------------------------------------------------
	# Member Functions
	
	# --- dpsidR, dpsidZ = derive_psi(R, Z, psiRZ, dR, dZ) ---
	# get both partial derivatives for psiRZ, NOT normalized!
	def derive_psi(self, R, Z, psiRZ, dR, dZ):
		dpsidR = zeros(psiRZ.shape)
		dpsidZ = zeros(psiRZ.shape)
	
		for i in xrange(1,len(R)-1):
			dpsidR[:,i] = 0.5*(psiRZ[:,i+1] - psiRZ[:,i-1])/dR
		
		for i in xrange(1,len(Z)-1):
			dpsidZ[i,:] = 0.5*(psiRZ[i+1,:] - psiRZ[i-1,:])/dZ
		
		return dpsidR, dpsidZ
	
	
	# --- jpar = jpar(psi) ---
	# calculates flux surface averaged parallel current density: < J dot B >
	# in A/m^2
	# psi can either be a 1D-array or a float
	def jpar(self, psi):
		mu0 = 4*pi*1e-7
		Bsq = self.ev(psi)
		Fpol = self.get_Fpol(psi)
		Fprime = self.get_FFprime(psi)/Fpol
		Pprime = self.get_Pprime(psi)
		jpar = (Fprime*Bsq/mu0 + Pprime*Fpol)/abs(self.g['Bt0'])

		return jpar
		

	# --- Bsq = ev(psi) ---
	# calculates flux surface averaged B_pol(psi)
	# psi can either be a 1D-array or a float
	# psi = 0.999798 is an empirically found upper boundary
	# for psi > 0.999798 B_square gets too large, because the flux surface becomes open
	def ev(self, psi):
		if isinstance(psi, ndarray):
			if(psi[-1] == 1.0): psi = psi.copy(); psi[-1] = 0.999798
			n = len(psi)
			Bsq = zeros(n)
			for i in xrange(n):
				Bsq[i] = self.B_square(psi[i])
				#print 'done:', i
		else:
			if(psi >= 0.999798): psi = 0.999798
			Bsq = self.B_square(psi)
	
		return Bsq


	# --- Bsq = B_square(psi0, N = 500) ---
	# returns flux surface averaged B**2 for one psi = psi0
	def B_square(self, psi0, N = 500):
		R, Z = self.flux_surface(psi0, N, 'bisec')
		B_th = self.B_pol(R, Z)
		B_t = self.get_Fpol(psi0)*ones(R.shape)/R
		#Bsq = sum(B_t**2 + B_th**2)/N
	
		# Integrate along surface
		B = 0; L = 0;
		for i in xrange(N-1):
			dli = sqrt((R[i+1]-R[i])**2 + (Z[i+1]-Z[i])**2)
			B += sqrt(B_t[i]**2 + B_th[i]**2)*dli
			L += dli
		
		# periodic boundary condition
		dli = sqrt((R[0]-R[N-1])**2 + (Z[0]-Z[N-1])**2)
		B += sqrt(B_t[N-1]**2 + B_th[N-1]**2)*dli
		L += dli
	
		# get surface average
		B /= L
	
		return B**2


	# --- B_th = B_pol(R, Z) ---
	# calculates B_poloidal for each point in the arrays R, Z 
	def B_pol(self, R, Z):
		B_R = self.get_dpsidZ.ev(R,Z)/R
		B_Z = -self.get_dpsidR.ev(R,Z)/R
	
		theta = RZ.get_theta(R, Z, self.g)
		sinp = sin(theta)
		cosp = cos(theta)
	
		#B_r = B_R*cosp + B_Z*sinp
		B_th = -B_R*sinp + B_Z*cosp
	
		return B_th


	# --- R, Z = flux_surface(psi0, N) ---
	# returns arrays R and Z of N points along psi = const. surface
	def flux_surface(self, psi0, N, method = 'best'):
		theta = linspace(0,2*pi,N+1)[0:-1]
		r = zeros(theta.shape)
	
		if(method == 'best'):
			for i in xrange(N): 
				try:
					r[i] = self.newton(psi0, theta[i], 0.25*sqrt(psi0))
				except RuntimeError:
					#print i, 'redo it with bisec'
					r[i] = self.bisec(psi0, theta[i])
		else:
			for i in xrange(N): r[i] = self.bisec(psi0, theta[i])
	
		R = r*cos(theta) + self.g['RmAxis']
		Z = r*sin(theta) + self.g['ZmAxis']
	
		return R, Z
		
		
	# --- f = funct(r, theta, psi0) ---
	# get f(r) = psi(R(theta,r),Z(theta,r))-psi0 with theta = const.
	def funct(self, r, theta, psi0, get_df = 0):
		R = r*cos(theta) + self.g['RmAxis']
		Z = r*sin(theta) + self.g['ZmAxis']	
		psi = self.get_psi.ev(R, Z)
	
		# exclude private flux region
		#if((psi < 1.2) & (Z < g['lcfs'][:,1].min())):
		#	psi = 1.2

		f = psi - psi0
	
		if(get_df == 1):
			df = self.get_dpsidR.ev(R,Z)*cos(theta) + self.get_dpsidZ.ev(R,Z)*sin(theta)
			df /= (self.g['psiSep'] - self.g['psiAxis'])	# use normalized psi here
			return f, df
		else: return f


	# --- r = bisec(psi0, theta, a = 0, b = 1.5) ---
	# get r for theta = const. and psi = psi0
	def bisec(self, psi0, theta, a = 0, b = 1.5):
		eps = 1e-14
	
		x = a
		f = self.funct(x, theta, psi0)
	
		if(f > 0):
			xo = a
			xu = b
		else:
			xo = b
			xu = a
	
		while(abs(xo-xu) > eps):
			x = (xo + xu)/2.0
			f = self.funct(x, theta, psi0)
			if(f > 0): xo = x
			else: xu = x
	
		return x


	# --- r = newton(psi0, theta, x0 = 0.5) ---
	# get r for theta = const. and psi = psi0
	def newton(self, psi0, theta, x0 = 0.3):
		eps = 1e-14	
		x = x0
		dx = 1
		n = 0
	
		while(abs(dx) > eps):
			f, df = self.funct(x, theta, psi0, 1)
			if(df == 0): 		# compensate rare case of divide by zero
				x = 0.5*(x+x0)
				continue
				
			dx = f/df
			x -= dx
			n += 1
			if(n >= 10): 
				raise RuntimeError('newton: no convergence')
	
		return x
	

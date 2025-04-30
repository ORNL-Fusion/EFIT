# IMAS_EQ.py
# description:	reads a netcdf, hdf5, json, equilibrium file formatted per IMAS/OMAS
# engineer:		T Looby
# date:			20241030

import os
import numpy as np
import scipy.interpolate as scinter

from . import extend_profiles as expro

class netCDF_IMAS:
	def __init__(self):
		"""
		Constructor
		"""
		self.data = {}
		return

	def readNetCDF(self, filename, time):
		"""
		reads from IMAS netCDF and assigns to the parameters we use in the equilParams_class ep object

		THIS FUNCTION IS NOT YET IMPLEMENTED
		NEEDS TO BE UPDATED WITH LATEST IMAS SCHEMA
				
		"""
		import netCDF4

		nc = netCDF4.Dataset(filename)
		nc.close()
		d = {}
		print("THIS FUNCTION NOT YET IMPLEMENTED.")
		return d
	



class JSON_IMAS:
	def __init__(self, filename):
		"""
		Constructor
		reads the IMAS JSON file
		"""
		import json
		
		self.eqd = None
		self.psiN = None
		self.profiles = None
		self.shot = None
		self.time = None
		
		with open(filename, 'r') as file:
			self.data = json.load(file) 
		
		return


	def getEQ(self, time, shot = None, psiMult = 1.0, BtMult = 1.0, IpMult = 1.0):
		"""
		Sets the time slice and assigns to the parameters we use in the equilParams_class ep object
		"""		   
		try:
			tIdx = np.where(np.round(np.array(self.data['equilibrium']['time']),8) == time)[0][0]
		except:
			print("Could not find timestep " + str(time) + " in JSON equilibrium dict.	Aborting.")
			return

		eqt = self.data['equilibrium']['time_slice'][tIdx]
		wall = self.data['wall']	
		if shot is None: self.shot = 1
		self.time = time	

		d = {}
		#ep object name left of '='
		d['R1D'] = np.array(eqt['profiles_2d'][0]['grid']['dim1'])
		d['Z1D'] = np.array(eqt['profiles_2d'][0]['grid']['dim2'])
		d['nw'] = len(d['R1D'])
		d['nh'] = len(d['Z1D']) 
		d['rcentr'] = np.array(self.data['equilibrium']['vacuum_toroidal_field']['r0'])
		d['bcentr'] = np.array(self.data['equilibrium']['vacuum_toroidal_field']['b0'][tIdx]) * BtMult
		d['rmaxis'] = np.array(eqt['global_quantities']['magnetic_axis']['r'])
		d['zmaxis'] = np.array(eqt['global_quantities']['magnetic_axis']['z'])
		d['Rmin'] = np.min(eqt['profiles_2d'][0]['grid']['dim1'])
		d['Rmax'] = np.max(eqt['profiles_2d'][0]['grid']['dim1'])
		d['Rlcfs'] = np.array(eqt['boundary']['outline']['r'])
		d['Zlcfs'] = np.array(eqt['boundary']['outline']['z'])
		d['Rbdry'] = np.max(d['Rlcfs'])
		d['Zmin'] = np.min(eqt['profiles_2d'][0]['grid']['dim2'])
		d['Zmax'] = np.max(eqt['profiles_2d'][0]['grid']['dim2'])
		d['Zlowest'] = np.min(d['Zlcfs'])
		d['siAxis'] = np.array(eqt['global_quantities']['psi_axis']) * psiMult
		d['siBry'] = np.array(eqt['global_quantities']['psi_boundary']) * psiMult

		# 1D profiles (if they arent nw long, interpolate them to be nw long)
		psiN = np.linspace(0,1,d['nw'])
		d['fpol'] = np.array(eqt['profiles_1d']['f'])
		if len(d['fpol']) != d['R1D']:
			d['fpol'] = np.interp(psiN, np.linspace(0,1,len(d['fpol'])), d['fpol']) * BtMult
		d['ffprime'] = np.array(eqt['profiles_1d']['f_df_dpsi'])
		if len(d['ffprime']) != d['R1D']:
			d['ffprime'] = np.interp(psiN, np.linspace(0,1,len(d['ffprime'])), d['ffprime'])
		d['pprime'] = np.array(eqt['profiles_1d']['dpressure_dpsi'])
		if len(d['pprime']) != d['R1D']:
			d['pprime'] = np.interp(psiN, np.linspace(0,1,len(d['pprime'])), d['pprime'])
		d['pres'] = np.array(eqt['profiles_1d']['pressure'])
		if len(d['pres']) != d['R1D']:
			d['pres'] = np.interp(psiN, np.linspace(0,1,len(d['pres'])), d['pres'])
		d['qpsi'] = np.array(eqt['profiles_1d']['q'])
		if len(d['qpsi']) != d['R1D']:
			d['qpsi'] = np.interp(psiN, np.linspace(0,1,len(d['qpsi'])), d['qpsi'])

		#2D profiles
		d['psirz'] = np.array(eqt['profiles_2d'][0]['psi']).T * psiMult
		if d['psirz'].shape[0] != d['nh']: d['psirz'] = d['psirz'].T
		
		d['lcfs'] = np.vstack((d['Rlcfs'], d['Zlcfs'])).T
		d['Rwall'] = np.array(wall['description_2d'][0]['limiter']['unit'][0]['outline']['r'])
		d['Zwall'] = np.array(wall['description_2d'][0]['limiter']['unit'][0]['outline']['z'])
		d['wall'] = np.vstack((d['Rwall'], d['Zwall'])).T
		d['rdim'] = d['Rmax'] - d['Rmin']
		d['zdim'] = d['Zmax'] - d['Zmin']
		#d['R0'] = eqt['global_quantities']['magnetic_axis']['r']
		d['R0'] = d['rcentr']
		d['R1'] = d['Rmin']
		d['Zmid'] = 0.0
		d['Ip'] = eqt['global_quantities']['ip'] * IpMult
		d['thetapnts'] = 2*d['nw']
		d['Rsminor'] = np.linspace(d['rmaxis'], d['Rbdry'], d['nw'])
		self.eqd = d
		self.psiN = psiN
		return d
	
	
	def coreProfiles(self, time, dx = 0.005, xmin = 0.7, xmax = 1.2, nsol = 0.02, Tsol = 1e-4, 
						preservePoints = True, extendForM3DC1 = False, correctionMargin = None, 
						usePressureFromEQDSK = True, interlace = True):
		"""
		Sets the time slice and gets the profiles form the JSON
		Sets the member variable self.profiles, a dictionary with keys: ['time', 'ne', 'Te', 'p', 'V', 'rho', 'psi', 'ions', 'D', 'extend']
		'D' has the main ion profiles 
		'extend' has all the profiles interpolated and extrapolated to 1.2
		If the routine returns 'Extension okay', then it is guaranteed that p - sum(n*T) >= 0 everywhere
		This requires that main ion ni is slightly modified at the separatrix and in the SOL
		dx = final resolution in psi of the profile
		xmin = min psi for profile fit; the profile fit takes data only from x > xmin; For x < xmin splines are used.
	  	xmax = max psi for profile fit; this is the extrapolation limit
	  	nsol = asymptotic value of SOL density for psi -> inf
	  	Tsol = asymptotic value of SOL temperature for psi -> inf
	  	preservePoints = Keep original psi grid points, otherwise replace with a linspace, default is True
	  	extendForM3DC1 = Assume ni = ne, adjust Te instead of ni and set preservePoints = False
	  	correctionMargin = value < 1, but close to 1, default is 0.99, to multiply sum(n*T) so that sum(n*T) < p even for interpolated values.
	  	usePressureFromEQDSK = Use pressure profile from the actual geqdsk in self.eqd instead of the pressure profile in data['core_profiles']
	  	interlace = try to combine pressure profiles in eqd and core_profiles. This is not likely to work, so check the figure and use interactive response to decide.
		""" 		
		Ntimes = len(self.data['core_profiles']['profiles_1d'])
		times = [self.data['core_profiles']['profiles_1d'][i]['time'] for i in range(Ntimes)]	   
		try:
			tIdx = np.where(np.round(np.array(times),8) == time)[0][0]
		except:
			print("Could not find timestep " + str(time) + " in JSON equilibrium dict.	Aborting.")
			return
			
		if self.eqd is None: _ = self.getEQ(time)
		
		profiles = self.data['core_profiles']['profiles_1d'][tIdx]
		psiAxis = profiles['grid']['psi'][0]
		psiSep = profiles['grid']['psi'][-1]
		psiN = (np.array(profiles['grid']['psi']) - psiAxis)/(psiSep - psiAxis)
		self.profiles = {'time':profiles['time'], 
			'ne':np.array(profiles['electrons']['density']), 'Te':np.array(profiles['electrons']['temperature']), 
			'p':np.array(profiles['pressure_thermal']), 'V':np.array(profiles['grid']['volume']), 
			'rho':np.array(profiles['grid']['rho_tor_norm']), 'psi':psiN, 'psiPres':psiN} 
			
		ions = [profiles['ion'][item]['label'] for item in range(len(profiles['ion']))]
		self.profiles['ions'] = ions
		for i,ion in enumerate(ions):
			self.profiles[ion] = {'ni':np.array(profiles['ion'][i]['density'])}
			if 'temperature' in profiles['ion'][i]: self.profiles[ion]['Ti'] = np.array(profiles['ion'][i]['temperature'])
				
		# Interpolate profiles to resolution dx, and extrapolate n,T profiles to psi = xmax
		if extendForM3DC1: preservePoints = False
		else: usePressureFromEQDSK = False
		self.profiles['extend'] = {}
		asymptote = [nsol,Tsol]	# this is already normalized
		norm = [1e20, 1e3]	# normalize before profile fitting
		
		for i,key in enumerate(['ne','Te']):
			x,y = expro.make_profile(self.profiles['psi'], self.profiles[key]/norm[i], key, asymptote = asymptote[i], show = False, xmin = xmin, xmax = xmax, dx = dx, preservePoints = preservePoints) 
			self.profiles['extend'][key] = y*norm[i]
		self.profiles['extend']['psi'] = x
				
		f = scinter.UnivariateSpline(self.profiles['psi'], self.profiles['V'], s = 0)
		self.profiles['extend']['psi1'] = self.profiles['extend']['psi'][self.profiles['extend']['psi'] <= 1]
		self.profiles['extend']['V'] = f(self.profiles['extend']['psi1'])
		
		f = scinter.UnivariateSpline(self.profiles['psi'], self.profiles['rho'], s = 0)
		self.profiles['extend']['rho'] = f(self.profiles['extend']['psi1'])	# rho is only interpolated and ends at separatrix
		self.profiles['extend']['rho'][0] = 0	# make sure the end points are exact.
		self.profiles['extend']['rho'][-1] = 1
				
		for ion in self.profiles['ions']:
			self.profiles['extend'][ion] = {}
			for i,key in enumerate(['ni','Ti']):
				if key in self.profiles[ion]:
					try: x,y = expro.make_profile(self.profiles['psi'], self.profiles[ion][key]/norm[i], key, asymptote = asymptote[i], show = False, xmin = xmin, xmax = xmax, dx = dx, preservePoints = preservePoints)
					except: y = np.zeros(self.profiles['extend']['psi'].shape)
					self.profiles['extend'][ion][key] = y*norm[i]
						
		# pressure needs special consideration: 
		# it needs to be splined for psi <= 1; do NOT use profile fit; it does not preserve original points, but makes a least square fit
		# on the other hand the extrapolation for psi > 1 needs to be monotonic and tanh asymptotic; regular cubic splines cannot do that
		# start with interpolate
		# Use a monotonic interpolation!!!! However, this is not as smooth as a regular spline. The first derivatives are guaranteed to be continuous, but the second derivatives may jump
		if usePressureFromEQDSK:
			chk = -1
			if interlace:
				chk = self.interlacePressure(self.profiles['psi'], self.profiles['p'], self.psiN, self.eqd['pres'])
			if chk == -1:
				self.profiles['psiPres'], self.profiles['p'] = self.psiN, self.eqd['pres']
			f = scinter.PchipInterpolator(self.profiles['psiPres'], self.profiles['p'])
			self.profiles['extend']['p'] = f(self.profiles['extend']['psi1'])
		else:
			self.profiles['psiPres'] = self.profiles['psi']
			f = scinter.PchipInterpolator(self.profiles['psi'], self.profiles['p'])
			self.profiles['extend']['p'] = f(self.profiles['extend']['psi1'])
		
		if extendForM3DC1: self.correct_ne(asymptote = nsol*norm[0], correctionMargin = correctionMargin)
		else: self.correct_ni(asymptote = nsol*norm[0], correctionMargin = correctionMargin)	# here usePressureFromEQDSK = False
		self.extendPressure()
		return


	def sanityCheck(self, time, m3dc1 = True):
		import matplotlib.pyplot as plt
		e = 1.60217663e-19
		Ntimes = len(self.data['core_profiles']['profiles_1d'])
		times = [self.data['core_profiles']['profiles_1d'][i]['time'] for i in range(Ntimes)]	   
		try:
			tIdx = np.where(np.round(np.array(times),8) == time)[0][0]
		except:
			print("Could not find timestep " + str(time) + " in JSON equilibrium dict.	Aborting.")
			return
			
		if self.eqd is None: _ = self.getEQ(time)
		
		profiles = self.data['core_profiles']['profiles_1d'][tIdx]
		psiAxis = profiles['grid']['psi'][0]
		psiSep = profiles['grid']['psi'][-1]
		psiN = (np.array(profiles['grid']['psi']) - psiAxis)/(psiSep - psiAxis)		
		
		psieq, peq = self.psiN, self.eqd['pres']
		pth = np.array(profiles['pressure_thermal'])
		ne = np.array(profiles['electrons']['density'])
		Te = np.array(profiles['electrons']['temperature'])
		ni = np.array(profiles['ion'][0]['density'])
		Ti = np.array(profiles['ion'][0]['temperature'])
		
		if m3dc1:
			psum = ne*(Te + Ti)*e
		else:
			psum = ne*Te*e + ni*Ti*e
		
		plt.figure()
		plt.plot(psieq, peq*1e-3, 'k-', lw = 2, label = 'geqdsk')
		plt.plot(psiN, pth*1e-3, 'b-', lw = 2, label = 'thermal')
		plt.plot(psiN, psum*1e-3, 'g-', lw = 2, label = 'ne*Te + ni*Ti')
		plt.xlabel('$\\psi$')
		plt.ylabel('pressure [kPa]')
		plt.xlim(0,1)
		plt.legend()
		
		

	def interlacePressure(self, psiProf, pProf, psiEQD, pEQD):
		"""
		If the pressure profiles in EQD and Profiles have different knots, but 'visually' match, then interlace them to increase overall resolution
		"""
		if len(psiProf) == len(psiEQD):
			if np.abs(psiProf - psiEQD).max() < 1e-6:
				print('Pressure profile knots are identical, no interlacing possible')
				return -1
				
		import matplotlib.pyplot as plt
		fig = plt.figure()
		plt.plot(psiProf, pProf*1e-3, 'ko-', lw = 2, label = 'Profiles pressure')
		plt.plot(psiEQD, pEQD*1e-3, 'bo-', lw = 2, label = 'EQD pressure')
		plt.xlim(0,1)
		plt.xlabel('$\\psi$')
		plt.ylabel('p$ [kPa]')
		plt.legend()
		
		x = np.linspace(0,1,300)
		fProf = scinter.PchipInterpolator(psiProf, pProf)
		fEQD = scinter.PchipInterpolator(psiEQD, pEQD)
		
		if np.abs(fProf(x) - fEQD(x)).max() < 1e-6:	
			print('Profiles similar. Attempting interlace')
		else:
			plt.plot(x, (fProf(x) - fEQD(x))*1e-3, 'r-', lw = 2, label = 'difference')
			plt.legend()
			print('Profiles appear different. Verify with plot if interlacing should be done.')
			answer = input("Proceed (y/n)?: ")
			if answer in ['n', 'no', 'No', 'N', 'False', 'false']:
				plt.close(fig) 
				return -1
		
		psiAll = np.append(psiProf,psiEQD)
		psiOrder = np.argsort(psiAll)
		psiAll = psiAll[psiOrder]
		pAll = np.append(pProf,pEQD)[psiOrder]
		d = np.abs(np.diff(psiAll))
		idx = np.where(d < 1e-12)[0]
		psiAll = np.delete(psiAll,idx)
		pAll = np.delete(pAll,idx)
		
		plt.plot(psiAll, pAll*1e-3, 'g-', lw = 2, label = 'Combined pressure')
		plt.legend()
		self.profiles['psiPres'], self.profiles['p'] = psiAll, pAll
		return 0
		

	def correct_ne(self, asymptote = 0, correctionMargin = None):
		"""		
		Correct ne, so that p - sum(n*T) >= 0 for all points in p = self.profiles['p']
		This ignores the original ni profile and assumes ni = ne, as required in M3D-C1, also ignore any impurities
		Then extrapolate ne using exponential decay and interpolate ne on extended psi grid
		"""
		fne = scinter.UnivariateSpline(self.profiles['psi'],self.profiles['ne'],s = 0)	# remap ne if usePressureFromEQDSK, else this does nothing
		ion = self.profiles['ions'][0]
		psiEx, Te, Ti = self.profiles['extend']['psi'], self.profiles['extend']['Te'], self.profiles['extend'][ion]['Ti']
		psi0, p0, ne = self.profiles['psiPres'], self.profiles['p'], fne(self.profiles['psiPres'])
		neNew = expro.correct_ne(psi0, p0, ne, psiEx, Te, Ti, asymptote = asymptote, correctionMargin = correctionMargin)
		self.profiles['extend']['ne'] = neNew
		self.profiles['extend'][ion]['ni'] = self.profiles['extend']['ne'].copy()


	def extendPressure(self):
		"""
		Extend the pressure using sum(n*T) for psi > 1
		"""
		ion = self.profiles['ions'][0]
		psiEx, ne, Te = self.profiles['extend']['psi'], self.profiles['extend']['ne'], self.profiles['extend']['Te']
		ni, Ti = self.profiles['extend'][ion]['ni'], self.profiles['extend'][ion]['Ti']
		psi0, p0 = self.profiles['extend']['psi1'], self.profiles['extend']['p']	# here on psi1 grid only
		pEx = expro.extendPressure(psiEx, ne, Te, ni, Ti, psi0, p0)
		self.profiles['extend']['p'] = pEx	# now on psiEx grid


	def checkExtension(self):
		"""
		#Verify that p - sum(n*T) >= 0
		#Using the extended psi grid and original grid
		"""
		ion = self.profiles['ions'][0]
		psiEx, ne, Te = self.profiles['extend']['psi'], self.profiles['extend']['ne'], self.profiles['extend']['Te']
		ni, Ti, pEx = self.profiles['extend'][ion]['ni'], self.profiles['extend'][ion]['Ti'], self.profiles['extend']['p']
		expro.checkExtension(psiEx, ne, Te, ni, Ti, pEx)


	def correct_ni(self, asymptote = 0, correctionMargin = None):
		"""		
		Correct main ion ni, so that p - sum(n*T) >= 0 for all points in p = self.profiles['p']
		Then extrapolate ni using exponential decay and interpolate ni on extended psi grid
		"""
		e = 1.60217663e-19
		if correctionMargin is None: correctionMargin = 0.99
		
		# Spline ne,te and ti for the original psi
		fne = scinter.UnivariateSpline(self.profiles['extend']['psi'], self.profiles['extend']['ne'], s = 0)
		fte = scinter.UnivariateSpline(self.profiles['extend']['psi'], self.profiles['extend']['Te'], s = 0)
		ion = self.profiles['ions'][0]
		ion0 = self.profiles['ions'][0]
		fti = scinter.UnivariateSpline(self.profiles['extend']['psi'], self.profiles['extend'][ion]['Ti'], s = 0)
		if len(self.profiles['ions']) > 1:
			y = 0
			for i in range(1,len(self.profiles['ions'])): 
				ion = self.profiles['ions'][i]
				if 'Ti' in self.profiles['extend'][ion]:
					y += self.profiles['extend'][ion]['ni'] * self.profiles['extend'][ion]['Ti']*e
				else:
					y += self.profiles['extend'][ion]['ni'] * self.profiles['extend'][ion0]['Ti']*e
			fnT_imp = scinter.UnivariateSpline(self.profiles['extend']['psi'], y, s = 0)
		else: fnT_imp = lambda x: 0
		
		# with original ni get d = p - sum(n*T), with original p, and extended/splined n & T (except ni)
		ion = self.profiles['ions'][0]
		x = self.profiles['psi']
		netex, tix, impx = fne(x) * fte(x)*e, fti(x)*e, fnT_imp(x)
		p = netex + self.profiles[ion]['ni'] * tix + impx 
		d = self.profiles['p'] - p
		
		# find points where d < 0
		idx = np.where(d < 0)[0]
		if len(idx) == 0: return	# done, d >=0 everywhere already
		
		# where d < 0 replace ni with new value so that  p - sum(n*T) >= 0 -> ni patched
		ni0 = (self.profiles['p'] - netex - impx) / tix		# this ni would make d = 0 everywhere
		niPatched = self.profiles[ion]['ni'].copy()
		niPatched[idx] = ni0[idx] * correctionMargin	# give it a tiny margin
		
		# Use a monotonic interpolation for ni_patched -> upscale ni to extended psi grid
		# !!!!!!! This interpolator does not overshoot like UnivariateSpline, but instead maintains a monotonic curve !!!!!!!!!!!!
		# However, this is not as smooth as a regular spline. The first derivatives are guaranteed to be continuous, but the second derivatives may jump
		f = scinter.PchipInterpolator(x, niPatched)
		sol = self.profiles['extend']['psi'] > 1
		psiCore = self.profiles['extend']['psi'][~sol]
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
		niNew = np.append(y, f(self.profiles['extend']['psi'][sol]))
		
		# verify & update
		fni = scinter.UnivariateSpline(self.profiles['extend']['psi'], niNew, s = 0)
		p2 = netex + fni(x) * tix + impx 	# Use original psi grid
		d2 = self.profiles['p'] - p2
		if any(d2 < 0): 
			print('Sum of n*T exceeds thermal pressure inside the separatrix. Check extended profiles.')
		else: 
			print('ni correction okay')
			self.profiles['extend'][ion]['ni'] = niNew
		return #x,p,d,ni0,niPatched,niNew,p2,d2
		
		
	def plotProfile(self, what = 'all', fig = None, c = None, label = '', extended = False, style = None):
		"""
		what: keyword of what profile to plot. default is 'all' and plots all 6 relevant profiles
		fig: integer number of figure window to use, e.g. 1
		c: string of color code, e.g. 'k' or 'r'
		label: string that becomes the label for the plot in the legend
		extended: plot the extended profiles
		"""
		import matplotlib.pyplot as plt
		if c is None: c = 'k'
		if style is None: style = '-'
		
		if extended: profiles = self.profiles['extend']
		else: profiles = self.profiles
		x = profiles['psi']
		species = self.profiles['ions'][0]	# just D for now
		
		if what in ['p','P','Pres','pres','pressure','Ptot','ptot','Press','press','Pressure']:
			ylabel = 'p$_{th}$ [kPa]'
			y = profiles['p']*1e-3
			x = self.profiles['psiPres']
		elif what in ['ne','density']:
			ylabel = 'n$_e$ [10$^{20}$/m$^{3}$]'
			y = profiles['ne']*1e-20
		elif what in ['te','Te','temperature']:
			ylabel = "T$_{e}$ [keV]"
			y = profiles['Te']*1e-3
		elif what in ['V','volume']:
			ylabel = "V [m$^3$]"
			x = self.profiles['psi']
			y = profiles['V']
		elif what in ['ni','iondensity']:
			ylabel = species + ', n$_i$ [10$^{20}$/m$^{3}$]'
			y = profiles[species]['ni']*1e-20
		elif what in ['ti','Ti','iontemperature']:
			ylabel = species + ", T$_{i}$ [keV]"
			y = profiles[species]['Ti']*1e-3
			
	   
		if what in ['all']:
			fig = plt.figure(figsize = (15,11))
			
			ax1 = fig.add_subplot(321, aspect = 'auto')
			ax1.set_ylabel('p$_{th}$ [kPa]')
			y = profiles['p']*1e-3
			ax1.set_xlim(0,x.max())
			ax1.get_xaxis().set_ticklabels([])
			if extended: 
				ax1.plot(x, y, style, color = 'k', lw = 2)
				ax1.plot(self.profiles['psiPres'], self.profiles['p']*1e-3, style, color = 'r', lw = 2)
			else: ax1.plot(self.profiles['psiPres'], y, '-', color = c, lw = 2)
			ax1.set_ylim(bottom=0)
			
			ax2 = fig.add_subplot(323, aspect = 'auto')
			ax2.set_ylabel('n$_e$ [10$^{20}$/m$^{3}$]')
			y = profiles['ne']*1e-20
			ax2.set_xlim(0,x.max())
			ax2.get_xaxis().set_ticklabels([])
			if extended: 
				ax2.plot(x, y, style, color = 'k', lw = 2)
				ax2.plot(self.profiles['psi'], self.profiles['ne']*1e-20, style, color = 'r', lw = 2)
			else: ax2.plot(x, y, '-', color = c, lw = 2)
			ax2.set_ylim(bottom=0)
			
			ax3 = fig.add_subplot(325, aspect = 'auto')
			ax3.set_ylabel("T$_{e}$ [keV]")
			y = profiles['Te']*1e-3
			ax3.set_xlim(0,x.max())
			ax3.set_xlabel('$\\psi$')
			if extended: 
				ax3.plot(x, y, style, color = 'k', lw = 2)
				ax3.plot(self.profiles['psi'], self.profiles['Te']*1e-3, style, color = 'r', lw = 2)
			else: ax3.plot(x, y, '-', color = c, lw = 2)
			ax3.set_ylim(bottom=0)
			
			ax4 = fig.add_subplot(322, aspect = 'auto')
			ax4.set_ylabel('V [m$^3$]')
			y = profiles['V']
			ax4.set_xlim(0,x.max())
			ax4.get_xaxis().set_ticklabels([])
			if extended: 
				ax4.plot(profiles['psi1'], y, style, color = 'k', lw = 2)
				ax4.plot(self.profiles['psi'], self.profiles['V'], style, color = 'r', lw = 2)
			else: ax4.plot(x, y, '-', color = c, lw = 2)
			ax4.set_ylim(bottom=0)
			
			ax5 = fig.add_subplot(324, aspect = 'auto')
			ax5.set_ylabel(species + ', n$_i$ [10$^{20}$/m$^{3}$]')
			y = profiles[species]['ni']*1e-20
			ax5.set_xlim(0,x.max())
			ax5.get_xaxis().set_ticklabels([])
			if extended: 
				ax5.plot(x, y, style, color = 'k', lw = 2)
				ax5.plot(self.profiles['psi'], self.profiles[species]['ni']*1e-20, style, color = 'r', lw = 2)
			else: ax5.plot(x, y, '-', color = c, lw = 2)
			ax5.set_ylim(bottom=0)
			
			ax6 = fig.add_subplot(326, aspect = 'auto')
			ax6.set_ylabel(species + ", T$_{i}$ [keV]")
			y = profiles[species]['Ti']*1e-3
			ax6.set_xlim(0,x.max())
			ax6.set_xlabel('$\\psi$')
			if extended: 
				ax6.plot(x, y, style, color = 'k', lw = 2)
				ax6.plot(self.profiles['psi'], self.profiles[species]['Ti']*1e-3, style, color = 'r', lw = 2)
			else: ax6.plot(x, y, '-', color = c, lw = 2)
			ax6.set_ylim(bottom=0)
		else:
			if fig is None: 
				fig = plt.figure()
				plt.xlim(0,x.max())
				plt.xlabel('$\\psi$')
				plt.ylabel(ylabel)
			else: 
				fig = plt.figure(fig)
			ax = fig.gca()
			ax.plot(x, y, style, color = c, lw = 2, label = label)
			plt.ylim(bottom=0)
			if len(label) > 0: plt.legend()
			
		fig.tight_layout()

	   
	def writeProfiles(self, keys = None, tag = None):
		"""
		Write all profiles to files
		keys = list of profiles, if keys is None: keys = ['ne','Te','ni','Ti']
		tag  = optional string to add to output file name
		"""
		if keys is None: keys = ['ne','Te','ni','Ti']
		else: keys = list(keys)
		psi = self.profiles['extend']['psi']
		ion = self.profiles['ions'][0]
		for key in keys:
			if 'i' in key: pro = self.profiles['extend'][ion][key]
			else: pro = self.profiles['extend'][key]
			if 'n' in key: norm = 1e20
			else: norm = 1e3
			expro.writeProfile(psi, pro, key, tag = tag, norm = norm)

	
	def writeGEQDSK(self, file, shot = None, time = None):
		"""
		writes a new gfile.	
		uses self.eqd:  dictionary containing all GEQDSK parameters
		User Input:
		file: name of new gfile
		shot: new shot number
		time: new shot timestep [ms]

		Note that this writes some data as 0 (ie rhovn, kvtor, etc.)
		"""

		if shot is None: shot = self.shot
		if time is None: time = self.time*1e3	# time is in ms in g-file
		
		KVTOR = 0
		RVTOR = 1.7
		NMASS = 0
		RHOVN = np.zeros((self.eqd['nw']))

		print('Writing to path: ' +file)
		with open(file, 'w') as f:
			f.write("{:<51}".format('  EFIT  xx/xx/xxxx  #' + str(shot) + '  ' + str(time) + 'ms'))	# this needs to be exactly 51 chars
			f.write('3 ' + str(self.eqd['nw']) + ' ' + str(self.eqd['nh']) + '\n')
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(self.eqd['rdim'], self.eqd['zdim'], self.eqd['rcentr'], self.eqd['Rmin'], self.eqd['Zmid']))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(self.eqd['rmaxis'], self.eqd['zmaxis'], self.eqd['siAxis'], self.eqd['siBry'], self.eqd['bcentr']))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(self.eqd['Ip'], 0, 0, 0, 0))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(0,0,0,0,0))
			self._write_array(self.eqd['fpol'], f)
			self._write_array(self.eqd['pres'], f)
			self._write_array(self.eqd['ffprime'], f)
			self._write_array(self.eqd['pprime'], f)
			self._write_array(self.eqd['psirz'].flatten(), f)
			self._write_array(self.eqd['qpsi'], f)
			f.write(str(len(self.eqd['lcfs'])).rjust(5) + str(len(self.eqd['wall'])).rjust(5) + '\n')	# these need to be 5 char long numbers with leading spaces
			self._write_array(self.eqd['lcfs'].flatten(), f)
			self._write_array(self.eqd['wall'].flatten(), f)
			f.write(str(KVTOR) + ' ' + format(RVTOR, ' .9E') + ' ' + str(NMASS) + '\n')
			self._write_array(RHOVN, f)

		print('Wrote new gfile')


	#=====================================================================
	#						private functions
	#=====================================================================
	# --- _write_array -----------------------
	# write numpy array in format used in g-file:
	# 5 columns, 9 digit float with exponents and no spaces in front of negative numbers
	def _write_array(self, x, f):
		N = len(x)
		rows = int(N/5)	 # integer division
		rest = N - 5*rows
		for i in range(rows):
			for j in range(5):
					f.write('% .9E' % (x[i*5 + j]))
			f.write('\n')
		if(rest > 0):
			for j in range(rest):
				f.write('% .9E' % (x[rows*5 + j]))
			f.write('\n')

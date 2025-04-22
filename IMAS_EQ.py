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


	def listTimeSLices(self):
		"""
		Show all available time slices in the JSON file.
		"""
		print('Available times are:')
		print(np.round(np.array(self.data['equilibrium']['time']),8))
		

	def getEQ(self, time, shot = None, psiMult = 1.0, BtMult = 1.0, IpMult = 1.0):
		"""
		Sets the time slice and assigns to the parameters we use in the equilParams_class ep object
		"""		   
		try:
			tIdx = np.where(np.round(np.array(self.data['equilibrium']['time']),8) == time)[0][0]
		except:
			print("Could not find timestep " + str(time) + " in JSON equilibrium dict.	Aborting.")
			self.listTimeSLices()
			return

		eqt = self.data['equilibrium']['time_slice'][tIdx]
		try: wall = self.data['wall']
		except: print('Wall not found in JSON. Using default.')
		if shot is None: self.shot = 1
		self.time = time	

		d = {}
		#ep object name left of '='
		d['R1D'] = eqt['profiles_2d'][0]['grid']['dim1']
		d['Z1D'] = eqt['profiles_2d'][0]['grid']['dim2']
		d['nw'] = len(d['R1D'])
		d['nh'] = len(d['Z1D']) 
		try:
			d['rcentr'] = self.data['equilibrium']['vacuum_toroidal_field']['r0']
			d['bcentr'] = self.data['equilibrium']['vacuum_toroidal_field']['b0'][tIdx] * BtMult
		except:
			print('Vacuum Toroidal Filed data is missing. Using a full field default')
			d['rcentr'] = 1.8495
			d['bcentr'] = 12.20329818869965
		d['rmaxis'] = eqt['global_quantities']['magnetic_axis']['r']
		d['zmaxis'] = eqt['global_quantities']['magnetic_axis']['z']
		d['Rmin'] = np.min(eqt['profiles_2d'][0]['grid']['dim1'])
		d['Rmax'] = np.max(eqt['profiles_2d'][0]['grid']['dim1'])
		d['Rlcfs'] = np.array(eqt['boundary']['outline']['r'])
		d['Zlcfs'] = np.array(eqt['boundary']['outline']['z'])
		d['Rbdry'] = np.max(d['Rlcfs'])
		d['Zmin'] = np.min(eqt['profiles_2d'][0]['grid']['dim2'])
		d['Zmax'] = np.max(eqt['profiles_2d'][0]['grid']['dim2'])
		d['Zlowest'] = np.min(d['Zlcfs'])
		d['siAxis'] = eqt['global_quantities']['psi_axis'] * psiMult
		d['siBry'] = eqt['global_quantities']['psi_boundary'] * psiMult

		# 1D profiles (if they arent nw long, interpolate them to be nw long)
		psiN = np.linspace(0,1,d['nw'])		# EFIT g-file standard assumes an equidistant psi with length nw
		psiN_profile = np.array(eqt['profiles_1d']['psi_norm'])	# This is the psi used for all 1D profiles. This can be different from psiN, typically for MEQ
		if psiN_profile[-1] != 1.0: 
			print('WARNING: psiN grid is not correctly normalized! Renormalizing...')
			psiAxis = psiN_profile[0]
			psiSep = psiN_profile[-1]
			psiN_profile = (psiN_profile - psiAxis)/(psiSep - psiAxis)	
		
		lenPsiN = len(psiN_profile)
		isMEQ = False
		if (lenPsiN != d['nw']) or ((psiN_profile[-1] - psiN_profile[-2]) != (psiN[-1] - psiN[-2])):	# true: not the same length or not the same grid size
			isMEQ = True
		
		d['fpol'] = np.array(eqt['profiles_1d']['f']) * BtMult
		d['ffprime'] = np.array(eqt['profiles_1d']['f_df_dpsi'])
		d['pprime'] = np.array(eqt['profiles_1d']['dpressure_dpsi'])
		d['pres'] = np.array(eqt['profiles_1d']['pressure'])
		d['qpsi'] = np.array(eqt['profiles_1d']['q'])
		if isMEQ: 
			d['fpol'] = np.interp(psiN, psiN_profile, d['fpol'])
			d['ffprime'] = np.interp(psiN, psiN_profile, d['ffprime'])
			d['pprime'] = np.interp(psiN, psiN_profile, d['pprime'])
			d['pres'] = np.interp(psiN, psiN_profile, d['pres'])
			d['qpsi'] = np.interp(psiN, psiN_profile, d['qpsi'])
			
		if d['pres'][-1] <= 0: 
			print('Problem: EFIT pressure at Separatrix is <= 0. This makes this EFIT unusable for M3DC1')
		
		#2D profiles
		d['psirz'] = np.array(eqt['profiles_2d'][0]['psi']).T * psiMult
		if d['psirz'].shape[0] != d['nh']: d['psirz'] = d['psirz'].T
		
		d['lcfs'] = np.vstack((d['Rlcfs'], d['Zlcfs'])).T
		try:
			d['Rwall'] = np.array(wall['description_2d'][0]['limiter']['unit'][0]['outline']['r'])
			d['Zwall'] = np.array(wall['description_2d'][0]['limiter']['unit'][0]['outline']['z'])
		except:
			d['Rwall'], d['Zwall'] = self._defaultWall()
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
						preservePoints = True, extendForM3DC1 = False, correctionMargin = None, correctionMarginCore = None, 
						usePressureFromEQDSK = True, doNotExtend = False):
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
	  	correctionMarginCore: same as above, but for the core. If this is used, a tanh transitions smoothly from this in the core to the one above in the edge
	  	usePressureFromEQDSK = Use pressure profile from the actual geqdsk in self.eqd instead of the pressure profile in data['core_profiles']
	  	doNotExtend = Skip the profile extension. This is for a quick check on the profiles, if the extension fails.
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
		
		
		if 'rho_pol_norm' in profiles['grid']: psiN = np.array(profiles['grid']['rho_pol_norm'])**2		# rho = sqrt(flux), so flux  = rho**2
		elif 'psi' in profiles['grid']:
			psiAxis = profiles['grid']['psi'][0]
			psiSep = profiles['grid']['psi'][-1]
			psiN = (np.array(profiles['grid']['psi']) - psiAxis)/(psiSep - psiAxis)	

		self.profiles = {'time':profiles['time'], 
			'ne':np.array(profiles['electrons']['density']), 'Te':np.array(profiles['electrons']['temperature']), 
			'p':np.array(profiles['pressure_thermal']), 'psi':psiN, 'psiPres':psiN}
		
		# make sure Te in eV and p in Pa
		if np.log10(self.profiles['Te'][0]) < 2: self.profiles['Te'] *= 1e3		# Te was in keV, now in eV
		if np.log10(self.profiles['p'][0]) < 4: self.profiles['p'] *= 1e3		# p was in kPa, now in Pa
		
		translate = {'volume':'V', 'rho_tor_norm':'rho'}
		for key in translate.keys():		
			if key in profiles['grid']: self.profiles[translate[key]] = np.array(profiles['grid'][key])
			else: self.profiles[translate[key]] = np.zeros(psiN.shape)

		ions = [profiles['ion'][item]['label'] for item in range(len(profiles['ion']))]
		self.profiles['ions'] = ions
		for i,ion in enumerate(ions):
			self.profiles[ion] = {'ni':np.array(profiles['ion'][i]['density'])}
			if 'temperature' in profiles['ion'][i]: 
				self.profiles[ion]['Ti'] = np.array(profiles['ion'][i]['temperature'])
				if np.log10(self.profiles[ion]['Ti'][0]) < 2: self.profiles[ion]['Ti'] *= 1e3		# Ti was in keV, now in eV

		if doNotExtend: return
				
		# Interpolate profiles to resolution dx, and extrapolate n,T profiles to psi = xmax
		if extendForM3DC1: preservePoints = False
		else: usePressureFromEQDSK = False
		self.profiles['extend'] = {}
		asymptote = [nsol,Tsol]	# this is already normalized
		norm = [1e20, 1e3, 1e3]	# normalize before profile fitting
		
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
					except: 
						y = np.zeros(self.profiles['extend']['psi'].shape)
						print('Extension failed for profile ' + key + 'for ion species: ' + ion)
					self.profiles['extend'][ion][key] = y*norm[i]
						
		# pressure needs special consideration: 
		# it needs to be splined for psi <= 1; do NOT use profile fit; it does not preserve original points, but makes a least square fit
		# on the other hand the extrapolation for psi > 1 needs to be monotonic and tanh asymptotic; regular cubic splines cannot do that
		# start with interpolate
		# Use a monotonic interpolation!!!! However, this is not as smooth as a regular spline. The first derivatives are guaranteed to be continuous, but the second derivatives may jump
		self.profiles['psiPres'] = self.profiles['psi']
		f = scinter.PchipInterpolator(self.profiles['psi'], self.profiles['p'])
		self.profiles['extend']['p'] = f(self.profiles['extend']['psi1'])
		
		if extendForM3DC1: self.correct_ne(asymptote = nsol*norm[0], correctionMargin = correctionMargin, correctionMarginCore = correctionMarginCore)
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
			tIdxeqd = np.where(np.round(np.array(self.data['equilibrium']['time']),8) == time)[0][0]
		except:
			print("Could not find timestep " + str(time) + " in JSON equilibrium dict.	Aborting.")
			return
			
		try: 
			if self.eqd is None: _ = self.getEQ(time)
		except: 
			print('getEQ failed!')
		
		profiles = self.data['core_profiles']['profiles_1d'][tIdx]
		eqt = self.data['equilibrium']['time_slice'][tIdxeqd]
		
		if 'rho_pol_norm' in profiles['grid']: psiN = np.array(profiles['grid']['rho_pol_norm'])**2		# rho = sqrt(flux), so flux  = rho**2
		elif 'psi' in profiles['grid']:
			psiAxis = profiles['grid']['psi'][0]
			psiSep = profiles['grid']['psi'][-1]
			psiN = (np.array(profiles['grid']['psi']) - psiAxis)/(psiSep - psiAxis)	
		else:
			print('psi is missing from core_profiles.profiles_1d.grid. Switching to rho')
			if 'rho_tor_norm' in profiles['grid']: psiN = np.array(profiles['grid']['rho_tor_norm'])
			else: 
				print('rho not found either')
				return
		if psiN[-1] != 1: print('Problem: Profiles psi grid is wrong')
		
		psieq, peq = np.array(eqt['profiles_1d']['psi_norm']), np.array(eqt['profiles_1d']['pressure'])
		if np.any(peq <= 0): print('Problem: EFIT pressure <= 0 somewhere')
		if psieq[-1] != 1: print('Problem: EFIT psi grid is wrong')

		try: 
			pth = np.array(profiles['pressure_thermal'])
			if np.any(pth <= 0): print('Problem: pth <= 0 somewhere')
		except:
			print('thermal pressure not found')
			pth = np.zeros(psiN.shape)
		ne = np.array(profiles['electrons']['density'])
		if np.any(ne <= 0): print('Problem: ne <= 0 somewhere')
		Te = np.array(profiles['electrons']['temperature'])
		if np.any(Te <= 0): print('Problem: Te <= 0 somewhere')
		
		foundIt = False
		ions = [profiles['ion'][item]['label'] for item in range(len(profiles['ion']))]
		for idx, ion in enumerate(ions):
			if ion in ['D', 'Deuterium', 'deuterium', 'd']: 
				foundIt = True
				break
		if not foundIt: 
			idx = 0
			ion = ions[idx]
			print('Cannot find the Deuterium ion species.')
		
		print('Using the ion species: ' + ion)
		ni = np.array(profiles['ion'][idx]['density'])
		if np.any(ni <= 0): print('Problem: ni <= 0 somewhere')
		Ti = np.array(profiles['ion'][idx]['temperature'])
		if np.any(Ti <= 0): print('Problem: Ti <= 0 somewhere')
		norm = [1e20, 1e3, 1e3]	# normalize n, T, p
		if np.log10(Te[0]) < 2: norm[1] = 1		# already normalized
		if np.log10(pth[0]) < 2: norm[2] = 1	# already normalized
		
		if m3dc1:
			psum = ne*(Te + Ti)*e
		else:
			psum = ne*Te*e + ni*Ti*e
		
		rawData = {'psiN':psiN, 'pth':pth, 'ne':ne, 'Te':Te, 'ni':ni, 'Ti':Ti, 'psiNeqd':psieq, 'pres':peq, 'ion':ion, 'psum':psum}
		
		plt.figure()
		plt.plot(psieq, peq*1e-3, 'k-', lw = 2, label = 'geqdsk')
		plt.plot(psiN, pth/norm[2], 'b-', lw = 2, label = 'thermal')
		plt.plot(psiN, psum/norm[1], 'g-', lw = 2, label = 'ne*Te + ni*Ti')
		plt.xlabel('$\\psi$')
		plt.ylabel('pressure [kPa]')
		plt.xlim(0,1)
		plt.legend()
		
		return rawData
		

	def correct_ne(self, asymptote = 0, correctionMargin = None, correctionMarginCore = None):
		"""		
		Correct ne, so that p - sum(n*T) >= 0 for all points in p = self.data['equilibrium']['time_slice'][tIdx]['profiles_1d']['pressure']
		This ignores the original ni profile and assumes ni = ne, as required in M3D-C1, also ignore any impurities
		Then extrapolate ne using exponential decay and interpolate ne on extended psi grid
		"""
		tIdx = np.where(np.round(np.array(self.data['equilibrium']['time']),8) == self.time)[0][0]
		eqt = self.data['equilibrium']['time_slice'][tIdx]
		psi0 = np.array(eqt['profiles_1d']['psi_norm'])
		p0 = np.array(eqt['profiles_1d']['pressure'])
		if p0[-1] <= 0:
			print('Problem: EFIT pressure <= 0 at separatrix. This is fatal. Aborting ne correction!')
			return
		
		fne = scinter.UnivariateSpline(self.profiles['psi'],self.profiles['ne'],s = 0)	# remap ne if usePressureFromEQDSK, else this does nothing
		
		foundIt = False
		for idx in range(len(self.profiles['ions'])):
			ion = self.profiles['ions'][idx]
			if ion in ['D', 'Deuterium', 'deuterium', 'd']: 
				foundIt = True
				break
		if not foundIt: 
			ion = self.profiles['ions'][0]
			print('Cannot find the Deuterium ion species. Using the ion species: ' + ion)
		
		psiEx, ne, Te, Ti = self.profiles['extend']['psi'], fne(psi0), self.profiles['extend']['Te'], self.profiles['extend'][ion]['Ti']
		neNew = expro.correct_ne(psi0, p0, ne, psiEx, Te, Ti, asymptote = asymptote, correctionMargin = correctionMargin, correctionMarginCore = correctionMarginCore)
		self.profiles['extend']['ne'] = neNew
		self.profiles['extend'][ion]['ni'] = self.profiles['extend']['ne'].copy()


	def extendPressure(self):
		"""
		Extend the pressure using sum(n*T) for psi > 1
		"""
		foundIt = False
		for idx in range(len(self.profiles['ions'])):
			ion = self.profiles['ions'][idx]
			if ion in ['D', 'Deuterium', 'deuterium', 'd']: 
				foundIt = True
				break
		if not foundIt: 
			ion = self.profiles['ions'][0]
			print('Cannot find the Deuterium ion species. Using the ion species: ' + ion)
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
		foundIt = False
		for idx in range(len(self.profiles['ions'])):
			ion = self.profiles['ions'][idx]
			if ion in ['D', 'Deuterium', 'deuterium', 'd']: 
				foundIt = True
				break
		if not foundIt: 
			ion = self.profiles['ions'][0]
			print('Cannot find the Deuterium ion species. Using the ion species: ' + ion)
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
		foundIt = False
		for idx in range(len(self.profiles['ions'])):
			ion = self.profiles['ions'][idx]
			if ion in ['D', 'Deuterium', 'deuterium', 'd']: 
				foundIt = True
				break
		if not foundIt: 
			ion = self.profiles['ions'][0]
			print('Cannot find the Deuterium ion species. Using the ion species: ' + ion)
		#ion0 = self.profiles['ions'][0]
		fti = scinter.UnivariateSpline(self.profiles['extend']['psi'], self.profiles['extend'][ion]['Ti'], s = 0)
		#if len(self.profiles['ions']) > 1:
		#	y = 0
		#	for i in range(1,len(self.profiles['ions'])): 
		#		ion = self.profiles['ions'][i]
		#		if 'Ti' in self.profiles['extend'][ion]:
		#			y += self.profiles['extend'][ion]['ni'] * self.profiles['extend'][ion]['Ti']*e
		#		else:
		#			y += self.profiles['extend'][ion]['ni'] * self.profiles['extend'][ion0]['Ti']*e
		#	fnT_imp = scinter.UnivariateSpline(self.profiles['extend']['psi'], y, s = 0)
		#else: fnT_imp = lambda x: 0
		fnT_imp = lambda x: 0
		
		# with original ni get d = p - sum(n*T), with original p, and extended/splined n & T (except ni)
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



	def interlacePressure(self, psiProf, pProf, psiEQD, pEQD):
		"""
		If the pressure profiles in EQD and Profiles have different knots, but 'visually' match, then interlace them to increase overall resolution
		This is not used anymore!
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


	def _defaultWall(self):
		Rwall = np.array([1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.269     , 1.269     , 1.269     , 1.269     ,
       1.269     , 1.26979217, 1.27703237, 1.28427258, 1.29151279,
       1.298753  , 1.30599321, 1.31323342, 1.32047362, 1.32771383,
       1.33495404, 1.34219425, 1.34943446, 1.35667466, 1.36391487,
       1.37115508, 1.37839529, 1.3856355 , 1.3928757 , 1.40011591,
       1.40735612, 1.41459633, 1.42183654, 1.4276647 , 1.4228986 ,
       1.40850639, 1.39302494, 1.37737755, 1.3618961 , 1.34641464,
       1.33093318, 1.31545173, 1.31301638, 1.33665351, 1.36165351,
       1.38665351, 1.41165351, 1.43665351, 1.46065351, 1.48565202,
       1.50929254, 1.5268016 , 1.54118836, 1.55557512, 1.56996187,
       1.58298011, 1.59737458, 1.61176904, 1.62616351, 1.64055797,
       1.65495244, 1.6693469 , 1.68374137, 1.69813583, 1.7125303 ,
       1.72      , 1.72      , 1.72      , 1.74202678, 1.76702678,
       1.79202678, 1.81702678, 1.84      , 1.84      , 1.84      ,
       1.84      , 1.84      , 1.84      , 1.84      , 1.84      ,
       1.83297322, 1.80797322, 1.78297322, 1.75797322, 1.73297322,
       1.70797322, 1.69236118, 1.68687588, 1.68139058, 1.67590527,
       1.67041997, 1.66493467, 1.65944937, 1.65318586, 1.65286881,
       1.66437071, 1.68435502, 1.70555681, 1.7267586 , 1.74796039,
       1.76916218, 1.79036397, 1.81156576, 1.83276754, 1.85227626,
       1.8709084 , 1.88954054, 1.90817269, 1.92680483, 1.94543697,
       1.96406911, 1.98270126, 2.0013334 , 2.01996554, 2.03853252,
       2.05689586, 2.0749678 , 2.09273421, 2.11018102, 2.12729599,
       2.14406372, 2.16047229, 2.17650835, 2.1921269 , 2.20733923,
       2.2221299 , 2.23648534, 2.25038761, 2.26380951, 2.27675173,
       2.28920455, 2.30115984, 2.31260987, 2.32354739, 2.33394206,
       2.34382398, 2.35317209, 2.3619823 , 2.3702487 , 2.37797486,
       2.38515876, 2.39180129, 2.39790283, 2.40346694, 2.4084921 ,
       2.41298431, 2.41694603, 2.42034607, 2.42324315, 2.42562473,
       2.42749269, 2.42884746, 2.42968947, 2.42997981, 2.42972985,
       2.4289689 , 2.42769551, 2.42590914, 2.42360947, 2.42079513,
       2.4174719 , 2.41358079, 2.4091596 , 2.40420622, 2.39871534,
       2.39268831, 2.38612075, 2.37901238, 2.37136248, 2.36317219,
       2.35443769, 2.34516542, 2.33535958, 2.32503534, 2.31416243,
       2.30277676, 2.29088546, 2.27849627, 2.26561731, 2.2522581 ,
       2.23842986, 2.22414531, 2.20941789, 2.19426106, 2.17868924,
       2.16274225, 2.14637947, 2.12965653, 2.11258574, 2.09518243,
       2.07745837, 2.05942758, 2.04110417, 2.02255688, 2.00392474,
       1.9852926 , 1.96666046, 1.94802831, 1.92939617, 1.91076403,
       1.89213188, 1.87349974, 1.8548676 , 1.83571627, 1.81451448,
       1.79331269, 1.7721109 , 1.75090911, 1.72970733, 1.70850554,
       1.68730375, 1.66680065, 1.65382808, 1.6524779 , 1.65868648,
       1.66417178, 1.66965708, 1.67514238, 1.68062768, 1.68611299,
       1.69159829, 1.70449624, 1.72949624, 1.75449624, 1.77949624,
       1.80449624, 1.82949624, 1.84      , 1.84      , 1.84      ,
       1.84      , 1.84      , 1.84      , 1.84      , 1.84      ,
       1.82050376, 1.79550376, 1.77050376, 1.74550376, 1.72050376,
       1.72      , 1.72      , 1.71453227, 1.7001378 , 1.68574334,
       1.67134887, 1.65695441, 1.64255994, 1.62816548, 1.61377101,
       1.59937654, 1.58498208, 1.57058761, 1.55757601, 1.54318926,
       1.5288025 , 1.51218765, 1.48910819, 1.46413049, 1.44013049,
       1.41513049, 1.39013049, 1.36513049, 1.34013049, 1.31592561,
       1.31329858, 1.32878004, 1.34426149, 1.35974295, 1.3752244 ,
       1.39087179, 1.40635325, 1.4212678 , 1.42768744, 1.4228435 ,
       1.41560329, 1.40836308, 1.40112287, 1.39388267, 1.38664246,
       1.37940225, 1.37216204, 1.36492183, 1.35768163, 1.35044142,
       1.34320121, 1.335961  , 1.32872079, 1.32148059, 1.31424038,
       1.30700017, 1.29975996, 1.29251975, 1.28527954, 1.27803934,
       1.27079913])
		Zwall = np.array([-0.5       , -0.475     , -0.45      , -0.425     , -0.4       ,
       -0.375     , -0.35      , -0.325     , -0.3       , -0.275     ,
       -0.25      , -0.225     , -0.2       , -0.175     , -0.15      ,
       -0.125     , -0.1       , -0.075     , -0.05      , -0.025     ,
        0.        ,  0.025     ,  0.05      ,  0.075     ,  0.1       ,
        0.125     ,  0.15      ,  0.175     ,  0.2       ,  0.225     ,
        0.25      ,  0.275     ,  0.3       ,  0.325     ,  0.35      ,
        0.375     ,  0.4       ,  0.425     ,  0.45      ,  0.475     ,
        0.5       ,  0.5191429 ,  0.54307154,  0.56700017,  0.5909288 ,
        0.61485743,  0.63878606,  0.66271469,  0.68664332,  0.71057195,
        0.73450058,  0.75842922,  0.78235785,  0.80628648,  0.83021511,
        0.85414374,  0.87807237,  0.902001  ,  0.92592963,  0.94985826,
        0.9737869 ,  0.99771553,  1.02164416,  1.04588324,  1.07015141,
        1.09051022,  1.1101399 ,  1.12836514,  1.14799483,  1.16762451,
        1.1872542 ,  1.20688388,  1.21457088,  1.21      ,  1.21      ,
        1.21      ,  1.21      ,  1.21      ,  1.209     ,  1.20908312,
        1.21634738,  1.2338722 ,  1.25431777,  1.27476334,  1.2952089 ,
        1.31543176,  1.3358719 ,  1.35631204,  1.37675218,  1.39719232,
        1.41763246,  1.4380726 ,  1.45851274,  1.47895288,  1.49939302,
        1.52202678,  1.54702678,  1.57202678,  1.575     ,  1.575     ,
        1.575     ,  1.575     ,  1.57297322,  1.54797322,  1.52297322,
        1.49797322,  1.47297322,  1.44797322,  1.42297322,  1.39797322,
        1.38      ,  1.38      ,  1.38      ,  1.38      ,  1.38      ,
        1.38      ,  1.36826628,  1.34387547,  1.31948467,  1.29509386,
        1.27070305,  1.24631225,  1.22192144,  1.19872603,  1.17397442,
        1.15208499,  1.13724819,  1.12400115,  1.11075411,  1.09750707,
        1.08426002,  1.07101298,  1.05776594,  1.04451889,  1.02901745,
        1.01234882,  0.99568019,  0.97901156,  0.96234293,  0.9456743 ,
        0.92900567,  0.91233703,  0.8956684 ,  0.87899977,  0.86225892,
        0.84529591,  0.82802282,  0.81043566,  0.79253144,  0.77430973,
        0.75576789,  0.73690731,  0.71772869,  0.69820807,  0.67836943,
        0.65821452,  0.63774728,  0.6169695 ,  0.59587835,  0.57448983,
        0.55281304,  0.53085819,  0.50863584,  0.48615687,  0.46342195,
        0.4404598 ,  0.41727517,  0.39388081,  0.37028881,  0.34651438,
        0.32257046,  0.2984707 ,  0.27422828,  0.24985682,  0.22536846,
        0.20077669,  0.17609382,  0.15132745,  0.12649766,  0.10161308,
        0.07668442,  0.05172216,  0.02673669,  0.00173837, -0.02325994,
       -0.0482473 , -0.07321336, -0.09814775, -0.12304003, -0.14787959,
       -0.17265684, -0.19735126, -0.22195616, -0.24645934, -0.27084756,
       -0.29510877, -0.31922921, -0.34319581, -0.36699505, -0.39061378,
       -0.41403671, -0.43725201, -0.46024705, -0.48301435, -0.50552474,
       -0.52778005, -0.54976937, -0.57148219, -0.5929083 , -0.61403852,
       -0.63486504, -0.65538146, -0.67558246, -0.69546335, -0.71502085,
       -0.73427356, -0.75317431, -0.77175684, -0.79002012, -0.80796673,
       -0.82559661, -0.84291268, -0.85991883, -0.87668151, -0.89335015,
       -0.91001878, -0.92668741, -0.94335604, -0.96002467, -0.9766933 ,
       -0.99336193, -1.01003056, -1.02669919, -1.04267651, -1.05592355,
       -1.06917059, -1.08241764, -1.09566468, -1.10891172, -1.12215876,
       -1.13540581, -1.14960715, -1.17063833, -1.19532279, -1.21852919,
       -1.24292   , -1.2673108 , -1.29170161, -1.31609242, -1.34048322,
       -1.36487403, -1.38      , -1.38      , -1.38      , -1.38      ,
       -1.38      , -1.38      , -1.39449624, -1.41949624, -1.44449624,
       -1.46949624, -1.49449624, -1.51949624, -1.54449624, -1.56949624,
       -1.575     , -1.575     , -1.575     , -1.575     , -1.575     ,
       -1.55050376, -1.52550376, -1.50223582, -1.48179568, -1.46135554,
       -1.4409154 , -1.42047526, -1.40003512, -1.37959498, -1.35915483,
       -1.33871469, -1.31827455, -1.29783441, -1.27760689, -1.25716132,
       -1.23671575, -1.21826045, -1.2094243 , -1.209     , -1.21      ,
       -1.21      , -1.21      , -1.21      , -1.21      , -1.21266674,
       -1.20961396, -1.18998427, -1.17035459, -1.1507249 , -1.13109522,
       -1.11286998, -1.0932403 , -1.07321546, -1.04935522, -1.02497213,
       -1.0010435 , -0.97711487, -0.95318624, -0.92925761, -0.90532897,
       -0.88140034, -0.85747171, -0.83354308, -0.80961445, -0.78568582,
       -0.76175719, -0.73782856, -0.71389993, -0.68997129, -0.66604266,
       -0.64211403, -0.6181854 , -0.59425677, -0.57032814, -0.54639951,
       -0.52247088])
		return Rwall,Zwall
# IMAS_EQ.py
# description:	reads a netcdf, hdf5, json, equilibrium file formatted per IMAS/OMAS
# engineer:		T Looby
# date:			20241030

import os
import numpy as np
import scipy.interpolate as scinter

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
		self.profiles = None
		
		with open(filename, 'r') as file:
			self.data = json.load(file) 
		
		return


	def getEQ(self, time, psiMult=1.0, BtMult=1.0, IpMult=1.0):
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

		d = {}
		#ep object name left of '='
		d['R1D'] = eqt['profiles_2d'][0]['grid']['dim1']
		d['Z1D'] = eqt['profiles_2d'][0]['grid']['dim2']
		d['nw'] = len(d['R1D'])
		d['nh'] = len(d['Z1D']) 
		d['rcentr'] = self.data['equilibrium']['vacuum_toroidal_field']['r0']
		d['bcentr'] = self.data['equilibrium']['vacuum_toroidal_field']['b0'][tIdx] * BtMult
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
		
		d['lcfs'] = np.vstack((d['Rlcfs'], d['Zlcfs'])).T
		d['Rwall'] = np.array(wall['description_2d'][0]['limiter']['unit'][0]['outline']['r'])
		d['Zwall'] = np.array(wall['description_2d'][0]['limiter']['unit'][0]['outline']['z'])
		d['wall'] = np.vstack((d['Rwall'], d['Zwall'])).T
		d['rdim'] = d['Rmax'] - d['Rmin']
		d['zdim'] = d['Zmax'] - d['Zmin']
		d['R0'] = eqt['global_quantities']['magnetic_axis']['r']
		d['R1'] = d['Rmin']
		d['Zmid'] = 0.0
		d['Ip'] = eqt['global_quantities']['ip'] * IpMult
		d['thetapnts'] = 2*d['nw']
		d['Rsminor'] = np.linspace(d['rmaxis'], d['Rbdry'], d['nw'])
		self.eqd = d
		return d
	
	
	def coreProfiles(self, time, dx = 0.005, xmin = 0.7, xmax = 1.2, nsol = 0.02, Tsol = 1e-4):
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
		""" 
		from . import extend_profiles as expro

		Ntimes = len(self.data['core_profiles']['profiles_1d'])
		times = [self.data['core_profiles']['profiles_1d'][i]['time'] for i in range(Ntimes)]	   
		try:
			tIdx = np.where(np.round(np.array(times),8) == time)[0][0]
		except:
			print("Could not find timestep " + str(time) + " in JSON equilibrium dict.	Aborting.")
			return
		
		profiles = self.data['core_profiles']['profiles_1d'][tIdx]
		psiAxis = profiles['grid']['psi'][0]
		psiSep = profiles['grid']['psi'][-1]
		psiN = (np.array(profiles['grid']['psi']) - psiAxis)/(psiSep - psiAxis)
		d = {'time':profiles['time'], 
			'ne':np.array(profiles['electrons']['density']), 'Te':np.array(profiles['electrons']['temperature']), 
			'p':np.array(profiles['pressure_thermal']), 'V':np.array(profiles['grid']['volume']), 
			'rho':np.array(profiles['grid']['rho_tor_norm']), 'psi':psiN} 
			
		ions = [profiles['ion'][item]['label'] for item in range(len(profiles['ion']))]
		d['ions'] = ions
		for i,ion in enumerate(ions):
			d[ion] = {'ni':np.array(profiles['ion'][i]['density']), 'Ti':np.array(profiles['ion'][i]['temperature'])}
		
		
		# Interpolate profiles to resolution dx, and extrapolate n,T profiles to psi = xmax
		#import warnings
		#warnings.filterwarnings("ignore")

		d['extend'] = {}
		asymptote = [nsol,Tsol]	# this is already normalized
		norm = [1e20, 1e3]	# normalize before profile fitting
		if len(d['psi']) < 100: 
			N = 101
			upscale = True
		else: upscale = False
		
		for i,key in enumerate(['ne','Te']):
			#print(key)
			if upscale:
				f = scinter.PchipInterpolator(d['psi'], d[key]/norm[i])		# this gets overwritten by a smooth profile fit
				x = np.linspace(0,1,N)
				y = f(x)
			else: x,y = d['psi'], d[key]/norm[i]
			x,y = expro.make_profile(x,y, key, asymptote = asymptote[i], show = False, xmin = xmin, dx = dx)
			d['extend'][key] = y*norm[i]
		d['extend']['psi'] = x
				
		f = scinter.UnivariateSpline(d['psi'], d['V'], s = 0)
		d['extend']['Vpsi'] = d['extend']['psi'][d['extend']['psi'] <= 1]
		d['extend']['V'] = f(d['extend']['Vpsi'])
				
		for ion in d['ions']:
			d['extend'][ion] = {}
			for i,key in enumerate(['ni','Ti']):
				#print(key)
				if upscale:
					f = scinter.PchipInterpolator(d['psi'], d[ion][key]/norm[i])	# this gets overwritten by a smooth profile fit
					x = np.linspace(0,1,N)
					y = f(x)
				else: x,y = d['psi'], d[ion][key]/norm[i]
				x,y = expro.make_profile(x,y, key, asymptote = asymptote[i], show = False, xmin = xmin, dx = dx)
				d['extend'][ion][key] = y*norm[i]
				
		# pressure needs special consideration: 
		# it needs to be splined for psi <= 1; do NOT use profile fit; it does not preserve original points, but makes a least square fit
		# on the other hand the extrapolation for psi > 1 needs to be monotonic and tanh asymptotic; regular cubic splines cannot do that
		# start with interpolate
		# Use a monotonic interpolation!!!! However, this is not as smooth as a regular spline. The first derivatives are guaranteed to be continuous, but the second derivatives may jump
		f = scinter.PchipInterpolator(d['psi'], d['p'])
		d['extend']['p'] = f(d['extend']['Vpsi'])
		
		self.profiles = d
		self.correct_ni()
		self.extendPressure()
		return


	def extendPressure(self):
		"""
		Extend the pressure using sum(n*T) for psi > 1
		"""
		e = 1.60217663e-19
		p = self.profiles['extend']['ne'] * self.profiles['extend']['Te']*e
		for ion in self.profiles['ions']:
			p += self.profiles['extend'][ion]['ni'] * self.profiles['extend'][ion]['Ti']*e

		xmax = self.profiles['extend']['psi'].max()
		idx = np.where(self.profiles['extend']['psi'] > xmax - 0.08)[0]
		
		psi_new = np.append(self.profiles['extend']['Vpsi'], self.profiles['extend']['psi'][idx])
		p_new = np.append(self.profiles['extend']['p'],p[idx])
		f = scinter.PchipInterpolator(psi_new,p_new)
		p_ex = f(self.profiles['extend']['psi'])
		self.profiles['extend']['p'] = p_ex
		self.checkExtension()
		return #p,psi_new,p_new
		

	def correct_ni(self):
		"""		
		Correct main ion ni, so that p - sum(n*T) >= 0 for all points in p = self.profiles['p']
		"""
		e = 1.60217663e-19
		
		# Spline ne,te and ti for the original psi
		fne = scinter.UnivariateSpline(self.profiles['extend']['psi'], self.profiles['extend']['ne'], s = 0)
		fte = scinter.UnivariateSpline(self.profiles['extend']['psi'], self.profiles['extend']['Te'], s = 0)
		ion = self.profiles['ions'][0]
		fti = scinter.UnivariateSpline(self.profiles['extend']['psi'], self.profiles['extend'][ion]['Ti'], s = 0)
		if len(self.profiles['ions']) > 1:
			y = 0
			for i in range(1,len(self.profiles['ions'])): 
				ion = self.profiles['ions'][i]
				y += self.profiles['extend'][ion]['ni'] * self.profiles['extend'][ion]['Ti']*e
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
		
		# where d < 0 replace ni with new value so that  p - sum(n*T) >= 0 -> ni_patched
		ni0 = (self.profiles['p'] - netex - impx) / tix		# this ni would make d = 0 everywhere
		ni_patched = self.profiles[ion]['ni'].copy()
		ni_patched[idx] = ni0[idx]*0.999	# give it a tiny margin
		
		# append ni_patched and ni_extended(psi > 1.18) to keep the asymptotic value
		xmax = self.profiles['extend']['psi'].max()
		idx = np.where(self.profiles['extend']['psi'] > (xmax - 0.02))[0]
		psi_patched = np.append(self.profiles['psi'], self.profiles['extend']['psi'][idx])
		ni_patched = np.append(ni_patched, self.profiles['extend'][ion]['ni'][idx])
		
		# Use a monotonic interpolation for ni_patched -> new ni
		# !!!!!!! This interpolator does not overshoot like UnivariateSpline, but instead maintains a monotonic curve !!!!!!!!!!!!
		# However, this is not as smooth as a regular spline. The first derivatives are guaranteed to be continuous, but the second derivatives may jump
		f = scinter.PchipInterpolator(psi_patched, ni_patched)
		ni_new = f(self.profiles['extend']['psi'])
		fni = scinter.UnivariateSpline(self.profiles['extend']['psi'], ni_new, s = 0)
		
		# verify & update
		p2 = netex + fni(x) * tix + impx 
		d2 = self.profiles['p'] - p2
		if any(d2 < 0): 
			print('Sum of n*T exceeds thermal pressure inside the separatrix. Check extended profiles.')
		else: 
			print('ni correction okay')
			self.profiles['extend'][ion]['ni'] = ni_new
		return #x,p,d,ni0,psi_patched,ni_patched,ni_new,p2,d2
		
		
	def checkExtension(self):
		"""
		Verify that p - sum(n*T) >= 0
		"""
		e = 1.60217663e-19
		p = self.profiles['extend']['ne'] * self.profiles['extend']['Te']*e
		for ion in self.profiles['ions']:
			p += self.profiles['extend'][ion]['ni'] * self.profiles['extend'][ion]['Ti']*e

		#idx = np.where(self.profiles['extend']['psi'] <= 1)[0]
		d = self.profiles['extend']['p'] - p
		
		if any(d < 0): print('Sum of n*T exceeds thermal pressure inside the separatrix. Check extended profiles.')
		else: print('Extension okay')
		return #p,d
		
	
	def plotProfile(self, what = 'all', fig = None, c = None, label = '', extended = False):
		"""
		what: keyword of what profile to plot. default is 'all' and plots all 6 relevant profiles
		fig: integer number of figure window to use, e.g. 1
		c: string of color code, e.g. 'k' or 'r'
		label: string that becomes the label for the plot in the legend
		extended: plot the extended profiles
		"""
		import matplotlib.pyplot as plt
		if c is None: c = 'k'
		
		if extended: profiles = self.profiles['extend']
		else: profiles = self.profiles
		x = profiles['psi']
		species = self.profiles['ions'][0]	# just D for now
		
		if what in ['p','P','Pres','pres','pressure','Ptot','ptot','Press','press','Pressure']:
			ylabel = 'p$_{th}$ [kPa]'
			y = profiles['p']*1e-3
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
			ylabel = species + ',  n$_i$ [10$^{20}$/m$^{3}$]'
			y = profiles[species]['ni']*1e-20
		elif what in ['ti','Ti','iontemperature']:
			ylabel = species + ",  T$_{i}$ [keV]"
			y = profiles[species]['Ti']*1e-3
			
	   
		if what in ['all']:
			fig = plt.figure(figsize = (15,11))
			
			ax1 = fig.add_subplot(321, aspect = 'auto')
			ax1.set_ylabel('p$_{th}$ [kPa]')
			y = profiles['p']*1e-3
			ax1.set_xlim(0,x.max())
			ax1.get_xaxis().set_ticklabels([])
			if extended: 
				ax1.plot(x, y, 'k-', lw = 2)
				ax1.plot(self.profiles['psi'], self.profiles['p']*1e-3, 'r-', lw = 2)
			else: ax1.plot(x, y, '-', color = c, lw = 2)
			ax1.set_ylim(bottom=0)
			
			ax2 = fig.add_subplot(323, aspect = 'auto')
			ax2.set_ylabel('n$_e$ [10$^{20}$/m$^{3}$]')
			y = profiles['ne']*1e-20
			ax2.set_xlim(0,x.max())
			ax2.get_xaxis().set_ticklabels([])
			if extended: 
				ax2.plot(x, y, 'k-', lw = 2)
				ax2.plot(self.profiles['psi'], self.profiles['ne']*1e-20, 'r-', lw = 2)
			else: ax2.plot(x, y, '-', color = c, lw = 2)
			ax2.set_ylim(bottom=0)
			
			ax3 = fig.add_subplot(325, aspect = 'auto')
			ax3.set_ylabel("T$_{e}$ [keV]")
			y = profiles['Te']*1e-3
			ax3.set_xlim(0,x.max())
			ax3.set_xlabel('$\\psi$')
			if extended: 
				ax3.plot(x, y, 'k-', lw = 2)
				ax3.plot(self.profiles['psi'], self.profiles['Te']*1e-3, 'r-', lw = 2)
			else: ax3.plot(x, y, '-', color = c, lw = 2)
			ax3.set_ylim(bottom=0)
			
			ax4 = fig.add_subplot(322, aspect = 'auto')
			ax4.set_ylabel('V [m$^3$]')
			y = profiles['V']
			ax4.set_xlim(0,x.max())
			ax4.get_xaxis().set_ticklabels([])
			if extended: 
				ax4.plot(profiles['Vpsi'], y, '-', lw = 2)
				ax4.plot(self.profiles['psi'], self.profiles['V'], 'r-', lw = 2)
			else: ax4.plot(x, y, '-', color = c, lw = 2)
			ax4.set_ylim(bottom=0)
			
			ax5 = fig.add_subplot(324, aspect = 'auto')
			ax5.set_ylabel(species + ',	 n$_i$ [10$^{20}$/m$^{3}$]')
			y = profiles[species]['ni']*1e-20
			ax5.set_xlim(0,x.max())
			ax5.get_xaxis().set_ticklabels([])
			if extended: 
				ax5.plot(x, y, 'k-', lw = 2)
				ax5.plot(self.profiles['psi'], self.profiles[species]['ni']*1e-20, 'r-', lw = 2)
			else: ax5.plot(x, y, '-', color = c, lw = 2)
			ax5.set_ylim(bottom=0)
			
			ax6 = fig.add_subplot(326, aspect = 'auto')
			ax6.set_ylabel(species + ",	 T$_{i}$ [keV]")
			y = profiles[species]['Ti']*1e-3
			ax6.set_xlim(0,x.max())
			ax6.set_xlabel('$\\psi$')
			if extended: 
				ax6.plot(x, y, 'k-', lw = 2)
				ax6.plot(self.profiles['psi'], self.profiles[species]['Ti']*1e-3, 'r-', lw = 2)
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
			ax.plot(x, y, '-', color = c, lw = 2, label = label)
			plt.ylim(bottom=0)
			if len(label) > 0: plt.legend()
			
		fig.tight_layout()
	   
	
	def writeGEQDSK(self, file, g, shot=None, time=None, ep=None):
		"""
		writes a new gfile.	 user must supply
		file: name of new gfile
		g:	  dictionary containing all GEQDSK parameters
		shot: new shot number
		time: new shot timestep [ms]

		Note that this writes some data as 0 (ie rhovn, kvtor, etc.)
		"""

		if shot==None:
			shot=1
		if time==None:
			time=1
		
		KVTOR = 0
		RVTOR = 1.7
		NMASS = 0
		RHOVN = np.zeros((g['NR']))

		print('Writing to path: ' +file)
		with open(file, 'w') as f:
			f.write('  EFIT	   xx/xx/xxxx	 #' + str(shot) + '	 ' + str(time) + 'ms		')
			f.write('	3 ' + str(g['NR']) + ' ' + str(g['NZ']) + '\n')
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(g['Xdim'], g['Zdim'], g['R0'], g['R1'], g['Zmid']))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(g['RmAxis'], g['ZmAxis'], g['psiAxis'], g['psiSep'], g['Bt0']))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(g['Ip'], 0, 0, 0, 0))
			f.write('% .9E% .9E% .9E% .9E% .9E\n'%(0,0,0,0,0))
			self._write_array(g['Fpol'], f)
			self._write_array(g['Pres'], f)
			self._write_array(g['FFprime'], f)
			self._write_array(g['Pprime'], f)
			self._write_array(g['psiRZ'].flatten(), f)
			self._write_array(g['qpsi'], f)
			f.write(str(g['Nlcfs']) + ' ' + str(g['Nwall']) + '\n')
			self._write_array(g['lcfs'].flatten(), f)
			self._write_array(g['wall'].flatten(), f)
			f.write(str(KVTOR) + ' ' + format(RVTOR, ' .9E') + ' ' + str(NMASS) + '\n')
			self._write_array(RHOVN, f)

		print('Wrote new gfile')


	def writeProfiles(self, keys = None):
		if keys is None: keys = ['ne','Te','ni','Ti']
		else: keys = list(keys)
		
		psi = self.profiles['extend']['psi']
		ion = self.profiles['ions'][0]
		for key in keys:
			if 'i' in key: pro = self.profiles['extend'][ion][key]
			else: pro = self.profiles['extend'][key]
			if 'n' in key: norm = 1e20
			else: norm = 1e3
			with open('profile_' + str.lower(key),'w') as f:
				#f.write('# ' + key + ' profile in ' + units + ' \n')
				#f.write('# psi          ' + key + ' [' + units + '] \n')
				for i in range(len(psi)):
					f.write(format(psi[i],' 13.7e') + ' \t' + format(pro[i]/norm,' 13.7e') + '\n')




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

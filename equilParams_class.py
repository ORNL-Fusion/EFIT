"""
Interface to read EFIT g-file equilibria.
Author: A. Wingen
First Implemented: Sep. 10. 2012
Please report bugs to: wingen@fusion.gat.com
Python 3 version
"""
_VERSION = 5.3
_LAST_UPDATE = 'June 10. 2025'

import os
import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp
import warnings

from . import geqdsk as gdsk	# this is a relative import
from . import equilibrium as eq

try: 
	from . import IMAS_EQ	# this is a relative import
except Exception as e:
	print(e)	# if not found skip this import; then only geqdsk functionality available

class equilParams:

		def __init__(self, filename, EQmode=None, 
					 nw=0, nh=0, thetapnts=0, grid2G=True,
					 tree=None, server='atlas.gat.com', time=10.4,
					 psiMult=1.0, BtMult=1.0, IpMult=1.0):
			"""
			Open EQ file, read it, and provide dictionary self.g as well as 1D and 2D interpolation functions.

			if EQmode == 'netcdf', uses a netCDF reader, 
			if EQmode == 'json', uses a json reader, 
			else uses GEQDSK reader

			filename is name of EQ file (GEQDSK, JSON, HDF5, NETCDF)
			EQmode is string that identifies file type
			nw is grid radial dimension
			nh is grid vertical dimension
			thetapnts is number of points in poloidal angle
			grid2G is 
			tree is string of tree if using MDS+
			server is MDS+ server
			time is timestep of this equilibrium
			psiMult is multiplier for psi quantities (ie psiRZ, psiSep, psiAxis)
			BtMult is multiplier for Bt quantities (Bt, Fpol)
			IpMult is multipler for Ip 

			"""

			#read GEQDSKs text file or IMAS formatted file (netCDF, HDF5, JSON)
			GEQDSKflag = False

			#if EQmode was not provided, determine
			if EQmode==None:
				extension = os.path.splitext(filename)[1]
				if extension == '.nc':
					EQmode = 'netcdf'
				elif extension == '.json':
					EQmode = 'json'
				else:
					EQmode = 'geqdsk'

			self.data = None
			if EQmode == 'netcdf':
				print("NetCDF EQmode")
				imas_nc = IMAS_EQ.netCDF_IMAS()
				self.data = imas_nc.readNetCDF(filename, time)
			elif EQmode == 'json':
				print("JSON EQmode")
				try:
					imas_js = IMAS_EQ.JSON_IMAS(filename)
					self.data = imas_js.getEQ(time)
				except Exception as e:
					print("Could not load JSON EQ.  Error:")
					print(e)
			else:
				#print("GEQDSK EQmode")
				GEQDSKflag = True
				self.readGfile(filename, tree, server, nw=0, nh=0, thetapnts=0, grid2G=True)
			
			if self.data is None: return

			if GEQDSKflag == False:
				self.shot = 1 #arbitrary for now
				self.time = time *1e3		# time is in ms 
				for key, value in self.data.items():
					setattr(self, key, value)

			# ---- normalize variables per arguments ----
			self.siAxis *= psiMult
			self.siBry *= psiMult
			self.bcentr *= BtMult
			self.Ip *= IpMult

			# ---- default Functions ----
			self.PROFdict = self.profiles(BtMult)
			self.RZdict = self.RZ_params()
			self.PSIdict = self.getPsi(psiMult)
			self.PHIdict = self.getTorPsi()

			# ---- more Variables ----
			self.dpsidZ, self.dpsidR = np.gradient(self.PSIdict['psi2D'], self.RZdict['dZ'],
												   self.RZdict['dR'])
			self.B_R = self.dpsidZ / self.RZdict['Rs2D']
			self.B_Z = -self.dpsidR / self.RZdict['Rs2D']
			self.Bp_2D = np.sqrt(self.B_R**2 + self.B_Z**2)
			self.theta = np.linspace(0.0, 2.*np.pi, self.thetapnts)

			psiN2D = self.PSIdict['psiN_2D'].flatten()
			idx = np.where(psiN2D <= 1)[0]
			Fpol_sep = self.PROFdict['ffunc'](1.0)	 #self.bcentr * abs(self.R0)
			Fpol2D = np.ones(psiN2D.shape) * Fpol_sep
			Fpol2D[idx] = self.PROFdict['ffunc'](psiN2D[idx])
			Fpol2D = Fpol2D.reshape(self.PSIdict['psiN_2D'].shape)
			self.Bt_2D = Fpol2D / self.RZdict['Rs2D']

			# ---- more Functions ----
			self.psiFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'],
													  self.PSIdict['psiN_2D'].T)
			self.BpFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'],
													 self.Bp_2D.T)
			self.BtFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'],
													 self.Bt_2D.T)
			self.BRFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'],
													 self.B_R.T)
			self.BZFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'],
													 self.B_Z.T)

			shape = self.get_FluxShape(1.0) # {'kappa', 'tri_avg', 'triUP', 'triLO', 'aminor', 'aspect', 'radius'}

			# ---- g dict ----
			self.g = {'shot': self.shot, 'time': self.time, 'NR': self.nw, 'NZ': self.nh,
					  'Xdim': self.rdim, 'Zdim': self.zdim,
					  'R0': self.R0, 'R1': self.R1,
					  'Zmid': self.Zmid, 'RmAxis': self.rmaxis, 'ZmAxis': self.zmaxis,
					  'psiAxis': self.siAxis, 'psiSep': self.siBry, 'Bt0': self.bcentr,
					  'Ip': self.Ip, 'Fpol': self.PROFdict['fpol'],
					  'Pres': self.PROFdict['pres'], 'FFprime': self.PROFdict['ffprime'],
					  'Pprime': self.PROFdict['pprime'], 'qpsi': self.PROFdict['q_prof'],
					  'q': self.PROFdict['q_prof'], 'psiRZ': self.PSIdict['psi2D'],
					  'R': self.RZdict['Rs1D'], 'Z': self.RZdict['Zs1D'], 'dR': self.RZdict['dR'],
					  'dZ': self.RZdict['dZ'], 'psiRZn': self.PSIdict['psiN_2D'],
					  'Nlcfs': len(self.lcfs), 'Nwall': len(self.wall),
					  'lcfs': self.lcfs, 'wall':self.wall, 
					  'psi': self.PSIdict['psi1D'], 'psiN': self.PSIdict['psiN1D'],'Rsminor': self.Rsminor,
					  'kappa': shape['kappa'], 'triUP': shape['triUP'], 'triLO': shape['triLO'], 
					  'aminor': shape['aminor'], 'aspect': shape['aspect']}
					  
			self.Swall, self.Swall_max = self.length_along_wall()
			self.isHelicity = self.helicity()
			self.g['helicity'] = self.isHelicity
			self.FluxSurfList = None
			return

		# --------------------------------------------------------------------------------
		def readGfile(self, gfileNam, tree=None, server='atlas.gat.com', 
					  nw=0, nh=0, thetapnts=0, grid2G=True):
			"""
			reads a geqdsk file into self.data
			"""
			# ---- get path, shot number and time from file name ----
			# returns location of last '/' in gfileNam or -1 if not found
			idx = gfileNam[::-1].find('/')
			if(idx == -1):
				gpath = '.'
				gfile = gfileNam
			else:
				idx *= -1
				gpath = gfileNam[0:idx - 1]	 # path without a final '/'
				gfile = gfileNam[idx::]
			
			if '_' in gfile:
				gfileA,gfileB = gfile.split('_')
				if '.' in gfileA:
					idx = gfileA.find('.')
					shot, time = gfileA[1:idx], gfileA[idx + 1::]
				else:
					shot, time = gfileA[1::], gfileB
				fmtstr = '0' + str(len(shot)) + 'd'
			elif '.' in gfile:
				idx = gfile.find('.')
				shot, time = gfile[1:idx], gfile[idx + 1::]
				fmtstr = '0' + str(len(shot)) + 'd'
			else:
				shot, time, fmtstr = 0, 0, '06d'
			try: shot = int(shot)
			except: shot, fmtstr = 0, '06d'
			try: time = float(time)
			except: time = 0

			self.shot = shot
			self.time = time

			if (not os.path.isfile(gfileNam)) and (tree is None):
				raise NameError('g-file not found -> Abort!')

			# ---- read from MDS+, if keyword tree is given ----
			if not (tree is None):
				# time needs not be exact, so time could change
				time = self._read_mds(shot, time, tree=tree, gpath=gpath, Server=server)
				# adjust filename to match time
				gfileNam = gpath + '/g' + format(shot, fmtstr) + '.' + format(time, '05d')

			# ---- open & read g-file ----
			self.data = gdsk.Geqdsk()
			self.data.openFile(gfileNam)

			# ---- Variables ----
			if grid2G:
				self.nw = self.data.get('nw')
				self.nh = self.data.get('nh')
				self.thetapnts = 2 * self.nh
			else:
				self.nw = nw
				self.nh = nh
				self.thetapnts = thetapnts
			self.bcentr = self.data.get('bcentr')
			self.rmaxis = self.data.get('rmaxis')
			self.zmaxis = self.data.get('zmaxis')
			self.Rmin = self.data.get('rleft')
			self.Rmax = self.Rmin + self.data.get('rdim')
			self.Rbdry = self.data.get('rbbbs').max()
			self.Rsminor = np.linspace(self.rmaxis, self.Rbdry, self.nw)
			self.Zmin = self.data.get('zmid') - self.data.get('zdim')/2.0
			self.Zmax = self.data.get('zmid') + self.data.get('zdim')/2.0
			self.Zlowest = self.data.get('zbbbs').min()
			self.siAxis = self.data.get('simag')
			self.siBry = self.data.get('sibry')
			
			self.lcfs = np.vstack((self.data.get('rbbbs'), self.data.get('zbbbs'))).T
			self.wall = np.vstack((self.data.get('rlim'), self.data.get('zlim'))).T
			self.rdim = self.data.get('rdim')
			self.zdim = self.data.get('zdim')
			self.R0 = abs(self.data.get('rcentr'))
			self.R1 = self.data.get('rleft')
			self.Zmid = self.data.get('zmid')
			self.Ip = self.data.get('current')

			return

		# --------------------------------------------------------------------------------
		def writeGEQDSK(self, file, shot = None):
			"""
			writes a new gfile.	 user must supply
			file: name of new gfile
			uses self.g:	  dictionary containing all GEQDSK parameters
	
			Note that this writes some data as 0 (ie rhovn, kvtor, etc.)
			"""
			KVTOR = 0
			RVTOR = 1.7
			NMASS = 0
			RHOVN = np.zeros(self.g['NR'])
			if shot is None: shot = self.g['shot']
	
			print('Writing to path: ' + file)
			with open(file, 'w') as f:
				# Char counting of line: "         10        20        30        40        50        60
				# Char counting of line: "123456789012345678901234567890123456789012345678901234567890
				# example of first line: "  EFITD    03/13/2012    #148712  4101ms           3 129 129"
				# From Fortran: 2000 format (6a8,3i4)
				# From Fortran: write (negdsk,2000) (case(i),i=1,6),idum,mw,mh
				f.write("{:<48}".format('  EFITD    xx/xx/xxxx    #' + str(shot).zfill(6) + '  ' + str(int(self.g['time']))))	# this needs to be exactly 48 chars
				f.write(str(3).rjust(4,' ') + str(self.g['NR']).rjust(4,' ') + str(self.g['NZ']).rjust(4,' ') + '\n')			# each of the numbers needs to have 4 chars, padded with blanks
				f.write('% .9E% .9E% .9E% .9E% .9E\n'%(self.g['Xdim'], self.g['Zdim'], self.g['R0'], self.g['R1'], self.g['Zmid']))
				f.write('% .9E% .9E% .9E% .9E% .9E\n'%(self.g['RmAxis'], self.g['ZmAxis'], self.g['psiAxis'], self.g['psiSep'], self.g['Bt0']))
				f.write('% .9E% .9E% .9E% .9E% .9E\n'%(self.g['Ip'], self.g['psiAxis'], 0, self.g['RmAxis'], 0))
				f.write('% .9E% .9E% .9E% .9E% .9E\n'%(self.g['ZmAxis'],0,self.g['psiSep'],0,0))
				self._write_array(self.g['Fpol'], f)
				self._write_array(self.g['Pres'], f)
				self._write_array(self.g['FFprime'], f)
				self._write_array(self.g['Pprime'], f)
				self._write_array(self.g['psiRZ'].flatten(), f)
				self._write_array(self.g['qpsi'], f)
				f.write(str(self.g['Nlcfs']).rjust(5) + str(self.g['Nwall']).rjust(5) + '\n')	# these need to be 5 char long numbers with leading spaces
				self._write_array(self.g['lcfs'].flatten(), f)
				self._write_array(self.g['wall'].flatten(), f)
				f.write(str(KVTOR) + ' ' + format(RVTOR, ' .9E') + ' ' + str(NMASS) + '\n')
				self._write_array(RHOVN, f)
	
			print('Wrote new gfile')

		# --------------------------------------------------------------------------------
		def readNetCDF(self, filename, nc_time):
			"""
			reads an IMAS formatted equilibrium netcdf
			"""
			imas_nc = IMAS_EQ.netCDF()
			self.data = imas_nc.readNetCDF(filename, nc_time)

			return
		  
		# --------------------------------------------------------------------------------
		# Interpolation function handles for all 1-D fields in the g-file
		def profiles(self, BtMult):
			# ---- Profiles ----
			fpol = self.data.get('fpol') * BtMult
			ffunc = interp.UnivariateSpline(np.linspace(0., 1., np.size(fpol)), fpol, s=0)
			fprime = self.data.get('ffprime')/fpol
			fpfunc = interp.UnivariateSpline(np.linspace(0., 1., np.size(fprime)), fprime, s=0)
			ffprime = self.data.get('ffprime')
			ffpfunc = interp.UnivariateSpline(np.linspace(0., 1., np.size(ffprime)), ffprime, s=0)
			pprime = self.data.get('pprime')
			ppfunc = interp.UnivariateSpline(np.linspace(0., 1., np.size(pprime)), pprime, s=0)
			pres = self.data.get('pres')
			pfunc = interp.UnivariateSpline(np.linspace(0., 1., np.size(pres)), pres, s=0)
			q_prof = self.data.get('qpsi')
			qfunc = interp.UnivariateSpline(np.linspace(0., 1., np.size(q_prof)), q_prof, s=0)

			return {'fpol': fpol, 'ffunc': ffunc, 'fprime': fprime, 'fpfunc': fpfunc,
					'ffprime': ffprime, 'ffpfunc': ffpfunc, 'pprime': pprime, 'ppfunc': ppfunc,
					'pres': pres, 'pfunc': pfunc, 'q_prof': q_prof, 'qfunc': qfunc}

		# --------------------------------------------------------------------------------
		# 1-D and 2-D (R,Z) grid
		def RZ_params(self):
			dR = (self.Rmax - self.Rmin)/(self.nw - 1)
			Rs1D = np.linspace(self.Rmin, self.Rmax, self.nw)
			dZ = (self.Zmax - self.Zmin)/(self.nh - 1)
			Zs1D = np.linspace(self.Zmin, self.Zmax, self.nh)
			Rs2D, Zs2D = np.meshgrid(Rs1D, Zs1D)
			return {'Rs1D': Rs1D, 'dR': dR, 'Zs1D': Zs1D, 'dZ': dZ, 'Rs2D': Rs2D, 'Zs2D': Zs2D}

		# --------------------------------------------------------------------------------
		# 1-D and 2-D poloidal flux, normalized and regular
		# compared to the integral definition of psipol
		# (psipol = 2pi integral_Raxis^Rsurf(Bpol * R * dR))
		# regular is shifted by self.siAxis and missing the factor 2*pi !!!
		# so: psipol = psi1D = 2pi * (self.siBry-self.siAxis) * psiN1D
		def getPsi(self, psiMult):
			psiN1D = np.linspace(0.0, 1.0, self.nw)
			psi1D = 2 * np.pi * (self.siBry - self.siAxis) * psiN1D
			psi2D = self.data.get('psirz') * psiMult
			psiN_2D = (psi2D - self.siAxis) / (self.siBry - self.siAxis)
			# psiN_2D[np.where(psiN_2D > 1.2)] = 1.2
			return {'psi2D': psi2D, 'psiN_2D': psiN_2D, 'psi1D': psi1D, 'psiN1D': psiN1D}

		# --------------------------------------------------------------------------------
		# 1-D normalized toroidal flux
		# dpsitor/dpsipol = q  ->  psitor = integral(q(psipol) * dpsipol)
		# dummy input variable for backward compatability <-> this input is unused!!!
		def getTorPsi(self, dummy=None):
			dpsi = (self.siBry - self.siAxis)/(self.nw - 1) * 2*np.pi
			#try except to accomodate multiple versions of scipy (after 1.6 cumtrapz -> cumulative_trapezoid)
			try:
				hold = integ.cumtrapz(self.PROFdict['q_prof'], dx=dpsi) * np.sign(self.bcentr)
			except:
				hold = integ.cumulative_trapezoid(self.PROFdict['q_prof'], dx=dpsi) * np.sign(self.bcentr)
			psitor = np.append(0, hold)
			psitorN1D = (psitor - psitor[0])/(psitor[-1] - psitor[0])
			return {'psitorN1D': psitorN1D, 'psitor1D': psitor}

		# --------------------------------------------------------------------------------
		# (R,Z) and B-fields on a single flux surface
		def getBs_FluxSur(self, psiNVal):
			R_hold = np.ones(self.thetapnts)
			Z_hold = np.ones(self.thetapnts)
			for thet in enumerate(self.theta):
				try:
					Rneu, Zneu = self.__comp_newt__(psiNVal, thet[1], self.rmaxis, self.zmaxis,
													self.psiFunc)
				except RuntimeError:
					Rneu, Zneu = self.__comp_bisec__(psiNVal, thet[1], self.rmaxis, self.zmaxis,
													 self.Zlowest, self.psiFunc)
				R_hold[thet[0]] = Rneu
				Z_hold[thet[0]] = Zneu

			Bp_hold = self.BpFunc.ev(R_hold, Z_hold)
			fpol_psiN = self.PROFdict['ffunc'](psiNVal)*np.ones(np.size(Bp_hold))
			fluxSur = eq.FluxSurface(fpol_psiN[0:-1], R_hold[0:-1], Z_hold[0:-1], Bp_hold[0:-1],
									 self.theta[0:-1])
			Bt_hold = np.append(fluxSur._Bt, fluxSur._Bt[0])	# add last point = first point
			Bmod = np.append(fluxSur._B, fluxSur._B[0])			# add last point = first point
			return {'Rs': R_hold, 'Zs': Z_hold, 'Bp': Bp_hold, 'Bt': Bt_hold, 'Bmod': Bmod,
					'fpol_psiN': fpol_psiN, 'FS': fluxSur}


		# --------------------------------------------------------------------------------
		# Shaping of a single flux surface
		def get_FluxShape(self, psiNVal, N = None):
			if N is None: N = self.thetapnts
			Rs,Zs = self.flux_surface(psiNVal, N)

			b = (Zs.max() - Zs.min())/2.0
			a = (Rs.max() - Rs.min())/2.0
			R = (Rs.max() + Rs.min())/2.0
			d = Rs.min() + a - Rs[Zs.argmax()]
			c = Rs.min() + a - Rs[Zs.argmin()]

			return {'kappa': (b/a), 'tri_avg': (c+d)/2/a, 'triUP': (d/a), 'triLO': (c/a), 'aminor': a, 'aspect': R/a, 'radius': R}


		# --------------------------------------------------------------------------------
		# Shaping of a single flux surface; Old DEPRECATED version
		def get_FluxShapeOLD(self, psiNVal):
			FluxSur = self.getBs_FluxSur(psiNVal)

			b = (FluxSur['Zs'].max()-FluxSur['Zs'].min())/2
			a = (FluxSur['Rs'].max()-FluxSur['Rs'].min())/2
			R = (FluxSur['Rs'].max()+FluxSur['Rs'].min())/2
			d = (FluxSur['Rs'].min()+a) - FluxSur['Rs'][np.where(FluxSur['Zs'] == FluxSur['Zs'].max())]
			c = (FluxSur['Rs'].min()+a) - FluxSur['Rs'][np.where(FluxSur['Zs'] == FluxSur['Zs'].min())]

			return {'kappa': (b/a), 'tri_avg': (c+d)/2/a, 'triUP': (d/a), 'triLO': (c/a), 'aminor': a, 'aspect': R/a, 'radius': R}

		# --------------------------------------------------------------------------------
		# (R,Z) and B-fields for all flux surfaces given by 1-D normalized poloidal flux array psiN1D
		def get_allFluxSur(self, rerun = False):
			if not rerun:
				if self.FluxSurfList is not None: return self.FluxSurfList
			
			FluxSurList = []
			getRZ_failed = False
			try: R, Z = self.__get_RZ__(self.theta, self.PSIdict['psiN1D'], quiet=True)
			except: getRZ_failed = True
			if getRZ_failed: print('getRZ failed, using backup method')

			for i, psiNVal in enumerate(self.PSIdict['psiN1D']):
				if getRZ_failed:
					R_hold,Z_hold = self.flux_surface(psiNVal, 0, theta = self.theta)
				else:
					R_hold = R[i, :]
					Z_hold = Z[i, :]

				Bp_hold = self.BpFunc.ev(R_hold, Z_hold)
				fpol_psiN = self.PROFdict['fpol'][i] * np.ones(self.thetapnts)

				# eq.FluxSurface requires theta = [0:2pi] without the last point
				fluxSur = eq.FluxSurface(fpol_psiN[0:-1], R_hold[0:-1], Z_hold[0:-1], Bp_hold[0:-1], self.theta[0:-1])
				Bt_hold = np.append(fluxSur._Bt, fluxSur._Bt[0])	# add last point = first point
				Bmod = np.append(fluxSur._B, fluxSur._B[0])			# add last point = first point

				FluxSur = {'Rs':R_hold, 'Zs':Z_hold, 'Bp':Bp_hold, 'Bt':Bt_hold,
						   'Bmod':Bmod, 'fpol_psiN':fpol_psiN, 'FS':fluxSur}

				FluxSurList.append(FluxSur)
				
			self.FluxSurfList = FluxSurList
			return FluxSurList

		# --------------------------------------------------------------------------------
		# B-fields and its derivatives in 2-D poloidal plane given by (R,Z) grid
		def getBs_2D(self, FluxSurfList=None):
			RZdict = self.RZdict
			dpsidR_loc = -1*self.dpsidR
			dpsidZ_loc = -1*self.dpsidZ
			d2psi_dZ2, _ = np.gradient(dpsidZ_loc, RZdict['dZ'], RZdict['dR'])
			d2psi_dRdZ, d2psi_dR2 = np.gradient(dpsidR_loc, RZdict['dZ'], RZdict['dR'])

			Rs_hold2D = np.ones((self.nw, self.thetapnts))
			Zs_hold2D = np.ones((self.nw, self.thetapnts))
			Btot_hold2D = np.ones((self.nw, self.thetapnts))
			Bp_hold2D = np.ones((self.nw, self.thetapnts))
			Bt_hold2D = np.ones((self.nw, self.thetapnts))
			dpsidR_hold2D = np.ones((self.nw, self.thetapnts))
			dpsidZ_hold2D = np.ones((self.nw, self.thetapnts))
			d2psidR2_hold2D = np.ones((self.nw, self.thetapnts))
			d2psidZ2_hold2D = np.ones((self.nw, self.thetapnts))
			d2psidRdZ_hold2D = np.ones((self.nw, self.thetapnts))

			dpsidRFunc = interp.RectBivariateSpline(RZdict['Rs1D'], RZdict['Zs1D'], dpsidR_loc.T)
			dpsidZFunc = interp.RectBivariateSpline(RZdict['Rs1D'], RZdict['Zs1D'], dpsidZ_loc.T)
			d2psidR2Func = interp.RectBivariateSpline(RZdict['Rs1D'], RZdict['Zs1D'], d2psi_dR2.T)
			d2psidZ2Func = interp.RectBivariateSpline(RZdict['Rs1D'], RZdict['Zs1D'], d2psi_dZ2.T)
			d2psidRdZFunc = interp.RectBivariateSpline(RZdict['Rs1D'], RZdict['Zs1D'], d2psi_dRdZ.T)

			if(FluxSurfList is None):
				FluxSurfList = self.get_allFluxSur()

			for i in range(self.nw):
				R_hold = FluxSurfList[i]['Rs']
				Z_hold = FluxSurfList[i]['Zs']

				Rs_hold2D[i, :] = R_hold
				Zs_hold2D[i, :] = Z_hold
				Btot_hold2D[i, :] = FluxSurfList[i]['Bmod']
				Bp_hold2D[i, :] = FluxSurfList[i]['Bp']
				Bt_hold2D[i, :] = FluxSurfList[i]['Bt']
				dpsidR_hold2D[i,:] = dpsidRFunc.ev(R_hold, Z_hold)
				dpsidZ_hold2D[i,:] = dpsidZFunc.ev(R_hold, Z_hold)
				d2psidR2_hold2D[i,:] = d2psidR2Func.ev(R_hold, Z_hold)
				d2psidZ2_hold2D[i,:] = d2psidZ2Func.ev(R_hold, Z_hold)
				d2psidRdZ_hold2D[i,:] = d2psidRdZFunc.ev(R_hold, Z_hold)

			return {'dpsidZ_2D':dpsidZ_hold2D,'dpsidR_2D':dpsidR_hold2D,'d2psidR2_2D':d2psidR2_hold2D,'d2psidZ2_2D':d2psidZ2_hold2D,
					'd2psidRdZ_2D':d2psidRdZ_hold2D,'Bp_2D':Bp_hold2D,'Bt_2D':Bt_hold2D,'Btot_2D':Btot_hold2D,
					'Rs_2D':Rs_hold2D,'Zs_2D':Zs_hold2D}


		# --------------------------------------------------------------------------------
		# normal and geodesic curvature, and local shear in 2-D plane
		def get_Curv_Shear(self, FluxSurfList = None, Bdict = None):
			curvNorm_2D = np.ones((self.nw,self.thetapnts))
			curvGeo_2D = np.ones((self.nw,self.thetapnts))
			shear_fl = np.ones((self.nw,self.thetapnts))

			if(FluxSurfList is None):
				FluxSurfList = self.get_allFluxSur()
			if(Bdict is None):
				Bdict = self.getBs_2D(FluxSurfList)

			for i, psiNVal in enumerate(self.PSIdict['psiN1D']):
				fprint_psiN = self.PROFdict['fpfunc'](psiNVal)*np.ones(self.thetapnts)

				R = FluxSurfList[i]['Rs']
				Bp = FluxSurfList[i]['Bp']
				B = FluxSurfList[i]['Bmod']
				fpol_psiN = FluxSurfList[i]['fpol_psiN']

				kapt1 = (fpol_psiN**2)*(Bdict['dpsidR_2D'][i, :])
				kapt2 = (Bdict['d2psidR2_2D'][i, :]*Bdict['dpsidZ_2D'][i, :]**2) + ((Bdict['dpsidR_2D'][i, :]**2)*Bdict['d2psidZ2_2D'][i, :])
				kapt3 = (2*Bdict['dpsidR_2D'][i,:]*Bdict['d2psidRdZ_2D'][i,:]*Bdict['dpsidZ_2D'][i,:])
				curvNorm_2D[i,:] = (kapt1 + Bdict['Rs_2D'][i,:]*(kapt2-kapt3))/(R**4 * Bp * B**2)

				kap2t1 = Bdict['d2psidRdZ_2D'][i,:]*(Bdict['dpsidR_2D'][i,:]**2 - Bdict['dpsidZ_2D'][i,:]**2)
				kap2t2 = Bdict['dpsidR_2D'][i,:]*Bdict['dpsidZ_2D'][i,:]*(Bdict['d2psidR2_2D'][i,:] - Bdict['d2psidZ2_2D'][i,:])
				kap2t3 = Bdict['dpsidZ_2D'][i,:] * R**2 * B**2
				# curvGeo_2D[i, :] = -1*fpol_psiN*(Bdict['Rs_2D'][i, :]*(kap2t1 - kap2t2 + kap2t3))/(R**5 * Bp * B**3)
				# maybe no -1 up front?
				curvGeo_2D[i, :] = fpol_psiN*(Bdict['Rs_2D'][i, :]*(kap2t1 - kap2t2) + kap2t3)/(R**5 * Bp * B**3)


				coeft1 = fpol_psiN / (R**4 * Bp**2 * B**2)
				coeft2 = ((Bdict['d2psidR2_2D'][i, :] -
						  Bdict['d2psidZ2_2D'][i, :]) * (Bdict['dpsidR_2D'][i, :]**2 -
						  Bdict['dpsidZ_2D'][i, :]**2) +
							(4*Bdict['dpsidR_2D'][i,:] * Bdict['d2psidRdZ_2D'][i,:] * Bdict['dpsidZ_2D'][i,:]))
				sht2 = fpol_psiN * Bdict['dpsidR_2D'][i,:] / (R**3 * B**2)
				sht3 = fprint_psiN * Bp**2 / (B**2)
				shear_fl[i, :] = coeft1 * coeft2 + sht2 - sht3

			return {'curvNorm_2D': curvNorm_2D, 'curvGeo_2D': curvGeo_2D,
					'localShear_2D': shear_fl, 'Rs_2D': Bdict['Rs_2D'], 'Zs_2D': Bdict['Zs_2D']}

		# --------------------------------------------------------------------------------
		# 1-D parallel current density profile
		def cur_density(self, FluxSurfList=None, get_jtor=True):
			import scipy.constants
			mu0 = scipy.constants.mu_0

			PSIdict = self.PSIdict
			PROFdict = self.PROFdict

			Bsqrd_prof = np.ones(self.nw)

			if(FluxSurfList is None):
				FluxSurfList = self.get_allFluxSur()

			for i, psi in enumerate(PSIdict['psiN1D']):
				Bsqrd_prof[i] = FluxSurfList[i]['FS'].Bsqav()

			# parallel current calc
			# <jpar> = <(J (dot) B)>/B0 = (fprime*<B^2>/mu0 + pprime*fpol)/B0
			jpar1D = (PROFdict['fprime']*Bsqrd_prof/mu0 + PROFdict['pprime']*PROFdict['fpol']) / self.bcentr/1.e6 * np.sign(self.Ip)

			# <jtor> = <R*pprime + ffprime/R/mu0>
			# jtor1D = np.abs(self.Rsminor*PROFdict['pprime'] +(PROFdict['ffprime']/self.Rsminor/mu0))/1.e6
			if get_jtor:
				jtor1D = self. jtor_profile(FluxSurfList)
			else:
				jtor1D = 0

			return {'jpar': jpar1D, 'jtor': jtor1D, 'Bsqrd': Bsqrd_prof}

		# --------------------------------------------------------------------------------
		# 1-D toroidal current density profile
		def jtor_profile(self, FluxSurfList=None):
			import scipy.constants
			mu0 = scipy.constants.mu_0

			PSIdict = self.PSIdict
			PROFdict = self.PROFdict

			jtor1D = np.ones(self.nw)

			if(FluxSurfList is None):
				FluxSurfList = self.get_allFluxSur()

			# get flux surface average
			for i, psi in enumerate(PSIdict['psiN1D']):
				jtorSurf = PROFdict['pprime'][i]*FluxSurfList[i]['Rs'] + PROFdict['ffprime'][i]/FluxSurfList[i]['Rs']/mu0
				f_jtorSurf = eq.interpPeriodic(self.theta[0:-1], jtorSurf[0:-1], copy = False)
				jtor1D[i] = FluxSurfList[i]['FS'].average(f_jtorSurf)

			# <jtor> = <R*pprime + ffprime/R/mu0>
			jtor1D = np.abs(jtor1D)/1.e6 * np.sign(self.Ip)
			return jtor1D


		# --------------------------------------------------------------------------------
		# Flux surface enclosed 2D-Volume (Area inside surface) and derivative (psi)
		def volume2D(self, FluxSurfList = None):
			V = np.zeros(self.nw)
			psi = self.PSIdict['psiN1D']

			if(FluxSurfList is None):
				FluxSurfList = self.get_allFluxSur()

			for i in range(1, self.nw):
				rsq = (FluxSurfList[i]['Rs'] - self.rmaxis)**2 + (FluxSurfList[i]['Zs'] - self.zmaxis)**2
				try: V[i] = 0.5 * integ.simps(rsq, self.theta)
				except: V[i] = 0.5 * integ.trapz(rsq, self.theta)

			# dV/dpsi
			dV = np.zeros(self.nw)
			for i in range(1, self.nw - 1):
				dV[i] = (V[i+1] - V[i-1]) / (psi[i+1] - psi[i-1])

			dV[0] = (-V[2] + 4*V[1] - 3*V[0]) / (psi[2] - psi[0])
			dV[-1] = (V[-3] - 4*V[-2] + 3*V[-1]) / (psi[-1] - psi[-3])

			return {'V':V, 'Vprime':dV}

		# --------------------------------------------------------------------------------
		# compute and return all of the above
		def get_all(self):
			paramDICT={'Rs1D':self.RZdict['Rs1D'], 'Zs1D':self.RZdict['Zs1D'], 'theta':self.theta,
					   'psi2D':self.PSIdict['psi2D'], 'psiN1D':self.PSIdict['psiN1D'], 'psiN_2D':self.PSIdict['psiN_2D']}

			# Prepare all flux surfaces
			FluxSurList = self.get_allFluxSur()

			# 2-D B-field properties
			Bdict = self.getBs_2D(FluxSurList)
			# returns 'dpsidZ_2D', 'dpsidR_2D', 'd2psidR2_2D', 'd2psidZ2_2D', 'd2psidRdZ_2D',
			#		  'Bp_2D', 'Bt_2D', 'Btot_2D', 'Rs_2D', 'Zs_2D'
			paramDICT['Rs_2D'] = Bdict['Rs_2D']
			paramDICT['Zs_2D'] = Bdict['Zs_2D']
			paramDICT['Btot_2D'] = Bdict['Btot_2D']
			paramDICT['Bp_2D'] = Bdict['Bp_2D']
			paramDICT['Bt_2D'] = Bdict['Bt_2D']

			# 2-D local shear
			SHEARdict = self.get_Curv_Shear(FluxSurList, Bdict) # returns 'curvNorm_2D', 'curvGeo_2D', 'localShear_2D'
			paramDICT['curvNorm_2D'] = SHEARdict['curvNorm_2D']
			paramDICT['curvGeo_2D'] = SHEARdict['curvGeo_2D']
			paramDICT['localShear_2D'] = SHEARdict['localShear_2D']

			# 1-D current density
			Jdict = self.cur_density(FluxSurList)	# returns 'jpar', 'jtor'
			paramDICT['jpar1D'] = Jdict['jpar']
			paramDICT['jtor1D'] = Jdict['jtor']

			# q profile
			qprof1D = self.PROFdict['qfunc'](self.PSIdict['psiN1D'])
			paramDICT['qprof1D'] = qprof1D

			# pressure profile
			press1D = self.PROFdict['pfunc'](self.PSIdict['psiN1D'])
			paramDICT['press1D'] = press1D

			# toroidal field profile
			btor1D = self.PROFdict['ffunc'](self.PSIdict['psiN1D'])/self.Rsminor
			paramDICT['btor1D'] = btor1D

			# toroidal flux
			paramDICT['psitorN1D'] = self.getTorPsi()['psitorN1D']

			return paramDICT

		# --------------------------------------------------------------------------------
		# returns arrays R and Z of N points along psi = const. surface
		def flux_surface(self, psi0, N, theta=None):
			if (theta is None):
				theta = np.linspace(0, 2 * np.pi, N + 1)[0:-1]
			else:
				N = len(theta)

			r = np.zeros(theta.shape)

			# get maximum r from separatrix in g-file
			Rsep = self.lcfs[:,0]
			Zsep = self.lcfs[:,1]
			idxup = Zsep.argmax()
			idxdwn = Zsep.argmin()
			rup = np.sqrt((Rsep[idxup] - self.rmaxis)**2 + (Zsep[idxup] - self.zmaxis)**2)
			rdwn = np.sqrt((Rsep[idxdwn] - self.rmaxis)**2 + (Zsep[idxdwn] - self.zmaxis)**2)
			rmax = max([rup, rdwn])

			# iterate
			for i in range(N):
				r[i] = self.__bisec__(psi0, theta[i], b=rmax)

			R = r * np.cos(theta) + self.rmaxis
			Z = r * np.sin(theta) + self.zmaxis

			return R, Z

		# --------------------------------------------------------------------------------
		def get_j2D(self):
			mu0 = 4 * np.pi*1e-7
			jR = np.zeros(self.RZdict['Rs2D'].shape)
			jZ = np.zeros(self.RZdict['Rs2D'].shape)
			jtor = np.zeros(self.RZdict['Rs2D'].shape)

			for i in range(self.nw):
				jR[:, i] = -deriv(self.Bt_2D[:, i], self.RZdict['Zs2D'][:, i])
				jtor[:, i] = deriv(self.B_R[:, i], self.RZdict['Zs2D'][:, i])

			for i in range(self.nh):
				jZ[i, :] = deriv(self.Bt_2D[i, :], self.RZdict['Rs2D'][i, :])
				jtor[i, :] -= deriv(self.B_Z[i, :], self.RZdict['Rs2D'][i, :])

			jZ += self.Bt_2D/self.RZdict['Rs2D']

			jR /= mu0
			jZ /= mu0
			jtor /= mu0
			
			Zsep = self.g['lcfs'][:,1]
			idx = np.where((self.PSIdict['psiN_2D'] > 1.0) | (abs(self.RZdict['Zs2D']) > abs(Zsep.max())))
			jR[idx] = 0
			jZ[idx] = 0
			jtor[idx] = 0

			jpar = (jR*self.B_R + jZ*self.B_Z + jtor*self.Bt_2D) * np.sign(self.Ip)
			jpar /= np.sqrt(self.B_R**2 + self.B_Z**2 + self.Bt_2D**2)
			jtot = np.sqrt(jR**2 + jZ**2 + jtor**2)
			jtor *= np.sign(self.Ip)

			return {'R':self.RZdict['Rs2D'], 'Z':self.RZdict['Zs2D'], 'j2D':jtot, 'jpar2D':jpar,
					'jR2D':jR, 'jZ2D':jZ, 'jtor2D':jtor}
					
		# --------------------------------------------------------------------------------
		def plot(self, fig = None, c = None):
			"""
			fig: integer number of figure window to use, e.g. 1
			c: string of color code, e.g. 'k' or 'r'
			"""
			import matplotlib.pyplot as plt
			if c is None: c = 'k'
			R1d = self.RZdict['Rs1D']
			Z1d = self.RZdict['Zs1D']
			Rs, Zs = np.meshgrid(R1d, Z1d)
			levs = np.append(0.01, np.arange(0.1, 1.1, 0.1))

			if fig is None: plt.figure(figsize=(6, 10))
			else: plt.figure(fig)
			
			plt.plot(self.rmaxis, self.zmaxis, c+'x', markersize = 5, linewidth = 2)
			cs1 = plt.contour(Rs, Zs, self.PSIdict['psiN_2D'], levs, colors = c)
			cs1.collections[0].set_label('EFIT')
			plt.plot(self.g['lcfs'][:,0],self.g['lcfs'][:,1], c, lw = 2, label = 'EFIT Bndy')
			
			if fig is None:
				plt.plot(self.g['wall'][:,0],self.g['wall'][:,1], 'k--')
				plt.xlim(self.g['wall'][:,0].min()*0.97, self.g['wall'][:,0].max()*1.03)
				plt.ylim(self.g['wall'][:,1].min()*1.03, self.g['wall'][:,1].max()*1.03)
				plt.xlabel('R [m]')
				plt.ylabel('Z [m]')
				plt.gca().set_aspect('equal')
				plt.title('Shot: ' + str(self.g['shot']) + '   Time: ' + str(self.g['time']) + ' ms', fontsize = 18)

		# --------------------------------------------------------------------------------
		def plotProfile(self, what = 'all', fig = None, c = None, label = ''):
			"""
			what: keyword of what profile to plot. default is 'all' and plots all 6 relevant profiles
			fig: integer number of figure window to use, e.g. 1
			c: string of color code, e.g. 'k' or 'r'
			label: string that becomes the label for the plot in the legend
			"""
			import matplotlib.pyplot as plt
			if c is None: c = 'k'
			
			psi = self.PSIdict['psiN1D']
			
			if what in ['p','P','Pres','pres','pressure','Ptot','ptot','Press','press','Pressure']:
				ylabel = 'p$_{tot}$ [kPa]'
				y = self.PROFdict['pres']*1e-3
			elif what in ['q','qprof','q_prof']:
				ylabel = 'q'
				y = self.PROFdict['q_prof']
			elif what in ['j','jtor','current density','J']:
				ylabel = "j$_{tor}$ [MA/m$^2$]"
				y = self.jtor_profile()
			elif what in ['I','Ip','current']:
				from scipy.integrate import cumtrapz
				ylabel = "I [MA]"
				FluxSurfList = self.get_allFluxSur()
				jtor = self.jtor_profile(FluxSurfList)
				Vprime = self.volume2D(FluxSurfList)['Vprime']
				y = np.append(0, cumtrapz(jtor * Vprime, psi)) # current profile: I = integral(jtor dV) = integral(jtor * dV/dpsi * dpsi) = integral(jtor * dV/dpsi * dpsi/ds * ds)
			elif what in ['F','f','fpol','Fpol']:
				ylabel = 'F$_{pol}$ [Tm]'
				y = self.PROFdict['fpol']
			elif what in ['pprime','Pprime','gradP']:
				ylabel = "dp/d$\\psi$ [kPa]"
				y = self.PROFdict['pprime']*1e-3
			elif what in ['ffprime','FFprime']:
				ylabel = 'F dF/d$\\psi$ [T$^2$m$^2$]'
				y = self.PROFdict['ffprime']
		   
			if what in ['all']:
				jtor = self.jtor_profile()
	
				fig = plt.figure(figsize = (15,11))
				
				ax1 = fig.add_subplot(321, aspect = 'auto')
				ax1.set_xlim(0,1)
				ax1.set_ylabel('p$_{tot}$ [kPa]')
				ax1.get_xaxis().set_ticklabels([])
				ax1.plot(psi, self.PROFdict['pres']*1e-3, '-', color = c, lw = 2)
				
				ax2 = fig.add_subplot(323, aspect = 'auto')
				ax2.set_xlim(0,1)
				ax2.set_ylabel('q')
				ax2.get_xaxis().set_ticklabels([])
				ax2.plot(psi, self.PROFdict['q_prof'], '-', color = c, lw = 2)
				
				ax3 = fig.add_subplot(325, aspect = 'auto')
				ax3.set_xlim(0,1)
				ax3.set_ylabel("dI/d$\\psi$ [MA]")
				ax3.set_xlabel('$\\psi$')
				ax3.plot(psi, jtor, '-', color = c, lw = 2)
				
				ax4 = fig.add_subplot(322, aspect = 'auto')
				ax4.set_xlim(0,1)
				ax4.set_ylabel('F$_{pol}$ [Tm]')
				ax4.get_xaxis().set_ticklabels([])
				ax4.plot(psi, self.PROFdict['fpol'], '-', color = c, lw = 2)
				
				ax5 = fig.add_subplot(324, aspect = 'auto')
				ax5.set_xlim(0,1)
				ax5.set_ylabel("dp/d$\\psi$ [kPa]")
				ax5.get_xaxis().set_ticklabels([])
				ax5.plot(psi, self.PROFdict['pprime']*1e-3, '-', color = c, lw = 2)
				
				ax6 = fig.add_subplot(326, aspect = 'auto')
				ax6.set_xlim(0,1)
				ax6.set_ylabel('F dF/d$\\psi$ [T$^2$m$^2$]')
				ax6.set_xlabel('$\\psi$')
				ax6.plot(psi, self.PROFdict['ffprime'], '-', color = c, lw = 2)
			else:
				if fig is None: 
					fig = plt.figure()
					plt.xlim(0,1)
					plt.xlabel('$\\psi$')
					plt.ylabel(ylabel)
				else: 
					fig = plt.figure(fig)
				ax = fig.gca()
				ax.plot(psi, y, '-', color = c, lw = 2, label = label)
				if len(label) > 0: plt.legend()
				
			fig.tight_layout()

		# --------------------------------------------------------------------------------
		def help(self):
			"""
			prints list of all 0D parameters in EFIT
			as well as information on available profiles and functions
			"""
			print('-----------------------------------------------------------------------')
			print('------------ List of all parameters in self.g -------------------------')
			print('-----------------------------------------------------------------------')
			print('shot = ', self.g['shot'])
			print('time = ', self.g['time'],'\t\t in ms')
			print('--- Grid ---')
			print('NR = ', self.nw,'\t\t R grid points')
			print('NZ = ', self.nh,'\t\t Z grid points')
			print('dR = ', format(self.RZdict['dR'],'.4g'),'\t\t R grid resolution in [m]')
			print('dZ = ', format(self.RZdict['dZ'],'g'),'\t\t Z grid resolution in [m]')
			print('Xdim = ', format(self.rdim,'g'),'\t\t total R grid length in [m]')
			print('Zdim = ', format(self.zdim,'g'),'\t\t total Z grid length in [m]')
			print('R0 = ', format(abs(self.R0),'g'),'\t\t center point in R in [m]')
			print('R1 = ', format(self.R1,'g'),'\t\t min grid point in R in [m]')
			print('Zmid = ', format(self.Zmid,'g'),'\t\t center point in Z in [m]')
			print('--- Shape ---')
			print('RmAxis = ', format(self.rmaxis,'g'),'\t R of magnetic axis in [m]')
			print('ZmAxis = ', format(self.zmaxis,'g'),'\t Z of magnetic axis in [m]')
			print('aminor = ', format(self.g['aminor'],'g'),'\t minor radius in [m]')
			print('aspect = ', format(self.g['aspect'],'g'),'\t aspect ratio')
			print('kappa = ', format(self.g['kappa'],'g'),'\t elongation')
			print('triUP = ', format(self.g['triUP'],'g'),'\t upper triangularity')
			print('triLO = ', format(self.g['triLO'],'g'),'\t lower triangularity')
			print('Nlcfs = ', len(self.lcfs),'\t\t number of separatrix points')
			print('--- Plasma ---')
			print('psiAxis = ', format(self.siAxis,'g'),'\t poloidal flux at magnetic axis in [Wb]')
			print('psiSep = ', format(self.siBry,'g'),'\t poloidal flux at separatrix in [Wb]')
			print('Bt0 = ', format(self.bcentr,'g'),'\t toroidal field at R0 in [T]')
			print('Ip = ', int(self.Ip),'\t total plasma current in [A]')
			print('helicity = ', self.isHelicity,'\t\t helicity = sign(Bp/Bt) = sign(Ip/Bt0)')
			print('--- Wall ---')
			print('Nwall = ', len(self.wall),'\t\t number of wall limiter points')
			print('-----------------------------------------------------------------------')
			print('------------ List of all arrays in self.g -----------------------------')
			print('-----------------------------------------------------------------------')
			print('Fpol','\t\t\t 1D profile of R * Btor in [Tm] vs. psiN')
			print('Pres','\t\t\t 1D pressure profile in [Pa] vs. psiN')
			print('FFprime','\t\t 1D profile of Fpol * dFpol/dpsi vs. psiN')
			print('Pprime','\t\t\t 1D profile of dPres/dpsi vs. psiN')
			print('q','\t\t\t 1D safety factor profile vs. psiN')
			print('psiN','\t\t\t 1D array of normalized poloidal flux (psi-psiAxis)/(psiSep-psiAxis)')
			print('psiRZ','\t\t\t 2D array or poloidal flux psi in [Wb] on RZ grid')
			print('R','\t\t\t 1D array of R')
			print('Z','\t\t\t 1D array of R')
			print('lcfs', '\t\t\t array of (R,Z) of separatrix')
			print('wall', '\t\t\t array of (R,Z) of wall limiter')
			print('-----------------------------------------------------------------------')
			print('------------ Info on available functionality --------------------------')
			print('-----------------------------------------------------------------------')
			print('Use the .plot() function to show the poloidal cross section')
			print('Use the .plotProfile() function to show all profiles, or')
			print("	  .plotProfile(keyword) with keyword in ['Fpol','Pres','FFprime','Pprime','q','jtor','Ip']")
			print('	  to show each profile individually')
			print('1D interpolations of profiles are readily available in .PROFdict')
			print('2D Interpolations on the R,Z grid are readily available via')
			print('	  .psiFunc.ev(R,Z),	 .BRFunc.ev(R,Z),  .BZFunc.ev(R,Z),	 .BpFunc.ev(R,Z),  .BtFunc.ev(R,Z)')
			print('Further available functions include: strike line locations, flux expansion, Swall cooridante transformations,')
			print('	  curvature and shear, 2D current density, and more...')
			print('Current code version is: ', _VERSION)
			
		# --------------------------------------------------------------------------------
		def length_along_wall(self):
			"""
			Compute length along the wall
			return Swall, Swall_max
			"""
			Rwall = self.g['wall'][:,0]
			Zwall = self.g['wall'][:,1]
			Nwall = len(Rwall)
			Swall = np.zeros(Nwall)
			dir = 1
			S0 = 0
		
			Swall[0] = np.sqrt((Rwall[0] - Rwall[-1])**2 + (Zwall[0] - Zwall[-1])**2)
			if (Swall[0] > 0):
				S0 = Zwall[-1]/(Zwall[-1] - Zwall[0])*Swall[0]
				if (Zwall[0] < Zwall[-1]): dir = 1; # ccw
				else: dir = -1						# cw

			for i in range(1,Nwall):
				Swall[i] = Swall[i-1] + np.sqrt((Rwall[i] - Rwall[i-1])**2 + (Zwall[i] - Zwall[i-1])**2)	#length of curve in m
				if ((Zwall[i]*Zwall[i-1] <= 0) & (Rwall[i] < self.g['R0'])):
					if (Zwall[i-1] == Zwall[i]): continue
					t = Zwall[i-1]/(Zwall[i-1] - Zwall[i])
					S0 = Swall[i-1] + t*(Swall[i] - Swall[i-1])
					if (Zwall[i] < Zwall[i-1]): dir = 1 # ccw
					else: dir = -1						# cw

			Swall_max = Swall[-1]

			# set direction and Swall = 0 location
			for i in range(Nwall):
				Swall[i] = dir*(Swall[i] - S0);
				if (Swall[i] < 0): Swall[i] += Swall_max
				if (Swall[i] > Swall_max): Swall[i] -= Swall_max
				if (abs(Swall[i]) < 1e-12): Swall[i] = 0
	
			return Swall, Swall_max
	
		# --------------------------------------------------------------------------------
		def all_points_along_wall(self, swall, get_index = False, get_normal = False):
			"""
			Compute matching point R,Z to given swall along wall
			This uses vectorization instead of a for loop and is much faster than the old version
			return R,Z
			"""
			Rwall = self.g['wall'][:,0]
			Zwall = self.g['wall'][:,1]
	
			isFloat = False
			if isinstance(swall, float) | isinstance(swall, np.float64): 
				swall = np.array([swall])
				isFloat = True
	
			N = len(swall)
			swall = swall % self.Swall_max
			swall[swall < 0] += self.Swall_max;

			idx = -np.ones(N, dtype = int)
			use = np.ones(N, dtype = bool)
			Nuse = N
	
			# locate discontinuity
			idx_jump = np.where(np.abs(np.diff(self.Swall)) > 0.5*self.Swall_max)[0]
			if len(idx_jump) == 0: idx_jump = 0
			else: idx_jump = idx_jump[0] + 1

			# This case occurs only when the wall does not have a point at inboard midplane
			k = np.where((swall > self.Swall.max()) | (swall < self.Swall.min()))[0]
			if len(k > 0):
				idx[k] = idx_jump
				use[k] = False
				Nuse -= len(k)
				swall_k = swall[k]
				k2 = np.where(np.abs(swall_k - self.Swall[idx_jump]) > 0.5*self.Swall_max)[0]
				if len(k2 > 0):
					swall_k2 = swall_k[k2]
					swall_k2[swall_k2 < self.Swall[idx_jump]] += self.Swall_max
					swall_k2[swall_k2 >= self.Swall[idx_jump]] -= self.Swall_max
					swall_k[k2] = swall_k2
				swall[k] = swall_k
	
			# locate intervall that brackets swall
			s0 = self.Swall[-1]
			for i in range(len(self.Swall)):
				if Nuse <= 0: break
				idxu = -np.ones(Nuse, dtype = int)
				uu = np.ones(Nuse, dtype = bool)
		
				s1 = self.Swall[i]
				if (abs(s1 - s0) > 0.5*self.Swall_max): #skip the jump around Swall = 0 point
					s0 = s1
					continue
		
				k = np.where((s1 - swall[use]) * (s0 - swall[use]) <= 0)[0]
				idxu[k] = i
				uu[k] = False
				Nuse -= len(k)
		
				idx[use] = idxu
				use[use] = uu
				s0 = s1

			# set bracket points
			R1,Z1,R2,Z2,i = Rwall[idx], Zwall[idx], Rwall[idx-1], Zwall[idx-1], 2
			id = np.where((R1 == R2) & (Z1 == Z2))[0]
			while len(id) > 0:		# if points are identical
				R2[id],Z2[id] = Rwall[idx[id]-i], Zwall[idx[id]-i] # works for idx == 0 as well
				id = np.where((R1 == R2) & (Z1 == Z2))[0]
				i += 1

			# linear interplation between bracket points
			dR,dZ = (R2-R1), (Z2-Z1)
			dx = np.sqrt(dR**2 + dZ**2)
			dR,dZ = dR/dx, dZ/dx				# dR,dZ is normalized
			x = np.abs(swall - self.Swall[idx])
			R,Z = (R1 + x*dR), (Z1 + x*dZ)
	
			if isFloat: R,Z,dR,dZ,idx = R[0],Z[0],dR[0],dZ[0],idx[0]
			if get_normal:	# get normal vector
				return R,Z,-dZ,dR		# nR,nZ = -dZ, dR
			else:
				if get_index: return R,Z,idx
				else: return R,Z 

		# --------------------------------------------------------------------------------
		def projectOnWall(self, Rx, Zx):
			"""
			Tries to project a point Rx,Zx onto the wall.
			The point R0,Z0 will be on the wall with s0 the wall coordinate.
			The vector (Rx-R0,Zx-Z0) is then parallel to the wall normal nR0,nZ0 in R0,Z0 and 
			has the length factor d. 
			If multiple projections are possible, usually the closest one is found. 
			This is not guaranteed and not checked.
			For points that are not projectable, like no perpendicular projection exists, 
			a wrong point R0,Z0 is returned. Such points have count == -1
			Otherwise count shows the number of iterations needed to converge.
			"""
			N = len(Rx)
			use = np.ones(N, dtype = bool)
			count = np.zeros(N)
	
			# find idx for Rwall,Zwall closest to Rx,Zx
			sint = np.linspace(0, self.Swall_max, 300)[1:-1]
			Swall = np.append(self.Swall,sint)
			Swall.sort()
			Rwall,Zwall,nRwall,nZwall = self.all_points_along_wall(Swall, get_normal = True)
			
			idx = np.zeros(N, dtype = int)
			for i in range(N):
				d2 = (Rwall - Rx[i])**2 + (Zwall - Zx[i])**2
				idx[i] = d2.argmin()
			idxstart = idx.copy()
	
			# Start point
			Rs,Zs,s = Rwall[idx], Zwall[idx], Swall[idx]
			nR,nZ = nRwall[idx], nZwall[idx]
	
			# Variables
			R0,Z0,s0,nR0,nZ0,ds,d = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
			eps = np.ones(N)
	
			# Smart Search
			for i in range(5):
				count[use] += 1
				ds[use] = -nZ[use]*(Rs[use]-Rx[use]) + nR[use]*(Zs[use]-Zx[use])
				s0[use] = s[use] + ds[use]
				R0[use],Z0[use],nR0[use],nZ0[use] = self.all_points_along_wall(s0[use], get_normal = True)
				d[use] = -nR0[use]*(R0[use]-Rx[use]) - nZ0[use]*(Z0[use]-Zx[use])
		
				# convergence paramerter
				eps[use] = (R0[use]+d[use]*nR0[use]-Rx[use])**2 + (Z0[use]+d[use]*nZ0[use]-Zx[use])**2
		
				# update use mask
				use[eps < 1e-16] = False
				if np.sum(use) == 0: break
		
				# update start values
				Rs[use],Zs[use],s[use],nR[use],nZ[use] = R0[use],Z0[use],s0[use],nR0[use],nZ0[use]
	
			# Smart Search did not converge all, try systematic search for the remaining
			if sum(use) > 0: 
				shifts = [-1,1,-2,2,-3,3,-4,4,-5,5,-6,6,-7,7,-8,8,-9,9]
				for i in range(18):
					# update start values
					count[use] += 1
					idxu = idxstart[use] + shifts[i]
					k = np.where(idxu >= len(Swall))
					if len(k) > 0: idxu[k] = idxu[k] - len(Swall)
					Rs[use],Zs[use],s[use] = Rwall[idxu], Zwall[idxu], Swall[idxu]
					nR[use],nZ[use] = nRwall[idxu], nZwall[idxu]
			
					# get new point
					ds[use] = -nZ[use]*(Rs[use]-Rx[use]) + nR[use]*(Zs[use]-Zx[use])
					s0[use] = s[use] + ds[use]
					R0[use],Z0[use],nR0[use],nZ0[use] = self.all_points_along_wall(s0[use], get_normal = True)
					d[use] = -nR0[use]*(R0[use]-Rx[use]) - nZ0[use]*(Z0[use]-Zx[use])
			
					# convergence paramerter
					eps[use] = (R0[use]+d[use]*nR0[use]-Rx[use])**2 + (Z0[use]+d[use]*nZ0[use]-Zx[use])**2
		
					# update use mask
					use[eps < 1e-16] = False
					if np.sum(use) == 0: break
			
			count[use] = -1
			return R0,Z0,s0,d,nR0,nZ0,count

		# --------------------------------------------------------------------------------
		def point_along_wall(self, swall, get_index = False):
			"""
			DEPRECATED
			Compute matching point R,Z to given swall along wall
			return R,Z
			"""
			print('Deprecation Warning: Use self.all_points_along_wall() instead')
			Rwall = self.g['wall'][:,0]
			Zwall = self.g['wall'][:,1]

			swall = swall%self.Swall_max
			if(swall < 0): swall += self.Swall_max;

			# locate discontinuity
			idx_jump = np.where(np.abs(np.diff(self.Swall)) > 0.5*self.Swall_max)[0]
			if len(idx_jump) == 0: idx_jump = 0
			else: idx_jump = idx_jump[0] + 1

			# locate intervall that brackets swall
			if ((swall > self.Swall.max()) | (swall < self.Swall.min())):
				print('This should never happen')
				idx = idx_jump;
				if (np.abs(swall - self.Swall[idx]) > 0.5*self.Swall_max):
					if (swall < self.Swall[idx]): swall += self.Swall_max
					else: swall -= self.Swall_max
			else:
				idx = 0;
				s0 = self.Swall[-1]
				for i in range(len(self.Swall)):
					s1 = self.Swall[i]

					if (abs(s1 - s0) > 0.5*self.Swall_max): #skip the jump around Swall = 0 point
						s0 = s1
						continue

					if((s1 - swall) * (s0 - swall) <= 0):
						idx = i
						break

					s0 = s1

			# set bracket points
			p1 = np.array([Rwall[idx], Zwall[idx]])
			p2, i = p1, 1
			while np.all(p2 == p1):		# if points are identical
				p2 = np.array([Rwall[idx-i], Zwall[idx-i]]) # works for idx == 0 as well
				i += 1

			# linear interplation between bracket points
			d = p2 - p1
			x = np.abs(swall - self.Swall[idx])/np.sqrt(d[0]**2 + d[1]**2); # x is dimensionless in [0,1]
			p = p1 + x*d
			if get_index: return p[0], p[1], idx
			else: return p[0], p[1] # R,Z

		# --------------------------------------------------------------------------------
		def all_points_along_wall_loop(self, swall, get_index = False, get_normal = False):
			"""
			DEPRECATED
			Compute matching point R,Z for all given s in array swall along simplified wall
			return R,Z arrays
			"""
			R,Z,idx = np.zeros(swall.shape),np.zeros(swall.shape),np.zeros(swall.shape,dtype = int)
			if get_normal:
				for i,s in enumerate(swall):		
					R[i],Z[i],idx[i] = self.point_along_wall(s,True)
				nR,nZ = self.wall_normal(idx)
				return R,Z,nR,nZ
			else:
				if get_index:
					for i,s in enumerate(swall):		
						R[i],Z[i],idx[i] = self.point_along_wall(s,get_index)
					return R,Z,idx
				else:
					for i,s in enumerate(swall):		
						R[i], Z[i] = self.point_along_wall(s)
					return R,Z

		# --------------------------------------------------------------------------------
		def wall_normal(self, idx):
			"""
			Get normal vector of wall segment idx
			"""
			Rwall = self.g['wall'][:,0]
			Zwall = self.g['wall'][:,1]

			if isinstance(idx, int): idx = [idx]
			nR,nZ = np.zeros(len(idx)),np.zeros(len(idx))
			
			for j,k in enumerate(idx):
				# set bracket points
				p1 = np.array([Rwall[k], Zwall[k]])
				p2, i = p1, 1
				while np.all(p2 == p1):		# if points are identical
					p2 = np.array([Rwall[k-i], Zwall[k-i]]) # works for idx == 0 as well
					i += 1

				# get normal vector
				d = p2 - p1
				n = np.array([-d[1], d[0]])
				n /= np.sqrt(np.sum(n**2))
				nR[j],nZ[j] = n[0],n[1]
				
			return nR, nZ

		# --------------------------------------------------------------------------------
		def strikeLines(self, quiet = True):
			"""
			Compute R,Z and swall for both strike points along the wall
			"""
			from scipy.optimize import bisect
			
			swall = np.linspace(0,self.Swall_max,300)
			R,Z = self.all_points_along_wall(swall)
			psi = self.psiFunc.ev(R,Z) - 1
			x = psi[1::] * psi[0:-1] # x < 0 only where psi changes sign
			idx = np.where(x < 0)[0]
			if not quiet: print(idx)
			if len(idx) < 2: return None	# no strike points. Probably limited discharge
			
			idxin = idx[0]
			idxout = idx[1]
			
			f = lambda s: np.float64(self.psiFunc.ev(*self.all_points_along_wall(s))) - 1
			swallin = bisect(f,swall[idxin],swall[idxin+1])
			swallout = bisect(f,swall[idxout],swall[idxout+1])
			
			# swap for upper single null
			if swallin > 0.5*self.Swall_max: swallin,swallout = swallout,swallin
			
			Rin,Zin = self.all_points_along_wall(swallin)
			Rout,Zout = self.all_points_along_wall(swallout)
			
			d = {'Rin':Rin,'Zin':Zin,'swallin':swallin,'Rout':Rout,'Zout':Zout,'swallout':swallout,'N':len(idx)}
			
			# possible double null
			if len(idx) > 3:
				idxin2 = idx[3]
				idxout2 = idx[2]
				swallin2 = bisect(f,swall[idxin2],swall[idxin2+1])
				swallout2 = bisect(f,swall[idxout2],swall[idxout2+1])
				Rin2,Zin2 = self.all_points_along_wall(swallin2)
				Rout2,Zout2 = self.all_points_along_wall(swallout2)
				d2 = {'Rin2':Rin2,'Zin2':Zin2,'swallin2':swallin2,'Rout2':Rout2,'Zout2':Zout2,'swallout2':swallout2}
				for key in d2.keys(): d[key] = d2[key]
			
			return d



		# --------------------------------------------------------------------------------
		def fluxExpansion(self, target='in'):
			"""
			Calculate flux expansion fx between outer midplane and wall
			target gives the strike line for which to get fx
			  lower targets: 'in', 'out'
			  upper targets: 'in2', 'out2'
			returns flux expansion fx
			Based on: A. Loarte et al., Journal of Nuclear Materials 266-269 (1999) 587-592
			"""			   
			Rmin,Rmax = self.rmaxis,self.g['wall'][:,0].max()
			R = np.linspace(Rmin,Rmax,200)
			Z = self.zmaxis * np.ones(200)
			psi = self.psiFunc.ev(R,Z)
			f = interp.UnivariateSpline(psi,R,s=0)			
			Rmid = f(1.0)
			Bp_mid = self.BpFunc.ev(Rmid,self.zmaxis)
			
			d = self.strikeLines()
			if ('R' + target) not in d:
				print('Unkown target in fluxExpansion')
				target = 'in'
			Rdiv = d['R' + target]
			Zdiv = d['Z' + target]
			Bp_div = self.BpFunc.ev(Rdiv,Zdiv)
			
			return (Rmid*Bp_mid) / (Rdiv*Bp_div)
			
		
		 # --------------------------------------------------------------------------------
		def helicity(self, verify = False):
			"""
			Calculate helicity from psiRZ and Fpol
			verify with Bt and Ip signs
			"""
			# get a point at outboard midplane, near the separatrix
			nR,nZ = abs(self.g['R'] - self.g['lcfs'][:,0].max()).argmin(), int(self.g['NZ']/2)
			
			# helicity = -sign(Bz/Bphi) at outboard midplane
			# Bz = -dpsidR / R;		Bphi = Fpol / R;	Fpol ~ Bt0 * R0;	Bz ~ Ip / (R-R0)
			isHelicity = np.sign(self.dpsidR[nZ,nR]/self.g['Fpol'][-1])
			expectHelicity = np.sign(self.g['Ip']/self.g['Bt0'])
			
			if not (expectHelicity == isHelicity):
				print('Helicity is inconsistent in g-File')
				if not (np.sign(self.g['Bt0']) == np.sign(self.g['Fpol'][-1])):
					print('Flip sign of Bt in g-file')
				else:
					print('Flip sign of Ip in g-file')
					
			if verify:
				print('Bt = ',format(self.g['Bt0'],'.2f'))
				print('Ip = ',format(self.g['Ip'],'.2f'))
				import matplotlib.pyplot as plt
				nR0 = abs(self.g['R'] - self.g['RmAxis']).argmin()
				plt.figure()
				plt.plot(self.g['R'][nR0:nR], self.dpsidR[nZ,nR0:nR],'k-',lw = 2, label = 'psiRZ along outer midplane')
				plt.xlabel('R [m]')
				plt.ylabel('$\\psi$')
				plt.legend()
				
				plt.figure()
				plt.plot(self.g['psiN'],self.g['Fpol'],'k-',lw = 2)
				plt.xlabel('$\\psi$')
				plt.ylabel('Fpol')
			
			return int(isHelicity)
		

		# --------------------------------------------------------------------------------
		# Private Functions
		# --------------------------------------------------------------------------------
		
		# --------------------------------------------------------------------------------
		def _write_array(self, x, f):
			"""
			write numpy array in format used in g-file:
			5 columns, 9 digit float with exponents and no spaces in front of negative numbers
			"""
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

		# --------------------------------------------------------------------------------
		def _s2psi(self, s, fluxLimit):
			f_psitor = interp.UnivariateSpline(self.PSIdict['psiN1D'], self.getTorPsi()['psitorN1D'], s=0)	 # EFIT based conversion function
			y = np.linspace(0,1,1000) * fluxLimit				   # limit normalized poloidal flux to fluxLimit
			x = f_psitor(y) / f_psitor(fluxLimit)				# x is renormalized (x = 0 -> 1) toroidal flux of psi = 0 -> fluxLimit
			f_psiN = interp.UnivariateSpline(x, y, s = 0)		# new conversion function, based on x
			return f_psiN(s)									# normalized poloidal flux, matching VMEC s

		def _psi2s(self, psi, fluxLimit):
			f_psitor = interp.UnivariateSpline(self.PSIdict['psiN1D'], self.getTorPsi()['psitorN1D'], s = 0)   # EFIT based conversion function
			y = np.linspace(0,1,1000) * fluxLimit				   # limit normalized poloidal flux to fluxLimit
			x = f_psitor(y) / f_psitor(fluxLimit)				# x is renormalized (x = 0 -> 1) toroidal flux of psi = 0 -> fluxLimit
			f_s = interp.UnivariateSpline(y, x, s = 0)			# new conversion function, based on y
			return f_s(psi)										# VMEC s, matching normalized poloidal flux

	   # --------------------------------------------------------------------------------
		# private functions to locate zero crossings through Newton method or Bisection
		def __comp_newt__(self,psiNVal,theta,rmaxis,zmaxis,psiFunc,r_st = 0.5):
			eps = 1.e-12
			litr = r_st
			dlitr = 1
			n = 0
			h=0.000005
			while(np.abs(dlitr) > eps):
				Rneu = litr*np.cos(theta)+rmaxis
				Zneu = litr*np.sin(theta)+zmaxis
				Rbac = (litr-h)*np.cos(theta)+rmaxis
				Zbac = (litr-h)*np.sin(theta)+zmaxis
				f = psiFunc.ev(Rneu,Zneu) - psiNVal
				df = (psiFunc.ev(Rbac,Zbac) - psiFunc.ev(Rneu,Zneu))/(-1*h)
				dlitr = f/df
				litr -=dlitr
				n +=1
				if(n > 100):
					raise RuntimeError("No Converg.")
			return Rneu,Zneu

		def __comp_bisec__(self,psiNVal,theta,rmaxis,zmaxis,Zlowest,psiFunc):
			running = True
			litr = 0.001
			while running:
				Rneu = litr*np.cos(theta)+rmaxis
				Zneu = litr*np.sin(theta)+zmaxis
				psineu = psiFunc.ev(Rneu,Zneu)
				if((psineu < 1.2) & (Zneu < Zlowest)):
					psineu = 1.2
				comp = psiNVal - psineu
				if(comp<0.):
					litr -=0.00001
				if(comp <= 1e-4):
					running = False
				elif(comp > 1e-4):
					litr +=0.001
				else:
					print("bunk!")
					break
			return Rneu,Zneu


		# --------------------------------------------------------------------------------
		# returns 2D-arrays R and Z for all (theta, psi) points
		# theta and psi are 1D base arrays of a regular grid.
		def __get_RZ__(self, theta, psi, quiet = False, verify = True):
			npsi = psi.size
			nt = theta.size
			R = np.zeros((npsi, nt))
			Z = np.zeros((npsi, nt))

			# get scaling factor for r
			radexp0 = 2.0		# psi scales as r**radexp0 near axis
			radexp1 = 0.25		# psi scales as r**radexp1 near edge
			unitv = np.linspace(0.0, 1.0, npsi)
			vector = np.sin(unitv*np.pi/2.0)**2
			radexp = vector*(radexp1 - radexp0) + radexp0
			rnorm = unitv**(radexp/2.0)

			# get r at last closed flux surface
			R_lcfs = self.g['lcfs'][:,0]
			Z_lcfs = self.g['lcfs'][:,1]
			r_lcfs = self.__get_r__(R_lcfs, Z_lcfs)[1::]		# here first == last, so skip first
			th_lcfs = self.__get_theta__(R_lcfs, Z_lcfs)[1::]	# here first == last, so skip first
			index = np.argsort(th_lcfs) # th-lcfs needs to be monotonically increasing

			# interpolate to get r(theta) at lcfs
			get_r_lcfs = interp.UnivariateSpline(th_lcfs[index], r_lcfs[index], s = 0)
			r = get_r_lcfs(theta)

			# set initial guess of R & Z
			for i in range(npsi):
				R[i,:] = rnorm[i]*r*np.cos(theta) + self.rmaxis
				Z[i,:] = rnorm[i]*r*np.sin(theta) + self.zmaxis

			# R, Z are an initial guess of what we are looking for, 2D arrays
			# psi is a 1D array, it is the psi we want R, Z to be on
			get_psi = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'], self.PSIdict['psiN_2D'].T)

			max_it = 10
			eps = 1e-8
			ratio = 1
			N = 0
			rho = np.sqrt(psi)

			while(ratio > eps):
				# get psi for the current R, Z
				psi_now = get_psi.ev(R.flatten(), Z.flatten())
				psi_now[psi_now < 1e-7] = 0		# check for small values (no negatives!)
				psi_now = psi_now.reshape(R.shape)
				rho_now = np.sqrt(psi_now)

				# store old values
				R_old = R.copy()
				Z_old = Z.copy()

				R -= self.rmaxis
				Z -= self.zmaxis

				# find new R, Z by interpolation
				for j in range(nt):
					x = rho_now[1::,j]
					index = np.argsort(x)

					y = R[1::,j]
					get_R = interp.UnivariateSpline(x[index], y[index], s = 0)
					R[:,j] = get_R(rho)

					y = Z[1::,j]
					get_Z = interp.UnivariateSpline(x[index], y[index], s = 0)
					Z[:,j] = get_Z(rho)

				r = np.sqrt(R**2 + Z**2)
				R += self.rmaxis
				Z += self.zmaxis

				# psi > 1 can cause nan in interpolations
				idx = np.isnan(r)
				R[idx] = R_old[idx]
				Z[idx] = Z_old[idx]

				# Compute convergence error
				delta = np.sqrt((R - R_old )**2 + (Z - Z_old)**2)
				ratio = (delta[np.invert(idx)]/(r[np.invert(idx)] + 1.0e-20)).max()

				# no convergence check
				N += 1
				if(N > max_it):
					if not quiet:
						print('Warning, iterate_RZ: bad convergence, check if error is acceptable')
						print('Iteration: ', N, ', Error: ', ratio)
					break

			if verify:
				R_chk,Z_chk = self.flux_surface(psi[2], 0, theta = theta)
				if np.abs(R[2,:] - R_chk).max() > 1e-5:
					raise RuntimeError("getRZ no convergence")
			return R, Z


		# --------------------------------------------------------------------------------
		# get poloidal angle from (R,Z) coordinates
		def __get_theta__(self, R, Z):
			Rm = R - self.rmaxis # R relative to magnetic axis
			Zm = Z - self.zmaxis # Z relative to magnetic axis

			if isinstance(Rm, np.ndarray):
				Rm[(Rm < 1e-16) & (Rm >= 0)] = 1e-16
				Rm[(Rm > -1e-16) & (Rm < 0)] = -1e-16
			else:
				if (Rm < 1e-16) & (Rm >= 0): Rm = 1e-16
				if (Rm > -1e-16) & (Rm < 0): Rm = -1e-16

			theta = np.arctan(Zm/Rm);
			if isinstance(theta, np.ndarray):
				theta[Rm < 0] += np.pi;
				theta[(Rm >= 0) & (Zm < 0)] += 2*np.pi;
			else:
				if(Rm < 0): theta += np.pi;
				if((Rm >= 0) & (Zm < 0)): theta += 2*np.pi;

			return theta

		# --------------------------------------------------------------------------------
		# get minor radius from (R,Z) coordinates
		def __get_r__(self, R, Z):
			Rm = R - self.rmaxis  # R relative to magnetic axis
			Zm = Z - self.zmaxis  # Z relative to magnetic axis
			return np.sqrt(Rm*Rm + Zm*Zm)

		# --------------------------------------------------------------------------------
		# get f(r) = psi(R(theta,r),Z(theta,r))-psi0 with theta = const.
		def __funct__(self, r, theta, psi0):
			R = r*np.cos(theta) + self.rmaxis
			Z = r*np.sin(theta) + self.zmaxis
			psi = self.psiFunc.ev(R, Z)
			f = psi - psi0
			return f

		# --------------------------------------------------------------------------------
		# get r for theta = const. and psi = psi0
		def __bisec__(self, psi0, theta, a=0, b=1.5):
			eps = 1e-14

			x = a
			f = self.__funct__(x, theta, psi0)

			if(f > 0):
				xo = a
				xu = b
			else:
				xo = b
				xu = a

			while(abs(xo-xu) > eps):
				x = (xo + xu)/2.0
				f = self.__funct__(x, theta, psi0)
				if(f > 0):
					xo = x
				else:
					xu = x

			return x

		# --- read_mds -------------------------------------------
		# reads g-file from MDS+ and writes g-file
		# Note: MDS+ data is only single precision!
		# specify shot, time (both as int) and ...
		#	tree (string)		->	EFIT tree name, default = 'EFIT01'
		# further keywords:
		#	exact (bool)		->	True: time must match time in EFIT tree, otherwise abort
		#							False: EFIT time closest to time is used (default)
		#	Server (string)		->	MDS+ server name or IP, default = 'atlas.gat.com' (for DIII-D)
		#	gpath (string)		->	path where to save g-file, default = current working dir
		def _read_mds(self, shot, time, tree='EFIT01', exact=False, Server='atlas.gat.com',
					  gpath='.'):
			import MDSplus
			print('Reading shot =', shot, 'and time =', time, 'from MDS+ tree:', tree)

			# in case those are passed in as strings
			shot = int(shot)
			time = int(time)

			# Connect to server, open tree and go to g-file
			MDS = MDSplus.Connection(Server)
			MDS.openTree(tree, shot)
			base = 'RESULTS:GEQDSK:'

			# get time slice
			signal = 'GTIME'
			k = np.argmin(np.abs(MDS.get(base + signal).data() - time))
			time0 = int(MDS.get(base + signal).data()[k])

			if (time != time0):
				if exact:
					raise RuntimeError(tree + ' does not exactly contain time ' + str(time) + '	 ->	 Abort')
				else:
					print('Warning: ' + tree + ' does not exactly contain time ' + str(time) + ' the closest time is ' + str(time0))
					print('Fetching time slice ' + str(time0))
					time = time0

			# store data in dictionary
			g = {'shot':shot, 'time':time}

			# get header line
			header = MDS.get(base + 'ECASE').data()[k]

			# get all signals, use same names as in read_g_file
			translate = {'MW': 'NR', 'MH': 'NZ', 'XDIM': 'Xdim', 'ZDIM': 'Zdim', 'RZERO': 'R0',
						 'RMAXIS': 'RmAxis', 'ZMAXIS': 'ZmAxis', 'SSIMAG': 'psiAxis',
						 'SSIBRY': 'psiSep', 'BCENTR': 'Bt0', 'CPASMA': 'Ip', 'FPOL': 'Fpol',
						 'PRES': 'Pres', 'FFPRIM': 'FFprime', 'PPRIME': 'Pprime', 'PSIRZ': 'psiRZ',
						 'QPSI': 'qpsi', 'NBBBS': 'Nlcfs', 'LIMITR': 'Nwall'}
			for signal in translate:
				g[translate[signal]] = MDS.get(base + signal).data()[k]

			g['R1'] = MDS.get(base + 'RGRID').data()[0]
			g['Zmid'] = 0.0

			RLIM = MDS.get(base + 'LIM').data()[:, 0]
			ZLIM = MDS.get(base + 'LIM').data()[:, 1]
			g['wall'] = np.vstack((RLIM, ZLIM)).T

			RBBBS = MDS.get(base + 'RBBBS').data()[k][:g['Nlcfs']]
			ZBBBS = MDS.get(base + 'ZBBBS').data()[k][:g['Nlcfs']]
			g['lcfs'] = np.vstack((RBBBS, ZBBBS)).T

			KVTOR = 0
			RVTOR = 1.7
			NMASS = 0
			RHOVN = MDS.get(base + 'RHOVN').data()[k]

			# convert floats to integers
			for item in ['NR', 'NZ', 'Nlcfs', 'Nwall']:
				g[item] = int(g[item])

			# convert single (float32) to double (float64) and round
			for item in ['Xdim', 'Zdim', 'R0', 'R1', 'RmAxis', 'ZmAxis', 'psiAxis', 'psiSep',
						 'Bt0', 'Ip']:
				g[item] = np.round(np.float64(g[item]), 7)

			# convert single arrays (float32) to double arrays (float64)
			for item in ['Fpol', 'Pres', 'FFprime', 'Pprime', 'psiRZ', 'qpsi', 'lcfs', 'wall']:
				g[item] = np.array(g[item], dtype=np.float64)

			# write g-file to disk
			if not (gpath[-1] == '/'):
				gpath += '/'
			with open(gpath + 'g' + format(shot, '06d') + '.' + format(time,'05d'), 'w') as f:
				if ('EFITD' in header[0]) and (len(header) == 6):
					for item in header: f.write(item)
				else:
					f.write('  EFITD	xx/xx/xxxx	  #' + str(shot) + '  ' + str(time) + 'ms		 ')

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

			return time

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


# ----------------------------------------------------------------------------------------
# ---- End of class ----------------------------------------------------------------------

def deriv(y, x, periodic = False):
	n = y.size
	if(n < 3):
		raise RuntimeError('deriv: arrays must have at least 3 points')

	d = np.zeros(n)
	for i in range(1, n-1):	 # inside the array
		d[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])

	if periodic:
		d[0] = 0.5*(y[1] - y[-1]) / (x[1] - x[0])
		d[-1] = 0.5*(y[0] - y[-2]) / (x[-1] - x[-2])
	else:
		if(abs(x[2] - 2*x[1] + x[0]) < 1e-12):		# equidistant x only
			d[0] = (-y[2] + 4*y[1] - 3*y[0]) / (x[2] - x[0])
		else:
			d[0] = (y[1] - y[0]) / (x[1] - x[0])

		if(abs(x[-1] - 2*x[-2] + x[-3]) < 1e-12):  # equidistant x only
			d[-1] = (y[-3] - 4*y[-2] + 3*y[-1]) / (x[-1] - x[-3])
		else:
			d[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

	return d


import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import EFIT.geqdsk as gdsk
import EFIT.equilibrium as eq
reload(eq)

class equilParams:
		
		def __init__(self, gfileNam, nw = 0, nh = 0, thetapnts = 0, grid2G = True):
			self.data = gdsk.Geqdsk()
			self.data.openFile(gfileNam)

			#---- Variables ----
			if grid2G:
				self.nw = self.data.get('nw')
				self.nh = self.data.get('nh')
				self.thetapnts = 2*self.nh
			else:
				self.nw=nw
				self.nh=nh
				self.thetapnts = thetapnts
			self.bcentr = np.abs(self.data.get('bcentr'))
			self.rmaxis = self.data.get('rmaxis')
			self.zmaxis = self.data.get('zmaxis')
			self.Rmin = self.data.get('rleft')
			self.Rmax = self.Rmin + self.data.get('rdim')
			self.Rbdry = self.data.get('rbbbs').max()
			self.Rsminor = np.linspace(self.rmaxis,self.Rbdry,self.nw)
			self.Zmin = self.data.get('zmid') - self.data.get('zdim')/2.0
			self.Zmax = self.data.get('zmid') + self.data.get('zdim')/2.0
			self.Zlowest = self.data.get('zbbbs').min()
			self.siAxis = self.data.get('simag')
			self.siBry = self.data.get('sibry')
			
			#---- default Functions ----
			self.PROFdict = self.profiles()
			self.RZdict = self.RZ_params()
			self.PSIdict = self.getPsi()
			
			#---- more Variables ----
			self.Bp_Z, self.Bp_R = np.gradient(-self.PSIdict['psi2D'], self.RZdict['dZ'], self.RZdict['dR'])
			self.Bp_2D = np.sqrt(self.Bp_R**2 + self.Bp_Z**2)/self.RZdict['Rs2D']
			self.theta = np.linspace(0.0, 2.*np.pi, self.thetapnts)

			#---- more Functions ----
			self.psiFunc = interp.RectBivariateSpline(self.RZdict['Zs1D'], self.RZdict['Rs1D'], self.PSIdict['psiN_2D'], kx=1, ky=1)
			self.BpFunc = interp.RectBivariateSpline(self.RZdict['Zs1D'], self.RZdict['Rs1D'], self.Bp_2D, kx=1, ky=1)


		# --------------------------------------------------------------------------------
		# Interpolation function handles for all 1-D fields in the g-file
		def profiles(self):
			#---- Profiles ----
			fpol = self.data.get('fpol')
			ffunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(fpol)),fpol,s=0)
			fprime = self.data.get('ffprime')/fpol
			fpfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(fprime)),fprime,s=0)
			ffprime = self.data.get('ffprime')
			ffpfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(ffprime)),fprime,s=0)
			pprime = self.data.get('pprime')
			ppfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(pprime)),pprime,s=0)
			pres = self.data.get('pres')
			pfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(pres)),pres,s=0)
			q_prof = self.data.get('qpsi')
			qfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(q_prof)),q_prof,s=0)

			return {'fpol':fpol,'ffunc':ffunc,'fprime':fprime,'fpfunc':fpfunc,'ffprime':ffprime,'ffpfunc':ffpfunc,
						'pprime':pprime,'ppfunc':ppfunc,'pres':pres,'pfunc':pfunc,'q_prof':q_prof,'qfunc':qfunc}


		# --------------------------------------------------------------------------------
		# 1-D and 2-D (R,Z) grid
		def RZ_params(self):
			dR = (self.Rmax - self.Rmin)/np.float64(self.nw - 1)
			Rs1D = np.arange(self.Rmin, self.Rmax*(1.+1.e-10), dR)
			dZ = (self.Zmax - self.Zmin)/np.float64(self.nh - 1)
			Zs1D = np.arange(self.Zmin,self.Zmax*(1.+1.e-10), dZ)
			Rs2D,Zs2D = np.meshgrid(Rs1D,Zs1D)

			return {'Rs1D':Rs1D,'dR':dR,'Zs1D':Zs1D,'dZ':dZ,'Rs2D':Rs2D,'Zs2D':Zs2D}


		# --------------------------------------------------------------------------------
		# 1-D and 2-D poloidal flux, normalized and regular
		def getPsi(self):
			g_psi2D = self.data.get('psirz')
			RZdict = self.RZdict
			psiN1D = np.linspace(0.0,1.0,self.nw)
			gRs,gZs = np.meshgrid(np.linspace(self.Rmin,self.Rmax,g_psi2D.shape[1]),np.linspace(self.Zmin,self.Zmax,g_psi2D.shape[0]))
			psi2D = interp.griddata((gRs.flatten(0),gZs.flatten(0)),g_psi2D.flatten(0),(RZdict['Rs1D'][:,None],RZdict['Zs1D'][None,:]),method='cubic',fill_value=0.0)
			psi2D = psi2D.T
			psiN_2D = (psi2D - self.siAxis)/(self.siBry-self.siAxis)
			psiN_2D[np.where(psiN_2D > 1.2)] = 1.2

			return {'psi2D':psi2D,'psiN1D':psiN1D,'psiN_2D':psiN_2D}


		# --------------------------------------------------------------------------------
		# 1-D normalized toroidal flux (from Canik's g3d.pro)
		def getTorPsi(self, qprof1D = None):
			if(qprof1D == None):
				qprof1D = self.PROFdict['qfunc'](self.PSIdict['psiN1D'])
			
			pn = np.arange(self.nw)/float((self.nw - 1))
			dpsi = (pn[1] - pn[0])*(self.siBry - self.siAxis)
			hold = np.cumsum(0.5*(qprof1D[0:self.nw-1] + qprof1D[1:self.nw])*dpsi)
			psitor = np.concatenate((np.array([0.]), hold))
			psitorN1D = (psitor - psitor[0])/(psitor[self.nw-1] - psitor[0])
			
			return psitorN1D


		# --------------------------------------------------------------------------------
		# (R,Z) and B-fields on a single flux surface
		def getBs_FluxSur(self,psiNVal):
			R_hold = np.ones(self.thetapnts)
			Z_hold = np.ones(self.thetapnts)

			for thet in enumerate(self.theta):
				try:
					Rneu, Zneu = self.__comp_newt__(psiNVal,thet[1],self.rmaxis,self.zmaxis,self.psiFunc)
				except RuntimeError:
					Rneu, Zneu = self.__comp_bisec__(psiNVal,thet[1],self.rmaxis,self.zmaxis,self.Zlowest,self.psiFunc)
				R_hold[thet[0]] = Rneu
				Z_hold[thet[0]] = Zneu

			Bp_hold = self.BpFunc.ev(Z_hold, R_hold)
			fpol_psiN = self.PROFdict['ffunc'](psiNVal)*np.ones(np.size(Bp_hold))
			fluxSur = eq.FluxSurface(fpol_psiN, R_hold, Z_hold, Bp_hold, self.theta)
			Bsqrd = fluxSur.Bsqav()

			return {'Rs':R_hold, 'Zs':Z_hold, 'Bp':Bp_hold, 'Bt':fluxSur._Bt, 'Bmod':fluxSur._B, 'Bsqrd':Bsqrd,
					'fpol_psiN':fpol_psiN}


		# --------------------------------------------------------------------------------
		# Shaping of a single flux surface
		def get_FluxShape(self,psiNVal):
			FluxSur = self.getBs_FluxSur(psiNVal)

			b= (FluxSur['Zs'].max()-FluxSur['Zs'].min())/2
			a= (FluxSur['Rs'].max()-FluxSur['Rs'].min())/2
			d = (FluxSur['Rs'].min()+a) - FluxSur['Rs'][np.where(FluxSur['Zs'] == FluxSur['Zs'].max())]
			c = (FluxSur['Rs'].min()+a) - FluxSur['Rs'][np.where(FluxSur['Zs'] == FluxSur['Zs'].min())]
			
			return {'kappa':(b/a),'tri_avg':(c+d)/2/a,'triUP':(d/a),'triLO':(c/a)}
		

		# --------------------------------------------------------------------------------
		# (R,Z) and B-fields for all flux surfaces given by 1-D normalized poloidal flux array psiN1D
		def get_allFluxSur(self):
			FluxSurList = []
			R,Z = self.__get_RZ__(self.theta, self.PSIdict['psiN1D'], quiet = True)
				
			for i in enumerate(self.PSIdict['psiN1D']):
				psiNVal = i[1]
				# FluxSur = self.getBs_FluxSur(psiNVal)
				
				R_hold = R[i[0],:]
				Z_hold = Z[i[0],:]
				
				Bp_hold = self.BpFunc.ev(Z_hold, R_hold)
				fpol_psiN = self.PROFdict['ffunc'](psiNVal)*np.ones(np.size(Bp_hold))
				fluxSur = eq.FluxSurface(fpol_psiN, R_hold, Z_hold, Bp_hold, self.theta)
				Bsqrd = fluxSur.Bsqav()

				FluxSur = {'Rs':R_hold, 'Zs':Z_hold, 'Bp':Bp_hold, 'Bt':fluxSur._Bt, 
						   'Bmod':fluxSur._B, 'Bsqrd':Bsqrd, 'fpol_psiN':fpol_psiN, 'FS':fluxSur}
					
				FluxSurList.append(FluxSur)

			return FluxSurList


		# --------------------------------------------------------------------------------
		# B-fields and its derivatives in 2-D poloidal plane given by (R,Z) grid
		def getBs_2D(self, FluxSurfList = None):
			RZdict = self.RZdict
			dBp_dZ,_ = np.gradient(self.Bp_Z, RZdict['dZ'], RZdict['dR'])
			dpsi_dRdZ,dBp_dR = np.gradient(self.Bp_R, RZdict['dZ'], RZdict['dR'])
	
			Bsqrd = np.ones(self.nw)
			dBpdR_hold = np.ones(self.thetapnts)
			dBpdZ_hold = np.ones(self.thetapnts)
			dpsidRdZ_hold = np.ones(self.thetapnts)
			Bp_R_hold = np.ones(self.thetapnts)
			Bp_Z_hold = np.ones(self.thetapnts)
			
			# np.size(PSIdict['psiN1D']) == self.nw
			Rs_hold2D = np.ones((self.nw,self.thetapnts))
			Zs_hold2D = np.ones((self.nw,self.thetapnts))
			Btot_hold2D = np.ones((self.nw,self.thetapnts))
			Bp_hold2D = np.ones((self.nw,self.thetapnts))
			Bt_hold2D = np.ones((self.nw,self.thetapnts))
			Bp_R_hold2D = np.ones((self.nw,self.thetapnts))
			Bp_Z_hold2D = np.ones((self.nw,self.thetapnts))
			dBpdR_hold2D = np.ones((self.nw,self.thetapnts))
			dBpdZ_hold2D = np.ones((self.nw,self.thetapnts))
			dpsidRdZ_hold2D = np.ones((self.nw,self.thetapnts))

			BpRFunc = interp.RectBivariateSpline(RZdict['Zs1D'],RZdict['Rs1D'],self.Bp_R,kx=1,ky=1)
			BpZFunc = interp.RectBivariateSpline(RZdict['Zs1D'],RZdict['Rs1D'],self.Bp_Z,kx=1,ky=1)
			dBpdRFunc = interp.RectBivariateSpline(RZdict['Zs1D'],RZdict['Rs1D'],dBp_dR,kx=1,ky=1)
			dBpdZFunc = interp.RectBivariateSpline(RZdict['Zs1D'],RZdict['Rs1D'],dBp_dZ,kx=1,ky=1)
			dpsidRdZFunc = interp.RectBivariateSpline(RZdict['Zs1D'],RZdict['Rs1D'],dpsi_dRdZ,kx=1,ky=1)

			if(FluxSurfList == None):
				FluxSurfList = self.get_allFluxSur()
				
			for i in enumerate(self.PSIdict['psiN1D']):
				R_hold = FluxSurfList[i[0]]['Rs']
				Z_hold = FluxSurfList[i[0]]['Zs']

				Bp_R_hold = BpRFunc.ev(Z_hold,R_hold)
				Bp_Z_hold = BpZFunc.ev(Z_hold,R_hold)
				dBpdR_hold = dBpdRFunc.ev(Z_hold,R_hold)
				dBpdZ_hold = dBpdZFunc.ev(Z_hold,R_hold)
				dpsidRdZ_hold = dpsidRdZFunc.ev(Z_hold,R_hold)

				Rs_hold2D[i[0],:] = R_hold
				Zs_hold2D[i[0],:] = Z_hold
				Bsqrd[i[0]] = FluxSurfList[i[0]]['Bsqrd']
				Btot_hold2D[i[0],:] = FluxSurfList[i[0]]['Bmod']
				Bp_hold2D[i[0],:] = FluxSurfList[i[0]]['Bp']
				Bt_hold2D[i[0],:] = FluxSurfList[i[0]]['Bt']
				Bp_R_hold2D[i[0],:] = Bp_R_hold
				Bp_Z_hold2D[i[0],:] = Bp_Z_hold
				dBpdR_hold2D[i[0],:] = dBpdR_hold
				dBpdZ_hold2D[i[0],:] = dBpdZ_hold
				dpsidRdZ_hold2D[i[0],:] = dpsidRdZ_hold

			return {'dpsidZ_2D':Bp_Z_hold2D,'dpsidR_2D':Bp_R_hold2D,'d2psidR2_2D':dBpdR_hold2D,'d2psidZ2_2D':dBpdZ_hold2D,
					'd2psidRdZ_2D':dpsidRdZ_hold2D,'Bp_2D':Bp_hold2D,'Bt_2D':Bt_hold2D,'Btot_2D':Btot_hold2D,
					'Rs_2D':Rs_hold2D,'Zs_2D':Zs_hold2D}
					
					
		# --------------------------------------------------------------------------------
		# normal and geodesic curvature, and local shear in 2-D plane
		def get_Curv_Shear(self, FluxSurfList = None, Bdict = None):				
			curvNorm_2D = np.ones((self.nw,self.thetapnts))
			curvGeo_2D = np.ones((self.nw,self.thetapnts))
			shear_fl = np.ones((self.nw,self.thetapnts))

			if(FluxSurfList == None):
				FluxSurfList = self.get_allFluxSur()
			if(Bdict == None): 
				Bdict = self.getB_2D(FluxSurfList)

			for i in enumerate(self.PSIdict['psiN1D']):
				psiNVal = i[1]
				fprint_psiN = self.PROFdict['fpfunc'](psiNVal)*np.ones(self.thetapnts)
				
				R = FluxSurfList[i[0]]['Rs']
				Bp = FluxSurfList[i[0]]['Bp']
				B = FluxSurfList[i[0]]['Bmod']
				fpol_psiN = FluxSurfList[i[0]]['fpol_psiN']		
				
				kapt1 = (fpol_psiN**2)*(Bdict['dpsidR_2D'][i[0],:])
				kapt2 = (Bdict['d2psidR2_2D'][i[0],:]*Bdict['dpsidZ_2D'][i[0],:]**2) + ((Bdict['dpsidR_2D'][i[0],:]**2)*Bdict['d2psidZ2_2D'][i[0],:])
				kapt3 = (2*Bdict['dpsidR_2D'][i[0],:]*Bdict['d2psidRdZ_2D'][i[0],:]*Bdict['dpsidZ_2D'][i[0],:])
				curvNorm_2D[i[0],:] = (kapt1 + Bdict['Rs_2D'][i[0],:]*(kapt2-kapt3))/(R**4 * Bp * B**2)
				
				kap2t1 = Bdict['d2psidRdZ_2D'][i[0],:]*(Bdict['dpsidR_2D'][i[0],:]**2 - Bdict['dpsidZ_2D'][i[0],:]**2)
				kap2t2 = Bdict['dpsidR_2D'][i[0],:]*Bdict['dpsidZ_2D'][i[0],:]*(Bdict['d2psidR2_2D'][i[0],:] - Bdict['d2psidZ2_2D'][i[0],:])
				kap2t3 = Bdict['dpsidZ_2D'][i[0],:] * R**2 * B**2
				curvGeo_2D[i[0],:] = -fpol_psiN*(Bdict['Rs_2D'][i[0],:]*kap2t1 - kap2t2 + kap2t3)/(R**5 * Bp * B**3)
			
				coeft1 = fpol_psiN/(R**4 * Bp**2 * B**2)
				coeft2 = ((Bdict['d2psidR2_2D'][i[0],:] - Bdict['d2psidZ2_2D'][i[0],:])*(Bdict['dpsidR_2D'][i[0],:]**2 - Bdict['dpsidZ_2D'][i[0],:]**2) +
							(4*Bdict['dpsidR_2D'][i[0],:]*Bdict['d2psidRdZ_2D'][i[0],:]*Bdict['dpsidZ_2D'][i[0],:]))
				sht2 = fpol_psiN*Bdict['dpsidR_2D'][i[0],:]/(R**3 * B**2)
				sht3 = fprint_psiN*Bp**2/(B**2)
				shear_fl[i[0],:] = coeft1*coeft2 + sht2 - sht3

			return {'curvNorm_2D':curvNorm_2D,'curvGeo_2D':curvGeo_2D,'localShear_2D':shear_fl}


		# --------------------------------------------------------------------------------
		# 1-D parallel current density profile
		def cur_density(self, FluxSurfList = None, get_jtor = True):
			import scipy.constants
			mu0 = scipy.constants.mu_0
			
			PSIdict = self.PSIdict
			PROFdict = self.PROFdict

			Bsqrd_prof = np.ones(self.nw)
			
			if(FluxSurfList == None):
				FluxSurfList = self.get_allFluxSur()

			for i, psi in enumerate(PSIdict['psiN1D']):
				Bsqrd_prof[i] = FluxSurfList[i]['Bsqrd']
											
			# parallel current calc
			# <jpar> = <(J (dot) B)>/B0 = (fprime*<B^2>/mu0 + pprime*fpol)/B0
			jpar1D = (PROFdict['fpfunc'](PSIdict['psiN1D'])*Bsqrd_prof/mu0 +PROFdict['ppfunc'](PSIdict['psiN1D'])*PROFdict['ffunc'](PSIdict['psiN1D']))/self.bcentr/1.e6
	
			# <jtor> = <R*pprime + ffprime/R/mu0>
			# jtor1D = np.abs(self.Rsminor*PROFdict['ppfunc'](PSIdict['psiN1D']) +(PROFdict['ffpfunc'](PSIdict['psiN1D'])/self.Rsminor/mu0))/1.e6
			if get_jtor:
				jtor1D = self. jtor_profile(FluxSurfList)
			else: 
				jtor1D = 0
			
			return {'jpar':jpar1D, 'jtor':jtor1D, 'Bsqrd':Bsqrd_prof}
			
		
		# --------------------------------------------------------------------------------
		# 1-D toroidal current density profile
		def jtor_profile(self, FluxSurfList = None):
			import scipy.constants
			mu0 = scipy.constants.mu_0
			
			PSIdict = self.PSIdict
			PROFdict = self.PROFdict

			jtor1D = np.ones(self.nw)
			
			if(FluxSurfList == None):
				FluxSurfList = self.get_allFluxSur()

			# get flux surface average
			for i, psi in enumerate(PSIdict['psiN1D']):
				jtorSurf = PROFdict['ppfunc'](psi)*FluxSurfList[i]['Rs'] + PROFdict['ffpfunc'](psi)/FluxSurfList[i]['Rs']/mu0
				f_jtorSurf = eq.interpPeriodic(self.theta, jtorSurf, copy = False)
				jtor1D[i] = FluxSurfList[i]['FS'].average(f_jtorSurf)

			# <jtor> = <R*pprime + ffprime/R/mu0>
			jtor1D = np.abs(jtor1D)/1.e6
			return jtor1D

			
		# --------------------------------------------------------------------------------
		# Flux surface enclosed 2D-Volume (Area inside surface) and derivative (psi)
		def volume2D(self, FluxSurfList = None):
			V = np.zeros(self.nw)
			psi = self.PSIdict['psiN1D']
			
			if(FluxSurfList == None):
				FluxSurfList = self.get_allFluxSur()
				
			for i in xrange(1, self.nw):
				rsq = (FluxSurfList[i]['Rs'] - self.rmaxis)**2 + (FluxSurfList[i]['Zs'] - self.zmaxis)**2
				V[i] = 0.5 * integ.simps(rsq, self.theta)
			
			# dV/dpsi
			dV = np.zeros(self.nw)
			for i in xrange(1, self.nw - 1):
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
			paramDICT['psitorN1D'] = self.getTorPsi(qprof1D)

			return paramDICT
		
		
		# --------------------------------------------------------------------------------
		# returns arrays R and Z of N points along psi = const. surface
		def flux_surface(self, psi0, N, theta = None):
			if(theta == None): 
				theta = np.linspace(0,2*np.pi,N + 1)[0:-1]
			else :
				N = len(theta)
				
			r = np.zeros(theta.shape)
	
			for i in xrange(N): 
				r[i] = self.__bisec__(psi0, theta[i])
	
			R = r*np.cos(theta) + self.rmaxis
			Z = r*np.sin(theta) + self.zmaxis
	
			return R, Z


		# --------------------------------------------------------------------------------
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
				f = psiFunc.ev(Zneu,Rneu) - psiNVal
				df = (psiFunc.ev(Zbac,Rbac) - psiFunc.ev(Zneu,Rneu))/(-1*h)
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
					print "bunk!"
					break
			return Rneu,Zneu
			
			
		# --------------------------------------------------------------------------------
		# returns 2D-arrays R and Z for all (theta, psi) points
		# theta and psi are 1D base arrays of a regular grid.
		def __get_RZ__(self, theta, psi, quiet = False):
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
			R_lcfs = self.data.get('rbbbs')
			Z_lcfs = self.data.get('zbbbs')
			r_lcfs = self.__get_r__(R_lcfs, Z_lcfs)[1::]		# here first == last, so skip first
			th_lcfs = self.__get_theta__(R_lcfs, Z_lcfs)[1::]	# here first == last, so skip first
			index = np.argsort(th_lcfs)	# th-lcfs needs to be monotonically increasing
	
			# interpolate to get r(theta) at lcfs
			get_r_lcfs = interp.UnivariateSpline(th_lcfs[index], r_lcfs[index], s = 0)
			r = get_r_lcfs(theta)
	
			# set initial guess of R & Z
			for i in xrange(npsi):
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
				psi_now[np.abs(psi_now) < eps] = 0		# check for small values (no negatives!)
				psi_now = psi_now.reshape(R.shape)
				rho_now = np.sqrt(psi_now)
			
				# store old values
				R_old = R.copy()
				Z_old = Z.copy()
		
				R -= self.rmaxis
				Z -= self.zmaxis
	
				# find new R, Z by interpolation
				for j in xrange(nt):
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
				ratio = (delta[-idx]/(r[-idx] + 1e-20)).max()
		
				# no convergence check
				N += 1
				if(N > max_it): 
					if not quiet:
						print 'Warning, iterate_RZ: bad convergence, check if error is acceptable'
						print 'Iteration: ', N, ', Error: ', ratio
					break
		
			return R, Z


		# --------------------------------------------------------------------------------
		# get poloidal angle from (R,Z) coordinates
		def __get_theta__(self, R, Z):
			Rm = R - self.rmaxis # R relative to magnetic axis
			Zm = Z - self.zmaxis # Z relative to magnetic axis
	
			Rm[(Rm < 1e-16) & (Rm >= 0)] = 1e-16
			Rm[(Rm > -1e-16) & (Rm < 0)] = -1e-16
	
			theta = np.arctan(Zm/Rm);
			if isinstance(theta, np.ndarray):
				theta[Rm < 0] += np.pi;
				theta[(Rm >= 0) & (Zm < 0)] += 2*np.pi;
			else:
				if(Rm < 0): theta += np.pi;
				if((Rm >= 0) & (Zm < 0)): theta += 2*np.pi;
	
			return theta;
	
	
		# --------------------------------------------------------------------------------
		# get minor radius from (R,Z) coordinates
		def __get_r__(self, R, Z):
			Rm = R - self.rmaxis # R relative to magnetic axis
			Zm = Z - self.zmaxis # Z relative to magnetic axis
	
			return np.sqrt(Rm*Rm + Zm*Zm);


		# --------------------------------------------------------------------------------
		# get f(r) = psi(R(theta,r),Z(theta,r))-psi0 with theta = const.
		def __funct__(self, r, theta, psi0):
			R = r*np.cos(theta) + self.rmaxis
			Z = r*np.sin(theta) + self.zmaxis
			psi = self.psiFunc.ev(Z, R)	
			f = psi - psi0
			return f


		# --------------------------------------------------------------------------------
		# get r for theta = const. and psi = psi0
		def __bisec__(self, psi0, theta, a = 0, b = 1.5):
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
				if(f > 0): xo = x
				else: xu = x
	
			return x

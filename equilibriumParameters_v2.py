import numpy as np
import scipy.interpolate as interp
import EFIT.geqdsk as gdsk
import EFIT.equilibrium as eq
import time

reload(eq)

g_fnam = 'g148712.04101'
#g_fnam = 'g149189.02400'

def equilParams(gfileNam,nw=0,nh=0,thetapnts=0,grid2G = False):
	import scipy.constants
	mu0 = scipy.constants.mu_0

	t0 = time.time()
		
	data = gdsk.Geqdsk()
	data.openFile(gfileNam)

	#---- Variables ----
	if(grid2G == True):
		nw = data.get('nw')
		nh = data.get('nh')
		thetapnts = 2*nh
	bcentr = np.abs(data.get('bcentr'))
	rmaxis = data.get('rmaxis')
	zmaxis = data.get('zmaxis')
	Rmin = data.get('rleft')
	Rmax = Rmin + data.get('rdim')
	Rbdry = data.get('rbbbs').max()
	Zmin = data.get('zmid') - data.get('zdim')/2.0
	Zmax = data.get('zmid') + data.get('zdim')/2.0
	Zlowest = data.get('zbbbs').min()
	
	siAxis = data.get('simag')
	siBry = data.get('sibry')
	
	#---- Profiles ----
	fpol = data.get('fpol')
	ffunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(fpol)),fpol,s=0)
	fprime = data.get('ffprime')/fpol
	fpfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(fprime)),fprime,s=0)
	ffprime = data.get('ffprime')
	ffpfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(ffprime)),fprime,s=0)
	pprime = data.get('pprime')
	ppfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(pprime)),pprime,s=0)
	pres = data.get('pres')
	pfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(pres)),pres,s=0)
	q_prof = data.get('qpsi')
	qfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(q_prof)),q_prof,s=0)
	
	g_psi2D = data.get('psirz')
	
	psiN1D = np.linspace(0.0,1.0,nw)
	Rsminor = np.linspace(rmaxis,Rbdry,nw)
	
	dR = (Rmax - Rmin)/np.float64(nw - 1)
	Rs1D = np.arange(Rmin, Rmax*(1.+1.e-10), dR)
	
	dZ = (Zmax - Zmin)/np.float64(nh - 1)
	Zs1D = np.arange(Zmin,Zmax*(1.+1.e-10), dZ)
	

	gRs,gZs = np.meshgrid(np.linspace(Rmin,Rmax,g_psi2D.shape[1]),np.linspace(Zmin,Zmax,g_psi2D.shape[0]))
	psi2D = interp.griddata((gRs.flatten(0),gZs.flatten(0)),g_psi2D.flatten(0),(Rs1D[:,None],Zs1D[None,:]),method='cubic',fill_value=0.0)
	psi2D = psi2D.T

	Bp_Z,Bp_R = np.gradient(-1.*psi2D,dZ,dR)

	dBp_dZ,_ = np.gradient(Bp_Z,dZ,dR)
	_,dBp_dR = np.gradient(Bp_R,dZ,dR)

	dpsi_dRdZ,_ = np.gradient(Bp_R,dZ,dR)
	
	Rs2D,Zs2D = np.meshgrid(Rs1D,Zs1D)
	Bp_2D = np.sqrt(Bp_R**2 + Bp_Z**2)/Rs2D
	
	psiN_2D = (psi2D - siAxis)/(siBry-siAxis)
	psiN_2D[np.where(psiN_2D > 1.2)] = 1.2
	
	theta = np.linspace(0.0,2.*np.pi,thetapnts)
	Bsqrd = np.copy(psiN1D)
	R_hold = theta.copy()
	Z_hold = theta.copy()
	Bp_hold = theta.copy()
	dBpdR_hold = theta.copy()
	dBpdZ_hold = theta.copy()
	dpsidRdZ_hold = theta.copy()
	Bp_R_hold = theta.copy()
	Bp_Z_hold = theta.copy()

	curvNorm_2D = np.ones((np.size(psiN1D),np.size(theta)))
	curvGeo_2D = np.ones((np.size(psiN1D),np.size(theta)))
	Rs_hold2D = np.ones((np.size(psiN1D),np.size(theta)))
	Zs_hold2D = np.ones((np.size(psiN1D),np.size(theta)))
	Btot_hold2D = np.ones((np.size(psiN1D),np.size(theta)))
	Bp_hold2D = np.ones((np.size(psiN1D),np.size(theta)))
	Bt_hold2D = np.ones((np.size(psiN1D),np.size(theta)))
	shear_fl = np.ones((np.size(psiN1D),np.size(theta)))

	psiFunc = interp.RectBivariateSpline(Zs1D,Rs1D,psiN_2D,kx=1,ky=1)
	BpFunc = interp.RectBivariateSpline(Zs1D,Rs1D,Bp_2D,kx=1,ky=1)
	BpRFunc = interp.RectBivariateSpline(Zs1D,Rs1D,Bp_R,kx=1,ky=1)
	BpZFunc = interp.RectBivariateSpline(Zs1D,Rs1D,Bp_Z,kx=1,ky=1)
	dBpdRFunc = interp.RectBivariateSpline(Zs1D,Rs1D,dBp_dR,kx=1,ky=1)
	dBpdZFunc = interp.RectBivariateSpline(Zs1D,Rs1D,dBp_dZ,kx=1,ky=1)
	dpsidRdZFunc = interp.RectBivariateSpline(Zs1D,Rs1D,dpsi_dRdZ,kx=1,ky=1)
	
	for i in enumerate(psiN1D):
		psiNVal = i[1]
		for thet in enumerate(theta):
			try:
				Rneu,Zneu = comp_newt(psiNVal,thet[1],rmaxis,zmaxis,psiFunc)
			except RuntimeError:
				Rneu,Zneu = comp_bisec(psiNVal,thet[1],rmaxis,zmaxis,Zlowest,psiFunc)
			R_hold[thet[0]] = Rneu
			Z_hold[thet[0]] = Zneu
			Bp_hold[thet[0]] = BpFunc.ev(Zneu,Rneu)
			Bp_R_hold[thet[0]] = BpRFunc.ev(Zneu,Rneu)
			Bp_Z_hold[thet[0]] = BpZFunc.ev(Zneu,Rneu)
			dBpdR_hold[thet[0]] = dBpdRFunc.ev(Zneu,Rneu)
			dBpdZ_hold[thet[0]] = dBpdZFunc.ev(Zneu,Rneu)
			dpsidRdZ_hold[thet[0]] = dpsidRdZFunc.ev(Zneu,Rneu)

		fpol_psiN = ffunc(psiNVal)*np.ones(np.size(Bp_hold))
		fprint_psiN = fpfunc(psiNVal)*np.ones(np.size(Bp_hold))
		fluxSur = eq.FluxSurface(fpol_psiN,R_hold,Z_hold,Bp_hold,theta)
		Rs_hold2D[i[0],:] = fluxSur._R
		Zs_hold2D[i[0],:] = fluxSur._Z
		Bsqrd[i[0]] = fluxSur.Bsqav()
		Btot_hold2D[i[0],:] = fluxSur._B
		Bp_hold2D[i[0],:] = fluxSur._Bp
		Bt_hold2D[i[0],:] = fluxSur._Bt
		kapt1 = (fpol_psiN**2)*(Bp_R_hold)
		kapt2 = (dBpdR_hold*Bp_Z_hold**2)+((Bp_R_hold**2)*dBpdZ_hold)
		kapt3 = (2*Bp_R_hold*dpsidRdZ_hold*Bp_Z_hold)
		curvNorm_2D[i[0],:] = (kapt1+R_hold*(kapt2-kapt3))/(fluxSur._R**4*fluxSur._Bp*fluxSur._B**2)
		kap2t1 = dpsidRdZ_hold*(Bp_R_hold**2-Bp_Z_hold**2)
		kap2t2 = Bp_R_hold*Bp_Z_hold*(dBpdR_hold-dBpdZ_hold)
		kap2t3 = Bp_Z_hold*fluxSur._R**2*fluxSur._B**2
		curvGeo_2D[i[0],:] = -fpol_psiN*(R_hold*kap2t1-kap2t2+kap2t3)/(fluxSur._R**5*fluxSur._Bp*fluxSur._B**3)
		coeft1 = fpol_psiN/(fluxSur._R**4*fluxSur._Bp**2*fluxSur._B**2)
		coeft2 = ((dBpdR_hold-dBpdZ_hold)*(Bp_R_hold**2-Bp_Z_hold**2)+(4*Bp_R_hold*dpsidRdZ_hold*Bp_Z_hold))
		sht2 = fpol_psiN*Bp_R_hold/(fluxSur._R**3*fluxSur._B**2)
		sht3 = fprint_psiN*fluxSur._Bp**2/(fluxSur._B**2)
		shear_fl[i[0],:] = coeft1*coeft2+sht2-sht3

	print Rs_hold2D[0,0], Zs_hold2D[0,0]
	# parallel current calc
	# jpar = J (dot) B = fprime*B^2/mu0 + pprime*fpol
	jpar1D = (fpfunc(psiN1D)*Bsqrd/mu0 +ppfunc(psiN1D)*ffunc(psiN1D))/bcentr/1.e6

	#jtor [A/m**2] = R*pprime +ffprime/R/mu0
	jtor1D = np.abs(Rsminor*ppfunc(psiN1D) +(ffpfunc(psiN1D)/Rsminor/mu0))/1.e6 

	#q profile
	qprof1D = qfunc(psiN1D)

	#pressure profile
	press1D = pfunc(psiN1D)

	#toroidal field profile
	btor1D = ffunc(psiN1D)/Rsminor

	#psitor from Canik's g3d.pro
	pn = np.arange(nw)/float((nw-1))
	dpsi = (pn[1]-pn[0])*(siBry - siAxis)
	hold = np.cumsum(0.5*(qprof1D[0:nw-1]+qprof1D[1:nw])*dpsi)
	psitor = np.concatenate((np.array([0.]),hold))
	psitorN1D = (psitor - psitor[0])/(psitor[nw-1]-psitor[0])

	paramDICT={}
	paramDICT['psitorN1D'] = psitorN1D
	paramDICT['psi2D'] = psi2D
	paramDICT['Rs1D'] = Rs1D
	paramDICT['Zs1D'] = Zs1D
	paramDICT['jpar1D'] = jpar1D
	paramDICT['jtor1D'] = jtor1D
	paramDICT['qprof1D'] = qprof1D
	paramDICT['press1D'] = press1D
	paramDICT['btor1D'] = btor1D
	paramDICT['psiN1D'] = psiN1D
	paramDICT['psiN_2D'] = psiN_2D
	paramDICT['theta'] = fluxSur._theta
	paramDICT['curvNorm_2D'] = curvNorm_2D
	paramDICT['curvGeo_2D'] = curvGeo_2D
	paramDICT['Rs_hold2D'] = Rs_hold2D
	paramDICT['Zs_hold2D'] = Zs_hold2D
	paramDICT['Btot_2D']=Btot_hold2D
	paramDICT['Bp_2D']=Bp_hold2D
	paramDICT['Bt_2D']=Bt_hold2D
	paramDICT['S_l_2D']=shear_fl

	t1 = time.time()
	print 'Time:', t1-t0
	return paramDICT

#figure();plot(psiN1D,jpar);axis([0.0,1,0,2]);

def comp_newt(psiNVal,theta,rmaxis,zmaxis,psiFunc,r_st = 0.5):
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

def comp_bisec(psiNVal,theta,rmaxis,zmaxis,Zlowest,psiFunc):
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
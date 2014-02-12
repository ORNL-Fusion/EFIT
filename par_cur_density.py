import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp
import geqdsk as gdsk
import equilibrium as eq

g_fnam = 'g148712.04101'
#g_fnam = 'g149189.02400'

def get_currs(gfileNam,nw=0,nh=0,thetapnts=0,grid2G=0,nakhdf=0):
	import scipy.constants
	mu0 = scipy.constants.mu_0

	data = gdsk.Geqdsk()
	data.openFile(gfileNam)

	#---- Variables ----
	if(grid2G == 1):
		nw = data.get('nw')
		nh = data.get('nh')
		thetapnts = nh
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
	
	dR = (Rmax - Rmin)/float(nw - 1)
	Rs1D = np.arange(Rmin, Rmax*(1.+1.e-10), dR)
	
	dZ = (Zmax - Zmin)/float(nh - 1)
	Zs1D = np.arange(Zmin, Zmax*(1.+1.e-10), dZ)
	
	gRs,gZs = np.meshgrid(np.linspace(Rmin,Rmax,g_psi2D.shape[1]),np.linspace(Zmin,Zmax,g_psi2D.shape[0]))
	psi2D = interp.griddata((gRs.flatten(0),gZs.flatten(0)),g_psi2D.flatten(0),(Rs1D[None,:],Zs1D[:,None]),method='cubic',fill_value=0.0)
	
	Bp_R,Bp_Z = np.gradient(psi2D,dR,dZ)
	
	Rs2D,Zs2D = np.meshgrid(Rs1D,Zs1D)
	Bp_2D = np.sqrt(Bp_R**2 + Bp_Z**2)/Rs2D
	
	psiN_2D = (psi2D - siAxis)/(siBry-siAxis)
	psiN_2D[np.where(psiN_2D > 1.2)] = 1.2
	
	theta = np.linspace(0.0,2.*np.pi,thetapnts)
	Bsqrd = np.copy(psiN1D)
	R_hold = np.copy(theta)
	Z_hold = np.copy(theta)
	Bp_hold = np.copy(theta)
	
	psiFunc = interp.RectBivariateSpline(Rs1D,Zs1D,psiN_2D.T,kx=1,ky=1)
	BpFunc = interp.RectBivariateSpline(Rs1D,Zs1D,Bp_2D.T,kx=1,ky=1)
	
	for i in enumerate(psiN1D):
		psiNVal = i[1]
		for thet in enumerate(theta):
			try:
				Rneu,Zneu = comp_newt(psiNVal,thet[1],rmaxis,zmaxis,psiFunc)
			except RuntimeError:
				Rneu,Zneu = comp_bisec(psiNVal,thet[1],rmaxis,zmaxis,Zlowest,psiFunc)
			R_hold[thet[0]] = Rneu
			Z_hold[thet[0]] = Zneu
			Bp_hold[thet[0]] = BpFunc.ev(Rneu,Zneu)
		fpol_psiN = ffunc(psiNVal)*np.ones(np.size(Bp_hold))
		fluxSur = eq.FluxSurface(fpol_psiN,R_hold,Z_hold,Bp_hold)
		Bsqrd[i[0]] = fluxSur.Bsqav()
	
	# parallel current calc
	# jpar = J (dot) B = fprime*B^2/mu0 + pprime*fpol
	jpar = (fpfunc(psiN1D)*Bsqrd/mu0 +ppfunc(psiN1D)*ffunc(psiN1D))/bcentr/1e6
	
	#jtor [A/m**2] = R*pprime +ffprime/R/mu0
	jtor = np.abs(Rsminor*ppfunc(psiN1D) +(ffpfunc(psiN1D)/Rsminor/mu0))/1.e6
	
	curDICT={}
	curDICT['jpar']=jpar
	curDICT['jtor']=jtor
	curDICT['psiN']=psiN1D
	return curDICT
	
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
		f = psiFunc.ev(Rneu,Zneu) - psiNVal
		df = (psiFunc.ev(Rbac,Zbac) - psiFunc.ev(Rneu,Zneu))/(-1*h)
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
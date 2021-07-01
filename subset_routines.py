import numpy as np
import time as systime
import scipy.interpolate as scinterp

#----------------------------------------------------------------------------
def antideriv1(x,y):
	x = np.float64(x)
	y = np.float64(y)
	nels = np.int32(np.size(y))
	integ = np.zeros(nels,dtype='float64')
	dh = np.float64(0.5)
	iter = np.array(np.linspace(1.,nels-1.),dtype='float64')
	for j in iter:
		integ[j] = integ[j-1]+dh*(x[j]-x[j-1])*(y[j]+y[j-1])
	return integ

#----------------------------------------------------------------------------
def definteg1(x,y):
	x = np.float64(x)
	y = np.float64(y)
	integ = antideriv1(x,y)
	integ = integ[np.size(y)-1]
	return integ

#----------------------------------------------------------------------------
#  Translated from Numerical Recipes by J. Menard 8/19/2000
#
def bcucof(y,y1,y2,y12,d1,d2):
	ny = np.size(y[:,0])
	x = np.zeros((ny,16),dtype='float64')
	w = np.zeros((16,16),dtype='float64')
	c = np.zeros((ny,4,4),dtype='float64')
	w[:,0] = [	1.,0.,-3.,2.,0.,0.,0.,0.,-3.,0.,9.,-6.,2.,0.,-6.,4.	]
	w[:,1] = [ 	0.,0.,0.,0.,0.,0.,0.,0.,3.,0.,-9.,6.,-2.,0.,6.,-4.	]
	w[:,2] = [ 	0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,9.,-6.,0.,0.,-6.,4.	]
	w[:,3] = [ 	0.,0.,3.,-2.,0.,0.,0.,0.,0.,0.,-9.,6.,0.,0.,6.,-4.	]
	w[:,4] = [ 	0.,0.,0.,0.,1.,0.,-3.,2.,-2.,0.,6.,-4.,1.,0.,-3.,2.	]
	w[:,5] = [ 	0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,3.,-2.,1.,0.,-3.,2.	]
	w[:,6] = [ 	0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-3.,2.,0.,0.,3.,-2.	]
	w[:,7] = [ 	0.,0.,0.,0.,0.,0.,3.,-2.,0.,0.,-6.,4.,0.,0.,3.,-2.	]
	w[:,8] = [	0.,1.,-2.,1.,0.,0.,0.,0.,0.,-3.,6.,-3.,0.,2.,-4.,2.	]
	w[:,9] = [ 	0.,0.,0.,0.,0.,0.,0.,0.,0.,3.,-6.,3.,0.,-2.,4.,-2.	]
	w[:,10] = [ 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-3.,3.,0.,0.,2.,-2.	]
	w[:,11] = [ 0.,0.,-1.,1.,0.,0.,0.,0.,0.,0.,3.,-3.,0.,0.,-2.,2.	]
	w[:,12] = [ 0.,0.,0.,0.,0.,1.,-2.,1.,0.,-2.,4.,-2.,0.,1.,-2.,1.	]
	w[:,13] = [ 0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,2.,-1.,0.,1.,-2.,1.	]
	w[:,14] = [	0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,-1.,0.,0.,-1.,1.	]
	w[:,15] = [ 0.,0.,0.,0.,0.,0.,-1.,1.,0.,0.,2.,-2.,0.,0.,-1.,1.	]
	x[:,0:4] = y
	x[:,4:8] = y1*d1
	x[:,8:12] = y2*d2
	x[:,12:16] = y12*d1*d2
	x = np.dot(x,w.T)
	c = np.transpose(np.reshape(x,(ny,4,4)),axes=[0,2,1])

	return c

#----------------------------------------------------------------------------
#  Translated from Numerical Recipes by J. Menard 8/19/2000
#
def bcuint(y,y1,y2,y12,x1l,x1u,x2l,x2u,x1,x2):
	etime = systime.time()
	ny	= np.size(y[:,0])
	d1	= np.zeros((ny,4),dtype='float64')
	d2	= np.zeros((ny,4),dtype='float64')
	for i in range(3):
		d1[:,i] = x1u-x1l
	for i in range(3):
		d2[:,i] = x2u-x2l
	c = bcucof(y,y1,y2,y12,d1,d2)
	t = (x1-x1l) / (x1u-x1l)
	u = (x2-x2l) / (x2u-x2l)
	ansy = np.zeros(ny,dtype='float64')
	ansy2 = np.zeros(ny,dtype='float64')
	ansy1 = np.zeros(ny,dtype='float64')

	for i in np.arange(3,0,-1):
  		ansy = t*ansy  + ((  c[:,i,3]*u+   c[:,i,2])*u+c[:,i,1])*u+c[:,i,0]
  		ansy2 = t*ansy2 + (3.*c[:,i,3]*u+2.*c[:,i,2])*u+c[:,i,1]
  		ansy1 = u*ansy1 + (3.*c[:,3,i]*t+2.*c[:,2,i])*t+c[:,1,i]

	ansy1 = ansy1/(x1u-x1l)
	ansy2 = ansy2/(x2u-x2l)

	return ansy,ansy1,ansy2

#----------------------------------------------------------------------------
#   U		= FUNCTION value U(x1,x2) on uniform 2D mesh
#   dUdx1	= First derivative of U with respect to x1 at mesh points
#   dUdx2	= First derivative of U with respect to x2 at mesh points
#   d2Udx1dx2	= Cross derivative of U with respect to x1 and x2 on mesh
#   x1v		= 1D vector of x1 values of mesh
#   x2v		= 1D vector of x2 values of mesh
#   x1in	= 1D vector of x1 values at which to perform interpolation
#   x2in	= 1D vector of x2 values at which to perform interpolation
#
def bicubic_interpolate(U,dUdx1,dUdx2,d2Udx1dx2,x1v,x2v,x1in,x2in):
	nx	= np.size(x1in)
	y 	= np.zeros((nx,4),dtype='float64')
	y1 	= np.zeros((nx,4),dtype='float64')
	y2 	= np.zeros((nx,4),dtype='float64')
	y12 = np.zeros((nx,4),dtype='float64')
	nx1	= np.size(x1v)
	nx2	= np.size(x2v)
	x1min = np.min(x1v)
	x2min = np.min(x2v)
	dx1	= x1v[1]-x1v[0]
	dx2	= x2v[1]-x2v[0]
	m	= nx1-2
	n	= nx2-2
	i0 	= np.int32((x1in-x1min)/dx1)
	j0 	= np.int32((x2in-x2min)/dx2)
	i0	= i0*np.greater_equal(i0,0)						#Set - indices to 0
	i0	= i0*np.less_equal(i0,m) + m*np.greater(i0,m) 	# Set indices > m = m
	j0	= j0*np.greater_equal(j0,0)						#Set - indices to 0
	j0	= j0*np.less_equal(j0,n) + n*np.greater(j0,n) 	# Set indices > n = n
	i1  = i0+1
	j1  = j0+1
	x1 	= x1in
	x2 	= x2in
	x1l	= np.array(x1v[i0],dtype='float64')
	x1u	= x1v[i1]
	x2l	= x2v[j0]
	x2u	= x2v[j1]
	y[:,0]	= U[i0,j0]
	y[:,1] 	= U[i1,j0]
	y[:,2] 	= U[i1,j1]
	y[:,3]  = U[i0,j1]
	y1[:,0] = dUdx1[i0,j0]
	y1[:,1] = dUdx1[i1,j0]
	y1[:,2] = dUdx1[i1,j1]
	y1[:,3] = dUdx1[i0,j1]
	y2[:,0] = dUdx2[i0,j0]
	y2[:,1]	= dUdx2[i1,j0]
	y2[:,2]	= dUdx2[i1,j1]
	y2[:,3] = dUdx2[i0,j1]
	y12[:,0] = d2Udx1dx2[i0,j0]
	y12[:,1] = d2Udx1dx2[i1,j0]
	y12[:,2] = d2Udx1dx2[i1,j1]
	y12[:,3] = d2Udx1dx2[i0,j1]

	iU,idUdx1,idUdx2 = bcuint(y,y1,y2,y12,x1l,x1u,x2l,x2u,x1,x2)
	return {'U':iU,'dUdx1':idUdx1,'dUdx2':idUdx2}

#----------------------------------------------------------------------------
# Transforms dx,dy in cartesian coordinates into polar coordinates
#
def rho_angle(dx,dy,**kwds):
	
	ang = np.array(np.arctan2(dy,dx))
	rho = np.sqrt(dx**2+dy**2)
	js 	= np.argsort(ang,axis=0)
	tpi	= np.float64(2.0*np.pi)

	if 'periodic' in kwds:
  		Is,ind 	= np.unique(ang,return_index=True)
  		Is = Is[ind]
  		rho	= rho[Is]
  		ang	= ang[Is]
  		nis	= np.size(Is)
  		rho	= [rho[nis-1],rho,rho[0]]
  		ang	= [ang[nis-1]-tpi,ang,ang[0]+tpi]

	if 'extend' in kwds:
		Is,ind 	= np.unique(ang,return_index=True)
		Is = Is[ind]
		rho	= rho[Is]
		ang	= ang[Is]
		nis	= np.size(Is)
		j	= np.arange(nis-1,dtype='int32')
		k	= j+1
		rho = [rho[j],    rho,rho[k]    ]
		ang = [ang[j]-tpi,ang,ang[k]+tpi]

	return {'rho':rho,'angle':ang,'isort':js}

#========================================================================
#
#  The following routines are for general post-processing of
#  g-structure data.  They are used primarily for computing which
#  array indices are enclosed by the plasma bounary and for refining
#  the boundary data.
#
#========================================================================
#
# Reforms the raw flux and R,Z grids from EFIT
#
def psistruc(s):
	nr	= s['nw']
	nz	= s['nh']
	ri	= np.arange(nr,dtype='int32')
	zi	= np.arange(nz,dtype='int32')
	i	= np.arange(nr*nz,dtype='int32') % nr
	j	= np.arange(nr*nz,dtype='int32') / nr
	rv	= s['r'][ri]
	zv	= s['z'][zi]
	psi	= np.array(s['psirz'][0:nr,0:nz],dtype='float64')
	r	= np.array(np.reshape(rv[i],(nr,nz)),dtype='float64')
	z	= np.array(np.reshape(zv[i],(nr,nz)),dtype='float64')

	return {'psi':psi,'r':r,'z':z,'nr':nr,'nz':nz}

#------------------------------------------------------------------------
# Translate IDL DERIV function
# Perform numerical differentiation using 3-point, Lagrangian interpolation.
def deriv(y,*x):
	n = np.size(y)
	if n < 3:
		print ('Parameters must have at least 3 points')
	if (len(x) > 1):
		if (n != np.size(x)):
			print ('Vectors must have same size')
		d = np.array((np.roll(y,-1) - np.roll(y,1))/(np.roll(x,-1) - np.roll(x,1)),dtype='float64')
		d[0] = (-3.0*y[0] + 4.0*y[1] - y[2])/(x[2]-x[0])
		d[n-1] = (3.0*y[n-1] - 4.*y[n-2] + y[n-3])/(x[n-1]-x[n-3])
	else:
		d = (np.roll(y,-1) - np.roll(y,1))/2.
		d[0] = (-3.0*y[0] + 4.0*y[1] - y[2])/2.
		d[n-1] = (3.*y[n-1] - 4.*y[n-2] + y[n-3])/2.
	return d

#------------------------------------------------------------------------
# Computes Grad-psi on solution grid for subsequent 2D interpolation
#
def gradpsi(psirz):
	psi		= psirz['psi']
	r		= psirz['r']
	z		= psirz['z']
	nr		= psirz['nr']
	nz		= psirz['nz']
	dpsidr	= np.zeros((nr,nz),dtype='float64')
	dpsidz	= np.zeros((nr,nz),dtype='float64')
	d2psidrdz= np.zeros((nr,nz),dtype='float64')
	for i in range(nz-1):
		dpsidr[:,i] = deriv(psi[:,i],r[:,i])
	for j in range(nr-1):
		dpsidz[j,:] = deriv(psi[j,:],z[j,:])
	for j in range(nr-1):
		d2psidrdz[j,:] 	= deriv(dpsidr[j,:],z[j,:])
	return {'dpsidr':dpsidr,'dpsidz':dpsidz,'d2psidrdz':d2psidrdz}

#------------------------------------------------------------------------
# Given raw boundary information from EFIT, re-order the boundary
# points and interpolate in theta
#------------------------------------------------------------------------
# Another routine which computes psi and derivative data on solution grid
#
def psidata(psirz):
	psi	= psirz['psi']
	r	= psirz['r']
	z	= np.transpose(psirz['z'])
	rv	= r[0,:].squeeze()
	zv	= z[:,0].squeeze()
	nr	= psirz['nr']
	nz	= psirz['nz']
	dr	= r[1,0]-r[0,0]
	dz	= z[0,1]-z[0,0]
	rmin= r.min()
	zmin= z.min()
	gs= gradpsi(psirz)

	return	{'psi':psi,'r':r,'z':z,'nr':nr,'nz':nz,'dr':dr,'dz':dz,'rmin':rmin,'zmin':zmin,
				'rv':rv,'zv':zv,'dpsidr':gs['dpsidr'],'dpsidz':gs['dpsidz'],'d2psidrdz':gs['d2psidrdz']}

#------------------------------------------------------------------------
# Compute flux surface average of "A" using Jacobian "J" & angle "theta"
# on a single flux surface (i.e. A, J, and theta are 1D theta arrays)
#
def fs_average(A, J, theta):
	return definteg1(theta,A*J)/definteg1(theta,J)

#------------------------------------------------------------------------
# Compute flux surface average of a 2D theta x psi array A
#
def fs_average_array(A, J, theta):
	nr	= np.size(A[0,:])
	result	= np.zeros(nr,dtype='float64')
	for k in range(nr-1):
		result[k] = fs_average(A[:,k],J[:,k],theta)
	return result.squeeze()

#------------------------------------------------------------------------
# Compute flux surface average of a 2D theta x psi array A using efc
#
def fs_average_array_efc(A, efc):
	nr	= np.size(A[0,:])
	result	= np.zeros(nr,dtype='float64')
	J	= efc['Jacobian']
	theta	= efc['theta']
	for k in range(nr-1):
		result[k] = fs_average(A[:,k],J[:,k],theta)
	return result.squeeze()

#------------------------------------------------------------------------
#  Given a 2D theta x psi array (A) and the equilibrium data returned by
#  "efitg_flux_coords" (efc), compute volume integral A*dV on each flux
#  surface
#
def volume_integrate(A,efc):
	nt	= efc['nt']
	nr	= efc['nr']
	J	= efc['Jacobian']
	theta	= efc['theta']
	psiv	= efc['psivec']
	tmp	= np.zeros(nr,dtype='float64')
	for i in range(nr-1):
		tmp[i] = definteg1(theta,A[:,i]*Jac[:,i])
	IntdV	= np.float64(2.0*np.pi)*antideriv1(psiv,tmp)
	return IntdV

#------------------------------------------------------------------------
#  Compute volume integral A*dV from axis to boundary
#
def volume_integral(A,efc,*average):
	tmp	= volume_integrate(A,efc)
	val	= tmp[np.size(tmp)-1]
	if(average):
		val = val / efc['vloume'].max()
	return val

#------------------------------------------------------------------------
#  Compute volume average of profile f(psin)*dV from axis to boundary
#
def volume_integral_profile(f,efc,*average):
	nt	= efc['nt']
	nr	= efc['nr']
	A = np.zeros((nt,nr),dtype='float64')
	for i in range(nt-1):
		A[i,:] = f
	tmp	= volume_integrate(A,efc)
	val	= tmp[np.size(tmp)-1]
	if(average):
		val = val / efc['vloume'].max()
	return val

#------------------------------------------------------------------------
#  Compute cross-sectional area integral A.dA on each flux surface
#
def area_integrate(A,efc):
	nt	= efc['nt']
	nr	= efc['nr']
	Jac	= efc['Jacobian']
	theta	= efc['theta']
	psiv	= efc['psivec']
	R	= efc['x']
	tmp	= np.zeros(nr,dtype='float64')
	for i in range(nr-1):
		tmp[i] = definteg1(theta,A[:,i]*Jac[:,i]/R[:,i])
	IntdA	= antideriv1(psiv,tmp)
	return IntdA

#------------------------------------------------------------------------
# Given raw boundary information from EFIT, re-order the boundary
# points and interpolate in theta

def fit_boundary(xb,zb,nthe,*udsym,**uniform):
	x	= np.float64(xb)
	z	= np.float64(zb)
	isymd	= 0
	if(udsym):
		isymd=1

	#--- Compute poloidal angle of x,z coordinates
	x0	= (x.max()+x.min())/2.0
	z0	= (z.max()+z.min())/2.0
	kappa	= (z.max()-z.min())/(x.max()-x.min())
	znorm	= 1./kappa
	angle	= (np.arctan2((z-z0)*znorm,x-x0)+(2.0*np.pi)) % (2.0*np.pi)

	#--- Sort unique angle data to match J-Solver indexing
	_,isort	= np.unique(angle,return_index=True)
	angle	= angle[isort]
	x	= x[isort]
	z	= z[isort]

	#--- Shift angle and add point to close interval [0,2*PI]
	angle	= angle-angle[0]
	angle	= np.append(angle,2.0*np.pi)
	x	= np.append(x,x[0])
	z	= np.append(z,z[0])

	#--- Spline fit boundary data to [0,2*pi] interval
	theta	= np.arange(nthe,dtype='float64')/np.float64(nthe-1)*2.0*np.pi
	#xspline	= scinterp.UnivariateSpline(x,angle)
	#xfit	= xspline(theta)
	#zspline	= scinterp.UnivariateSpline(z,angle)
	#zfit	= zspline(theta)
	xinterp	= scinterp.interp1d(angle,x,kind='quadratic')
	xfit	= xinterp(theta)
	zinterp	= scinterp.interp1d(angle,z,kind='quadratic')
	zfit	= zinterp(theta)
	#xinterp	= scinterp.interp1d(x,angle)
	#xfit	= xinterp(theta)
	#zinterp	= scinterp.interp1d(z,angle)
	#zfit	= zinterp(theta)

	#--- Force coordinates of Min and Max Z to be included in fit
	#
	if not(uniform):
		izmin	= np.where(z == z.min())[0]
		izmax	= np.where(z == z.max())[0]
		anglen	= angle[izmin]
		anglex	= angle[izmax]
		distn	= np.sqrt((xfit-x[izmin])**2 + (zfit-z[izmin])**2)
		distx	= np.sqrt((xfit-x[izmax])**2 + (zfit-z[izmax])**2)
		imin	= np.where(distn == distn.min())[0]
		imax	= np.where(distx == distx.min())[0]
		xfit[imin] 	= x[izmin]
		zfit[imin] 	= z[izmin]
		theta[imin]	= anglen
		xfit[imax] 	= x[izmax]
		zfit[imax] 	= z[izmax]
		theta[imax]	= anglex

	#--- Force same physical location to have same X
	xend = (xfit[0]+xfit[nthe-1])/2.0
	xfit[0] = xend
	xfit[nthe-1] = xend

	#--- Force same physical location to have same Z
	zend = (zfit[0]+zfit[nthe-1])/2.0
	zfit[0] = zend
	zfit[nthe-1] = zend

	#--- Choose between standard fit and symmetrized fit
	xsym 	= (xfit+xfit[::-1])/2
	zsym 	= (zfit-zfit[::-1])/2
	if(isymd == 1):
		xfit = xsym
	if(isymd == 1):
		zfit = zsym
	if(isymd == 1):
		print ('		')
		print ('Boundary fit forced to be up/down symmetric.')
		print ('		')

	return {'xfit':xfit,'zfit':zfit,'nthe':nthe,'angle':angle,'theta':theta}

#----------------------------------------------------------------------------
# Refines the position of a poloidal field null using bi-cubic interpolation
# Two methods are used simultaneously to find both X-points and O-points.
#=============================================
# Define common variables
def intpsi(psidata):
	U = psidata['psi']
	dUdx1 = psidata['dpsidr']
	dUdx2 = psidata['dpsidz']
	d2Udx1dx2 = psidata['d2psidrdz']
	x1v = psidata['rv']
	x2v = psidata['zv']
	return U,dUdx1,dUdx2,d2Udx1dx2,x1v,x2v
#=============================================
def refine_null_position(rin,zin,psidata,*quiet,**kwds):
	# optional kwds = {ixpoint,iopoint}
	
	ixpoint	= np.int32(-1)

	maxiters= 400
	p = psidata
	U,dUdx1,dUdx2,d2Udx1dx2,x1v,x2v = intpsi(p)

	delfac	= np.float64(1.0e-4)
	dl 	= np.sqrt(p['dr']**2 + p['dz']**2)*delfac
	lconv	= dl/np.sqrt(delfac)
	rnew1   = rin
	znew1   = zin
	rnew2   = rin
	znew2   = zin
	icount	= 0
	if not (quiet):
		verbose	= True

#---
# Once a null has been identified, compute B around null to
# identify as X-point or O-point.
#
	ncheck	= 201
	theta	= np.zeros(ncheck,dtype='float64')/np.float64(ncheck)*2.0*np.pi
	sinthe	= np.sin(theta)
	costhe	= np.cos(theta)
	#dlchkn	= 2.0
	dlchkn	= 5.0
	drchk	= dlchkn*lconv*costhe
	dzchk	= dlchkn*lconv*sinthe

	itry1	= 1
	itry2	= 1

#PRINT,'-------------------------------------->'

	t0 = systime.time()

	while 1:
		icount	= icount+1

		# METHOD 1 FOR FINDING NULL
		if (itry1 == 1):
			# Compute gradients of psi at new R,Z
			ip1	= interpolate_psi_common(rnew1,znew1,p)
			dpsidr1 = ip1['dpsidr']
			dpsidz1 = ip1['dpsidz']
			RBp1    = np.sqrt(dpsidr1**2+dpsidz1**2)
			f1      = RBp1
			# Compute R derivative of |grad-psi| at new R,Z
			ipp1	= interpolate_psi_common(rnew1+dl,znew1,p)
			ipm1	= interpolate_psi_common(rnew1-dl,znew1,p)
			RBpp1 	= np.sqrt(ipp1['dpsidr']**2+ipp1['dpsidz']**2)
			RBpm1 	= np.sqrt(ipm1['dpsidr']**2+ipm1['dpsidz']**2)
			dRBpdR1 = (RBpp1-RBpm1)/np.float64(2.0*dl)
			#Compute Z derivative of |grad-psi| at new R,Z
			ipp1	= interpolate_psi_common(rnew1,znew1+dl,p)
			ipm1	= interpolate_psi_common(rnew1,znew1-dl,p)
			RBpp1 	= np.sqrt(ipp1['dpsidr']**2+ipp1['dpsidz']**2)
			RBpm1 	= np.sqrt(ipm1['dpsidr']**2+ipm1['dpsidz']**2)
			dRBpdZ1 = (RBpp1-RBpm1)/np.float64(2.0*dl)

			t1 = dRBpdR1
			t2 = dpsidr1
			t3 = dRBpdZ1
			t4 = dpsidz1
			alpha1 	= f1/(t1*t2+t3*t4)

			dr1 = -alpha1 * dpsidr1
			dz1 = -alpha1 * dpsidz1

			rnew1	= rnew1 + dr1
			znew1	= znew1 + dz1
			dlnew1	= np.sqrt(dr1**2 + dz1**2)

			if(rnew1 > p['rv'].max()):
				itry1 = 0
			if(znew1 > p['zv'].max()):
				itry1 = 0
			if(rnew1 < p['rv'].min()):
				itry1 = 0
			if(znew1 < p['zv'].min()):
				itry1 = 0

			if(dlnew1 < lconv):
				rin = rnew1
				zin = znew1

				rchk	= rin[0]+drchk
				zchk	= zin[0]+dzchk
				ipfin2	= interpolate_psi_common(rchk,zchk,p)
				chkvar	= costhe*ipfin2['dpsidr']+sinthe*ipfin2['dpsidz']
				chkvars	= np.sign(chkvar/np.mean(np.abs(chkvar)))
				ichange	= np.where(chkvars != chkvars[0])
				ixpoint	= 0
				if(ichange[0] != -1):
					ixpoint = 1
				iopoint	= 1-ixpoint

				fmtnll1= "Found possible poloidal field {}-point at (R,Z) [m]: ({:9.5f},{:9.5f})"
				if ('ixpoint' in kwds) and verbose:
					print (fmtnll1.format('X',rin,zin))
				if ('iopoint' in kwds) and verbose:
					print (fmtnll1.format('O',rin,zin))
				return rin,zin

		# METHOD 2 FOR FINDING NULL
		if(itry2 == 1):
			ip2	= interpolate_psi_common(rnew2,znew2,p)
			dpsidr2	= ip2['dpsidr']
			dpsidz2	= ip2['dpsidz']
			RBp2    = np.sqrt(dpsidr2**2+dpsidz2**2)
			f2    	= RBp2

			ipp2	= interpolate_psi_common(rnew2+dl,znew2,p)
			ipm2	= interpolate_psi_common(rnew2-dl,znew2,p)
			RBpp2 	= np.sqrt(ipp2['dpsidr']**2+ipp2['dpsidz']**2)
			RBpm2 	= np.sqrt(ipm2['dpsidr']**2+ipm2['dpsidz']**2)
			dRBpdR2 = (RBpp2-RBpm2)/np.float64(2.0*dl)

			ipp2	= interpolate_psi_common(rnew2,znew2+dl,p)
			ipm2	= interpolate_psi_common(rnew2,znew2-dl,p)
			RBpp2 	= np.sqrt(ipp2['dpsidr']**2+ipp2['dpsidz']**2)
			RBpm2 	= np.sqrt(ipm2['dpsidr']**2+ipm2['dpsidz']**2)
			dRBpdZ2 	= (RBpp2-RBpm2)/np.float64(2.0*dl)

			t1 	= dRBpdR2
			t2 	= dpsidr2
			t3 	= dRBpdZ2
			t4 	= dpsidz2
			alpha2	= f2/(t1*t1+t3*t3)

			dr2   	= -alpha2 * dRBpdR2
			dz2   	= -alpha2 * dRBpdZ2

			rnew2	= rnew2 + dr2
			znew2 	= znew2 + dz2
			dlnew2 = np.sqrt(dr2**2 + dz2**2)

			if(rnew2 > p['rv'].max()):
				itry2 = 0
			if(znew2 > p['zv'].max()):
				itry2 = 0
			if(rnew2 < p['rv'].min()):
				itry2 = 0
			if(znew2 < p['zv'].min()):
				itry2 = 0
			if(dlnew2 < lconv):
				rin 	= rnew2
				zin 	= znew2

				rchk	= rin[0]+drchk
				zchk	= zin[0]+dzchk
				ipfin2	= interpolate_psi_common(rchk,zchk,p)
				chkvar	= costhe*ipfin2['dpsidr']+sinthe*ipfin2['dpsidz']
				chkvars	= np.sign(chkvar/np.mean(np.abs(chkvar)))
				ichange	= np.where(chkvars != chkvars[0])
				ixpoint	= 0
				if(ichange[0] != -1):
					ixpoint = 1
				iopoint	= 1-ixpoint
				fmtnll2= "Found possible poloidal field {}-point at (R,Z) [m]: ({:9.5f},{:9.5f})"
				if ('ixpoint' in kwds) and verbose:
					print (fmtnll2.format('X',rin,zin))
				if ('iopoint' in kwds) and verbose:
					print (fmtnll2.format('O',rin,zin))
				#xyz = ''
				#read,'Enter <s> to stop: ',xyz
				#if(xyz == 's'):
				#	stop
				return
		if((itry1 == 0) and (itry2 == 0)):
			return rin,zin
		if(icount > maxiters):
			if verbose:
				print ('Maximum number of iterations exceeded.')
			break
	return

#========================================================================
# 	The following functions compute flux-coordinate equilibria from
#	the R,Z grid data, and time derivatives and moments of plasma
#	quantities on the flux-coordinate grid.
#========================================================================
# Computes the interpolated flux function at R,Z using psi data
# stored in the common block
def interpolate_psi_common(r,z,p):
	U,dUdx1,dUdx2,d2Udx1dx2,x1v,x2v=intpsi(p)
	bi = bicubic_interpolate(U,dUdx1,dUdx2,d2Udx1dx2,x1v,x2v,r,z)
	return {'psi':bi['U'],'dpsidr':bi['dUdx1'],'dpsidz':bi['dUdx2']}

#----------------------------------------------------------------------------
# Computes the interpolated flux function at R,Z given psi-structure p
#
def interpolate_psi(r,z,p):
	U,dUdx1,dUdx2,d2Udx1dx2,x1v,x2v=intpsi(p)
	bi = bicubic_interpolate(U,dUdx1,dUdx2,d2Udx1dx2,x1v,x2v,r,z)
	return {'psi':bi['U'],'dpsidr':bi['dUdx1'],'dpsidz':bi['dUdx2']}

def interpolate_psi_only(r,z,psidata):
	psis = interpolate_psi(r,z,psidata)
	return psis['psi']

#-----------------------------------------------------------------------------------
#  Given the 2D arrays X & Z and 1D arrays theta and psi, compute the flux
#  coordinate Jacobian and related derivatives on the X,Z grid.  Arrays must repeat
#  themselves in the theta direction spanning [0,2PI] (i.e. strictly periodic).
#  Del-star psi on the grid is optionally computed with the "delstar_psi" keyword.
def flux_coordinate_jacobian(x,z,thev,psiv,*delstar_psi):
	#--- Convert input to double precision
	small	= np.float64(1.0e-50)
	dx	= np.array(x,dtype='float64')
	dz	= np.array(z,dtype='float64')
	the	= np.array(thev,dtype='float64')
	psi	= np.array(psiv,dtype='float64')

	#--- Define 2D arrays
	nthe	= np.size(the)
	npsi	= np.size(psi)
	xthe	= np.zeros((nthe,npsi),dtype='float64')
	zthe	= np.zeros((nthe,npsi),dtype='float64')
	xrho	= np.zeros((nthe,npsi),dtype='float64')
	zrho	= np.zeros((nthe,npsi),dtype='float64')
	xpsi	= np.zeros((nthe,npsi),dtype='float64')
	zpsi	= np.zeros((nthe,npsi),dtype='float64')
	rhoarr	= np.zeros((nthe,npsi),dtype='float64')
	delstar	= np.zeros((nthe,npsi),dtype='float64')

	#--- Estimate psi at axis with extrapolation
	meanx0	= np.mean(x[0:nthe-2,0])
	meanz0	= np.mean(z[0:nthe-2,0])
	distx	= np.sqrt((x[0,:]-meanx0)**2+(z[0,:]-meanz0)**2)
	diste	= np.sqrt((x[0,npsi-1]-meanx0)**2+(z[0,npsi-1]-meanz0)**2)

	psi0interp = scinterp.InterpolatedUnivariateSpline(psi,distx**2)
	psi0ex	= psi0interp(np.float64(0.0))

	# Define arrays for radial derivatives
	#dpsi	= psi[npsi-1]-psi[0]
	dpsi	= psi[npsi-1]-psi0ex
	tdpsi	= np.float64(2.0*dpsi)
	psin	= (psi-psi0ex)/dpsi
	rho	= np.sqrt(psin)
	if(rho[0] < small):
		rho[0] = small

	#--- rhoarr = 2D rho array
	for i in range(nthe-1):
		rhoarr[i,:] = rho

	#--- Compute theta derivatives of X and Z
	for j in range(npsi-1):
		xthe[:,j] = deriv(the,dx[:,j])
	for j in range(npsi-1):
		zthe[:,j] = deriv(the,dz[:,j])

	#--- Correct theta derivative at end-points
	thep	= [the[0]-(the[nthe-1]-the[nthe-2]),the[0],the[1]]
	dxthe	= np.ones((3,npsi),dtype='float64')
	dzthe	= np.ones((3,npsi),dtype='float64')
	for j in range(npsi-1):
		dxthe[:,j] = deriv(thep,dx[[nthe-2,0,1],j])
	for j in range(npsi-1):
		dzthe[:,j] = deriv(thep,dz[[nthe-2,0,1],j])
	xthe[0,:]	= dxthe[1,:]
	zthe[0,:]	= dzthe[1,:]
	xthe[nthe-1,:]	= dxthe[1,:]
	zthe[nthe-1,:]	= dzthe[1,:]

	#--- Compute renormalized theta derivatives
	xtheh	= xthe/rhoarr
	ztheh	= zthe/rhoarr
	#--- Extrapolate renormalized theta derivatives into axis
	xtheh[:,0] = 2.0*xtheh[:,1]-xtheh[:,2]
	ztheh[:,0] = 2.0*ztheh[:,1]-ztheh[:,2]
	#--- Compute rho derivatives
	for i in range(nthe-1):
		xrho[i,:] = deriv(rho,dx[i,:])
	for i in range(nthe-1):
		zrho[i,:] = deriv(rho,dz[i,:])

	#--- Compute psi derivatives
	xpsi	= xrho/tdpsi/rhoarr
	zpsi	= zrho/tdpsi/rhoarr

	#--- Compute the Jacobian
	det	= (xrho*ztheh-xtheh*zrho)/tdpsi
	jac 	= dx*det
	#--- Compute moments of the Jacobian
	Bp2	= (xthe**2+zthe**2)/jac**2
	iJdt	= np.ones(npsi,dtype='float64')
	iJBp2dt	= np.ones(npsi,dtype='float64')
	iJRm2dt	= np.ones(npsi,dtype='float64')
	for j in range(npsi-1):
		iJdt[j] = definteg1(the,jac[:,j])
	for j in range(npsi-1):
		iJBp2dt[j] = definteg1(the,jac[:,j]*Bp2[:,j])
	for j in range(npsi-1):
		iJRm2dt[j] = definteg1(the,jac[:,j]/x[:,j]**2)
	mom	= {'iJdt':iJdt,'iJBp2dt':iJBp2dt,'iJRm2dt':iJRm2dt}

	if(delstar_psi):
		#--- Compute rho derivatives for delstar-psi
		rdr	= np.ones((nthe,npsi),dtype='float64')
		zdr	= np.ones((nthe,npsi),dtype='float64')
		rdrarg	= zthe/jac
		zdrarg	= xthe/det
		for i in range(nthe-1):
			rdr[i,:] = deriv(rho,rdrarg[i,:])
		for i in range(nthe-1):
			zdr[i,:] = deriv(rho,zdrarg[i,:])

		#--- Compute theta derivatives for delstar-psi
		rdt	= np.ones((nthe,npsi),dtype='float64')
		zdt	= np.ones((nthe,npsi),dtype='float64')
		rdtarg	= ztheh/jac
		zdtarg	= xtheh/det
		for j in range(npsi-1):
			rdt[:,j] = deriv(the,rdtarg[:,j])
		for j in range(npsi-1):
			zdt[:,j] = deriv(the,zdtarg[:,j])

		#--- Correct theta derivative at end-points
		thep	= [the[0]-(the[nthe-1]-the[nthe-2]),the[0],the[1]]
		dxt	= np.ones((3,npsi),dtype='float64')
		dzt	= np.ones((3,npsi),dtype='float64')
		for j in range(npsi-1):
			dxt[:,j] = deriv(thep,rdtarg[[nthe-2,0,1],j])
		for j in range(npsi-1):
			dzt[:,j] = deriv(thep,zdtarg[[nthe-2,0,1],j])
		rdt[0,:]	= dxt[1,:]
		zdt[0,:]	= dzt[1,:]
		rdt[nthe-1,:]	= dxt[1,:]
		zdt[nthe-1,:]	= dzt[1,:]

		#--- Compute the R & Z components of delstar-psi, then sum
		dsr	= (ztheh*rdr-zrho*rdt)/(tdpsi*det)*dx
		dsz	= (xtheh*zdr-xrho*zdt)/(tdpsi*det)
		delstar	= dsr+dsz

	return {'jacobian':jac,'xthe':xthe,'xpsi':xpsi,'zthe':zthe,'zpsi':zpsi,
				'delstar_psi':delstar,'moments':mom}

#-----------------------------------------------------------------------------------
#  Given the 2D arrays X & Z and 1D arrays theta and psi, re-map the X,Z arrays
#  in the theta variable - possibly with a different Jacobian.
#
#  ntheta = #     --> number of points in the new theta coordinate.
#  coords = 'pt'  --> "pest theta" poloidal coordinates
#  coords = 'ea'  --> "equal-arc"  poloidal coordinates
#  coords = 'ha'  --> "hamada"     poloidal coordinates
#  coords = 'lr'  --> "linear-ray" poloidal coordinates
#
#  METHOD:  	Conservation of volume ==> J1 * dthe1 = J2 * dthe2(the1).
#		Since J2 is specified, dthe2 = J1 / J2 * dthe1.
#		Then, compute anti-derivative of dthe2 and re-normalize.
#
def change_theta_coordinate(x,z,thev,psiv,**kwds):
	#kwds options = ntheta & coords
	nt	= np.size(x[:,0])
	nr	= np.size(x[0,:])
	ntnew	= nt
	if 'ntheta' in kwds:
		ntnew = ntheta

	arg	= np.ones((nt,nr),dtype='float64')
	Js	= flux_coordinate_jacobian(x,z,thev,psiv)

	#--- Is magnetic axis included in fit?
	axis0	= np.mean(np.abs(x[:,0]-x[0,0])) == 0.0

	txt0	= '  Changing theta coordinates to:  '
	txtpt	= txt0+'PEST-THETA...'
	txtea	= txt0+'EQUAL-ARC...'
	txtha	= txt0+'HAMADA...'
	txtlr	= txt0+'LINEAR-RAY'
	if(kwds['coords'].lower() == 'pt'):
		print (txtpt)
	if(kwds['coords'].lower() == 'ea'):
		print (txtea)
	if(kwds['coords'].lower() == 'ha'):
		print (txtha)
	if(kwds['coords'].lower() == 'lr'):
		print (txtlr)

	if 'coords' in kwds:
		if(kwds['coords'].lower() == 'pt'):
			arg = Js['jacobian']/ x**2
		if(kwds['coords'].lower() == 'ea'):
			arg = np.sqrt(Js['xthe']**2+Js['zthe']**2)
		if(kwds['coords'].lower() == 'ha'):
			arg = Js['jacobian']
		if(kwds['coords'].lower() == 'lr'):
			arg = np.ones((nt,nr),dtype='float64')

	#--- Exclude axis point if necessary
	irmin 	= 0
	if (axis0 == 0.0):
		irmin = 1

	#--- compute integral of dthe2 and normalize range to [0,1]
	integ	= np.ones((nt,nr),dtype='float64')
	ninteg	= np.ones((nt,nr),dtype='float64')
	for i in range(irmin,nr-1):
		integ[:,i]  = antideriv1(thev,arg[:,i])
	for i in range(irmin,nr-1):
		ninteg[:,i] = np.abs(integ[:,i])/np.max(np.abs(integ[:,i]))

	#--- generate uniform theta arrays using new number of theta points
	tnewn		= np.arange(ntnew,dtype='float64')/np.float64(ntnew-1)
	newthen		= np.ones((ntnew,nr),dtype='float64')
	newx		= np.ones((ntnew,nr),dtype='float64')
	newz		= np.ones((ntnew,nr),dtype='float64')

	#--- Generate normalized thev array
	nthev	= np.size(thev)
	tnorm	= (thev-thev[0]*np.ones(nthev)) / (thev[nthev-1]-thev[0])

	#--- Interpolate normalized theta2(theta1) onto new uniform theta1 grid
	#FOR i=irmin,nr-1 DO newthen[*,i] = SPLINE(ninteg[*,i],tnorm,tnewn)
	for i in range(irmin,nr-1):
		newthenspline = scinterp.UnivariateSpline(tnorm,ninteg[:,i])
		newthen[:,i] = newthenspline(tnewn)

	#--- Interpolate x,z(theta1) onto normalized theta2(theta1) grid
	for i in range(irmin,nr-1):
		newxinterp = scinterp.interp1d(tnorm,x[:,i])
		newx[:,i] = newxinterp(newthen[:,i])
	for i in range(irmin,nr-1):
		newzinterp = scinterp.interp1d(tnorm,z[:,i])
		newz[:,i]  = newzinterp(newthen[:,i])
	#FOR i=1,nr-1 DO newx[*,i]    = INTERPOL(x[*,i],tnorm,newthen[*,i],/SPLINE)
	#FOR i=1,nr-1 DO newz[*,i]    = INTERPOL(z[*,i],tnorm,newthen[*,i],/SPLINE)
	#stop
	#FOR i=1,nr-1 DO newx[*,i]    = SPLINE(tnorm,x[*,i],newthen[*,i])
	#FOR i=1,nr-1 DO newz[*,i]    = SPLINE(tnorm,z[*,i],newthen[*,i])

	if axis0:
		#--- Make all axis values the same
		newx[:,0]	= x[0,0]*np.ones(ntnew)
		newz[:,0]	= z[0,0]*np.ones(ntnew)

	#--- Enforce periodicity
	newx[0,:] = newx[ntnew-1,:]
	newz[0,:] = newz[ntnew-1,:]

	#--- New theta array from [0,2PI]
	newtheta = tnewn * np.float64(2.0) * np.float64(np.pi)
	qoverF	= np.zeros(nr,dtype='float64')

	#--- Compute q / F where F = R*Bt
	Jsn = flux_coordinate_jacobian(newx,newz,newtheta,psiv)
	for i in range(nr-1):
		qoverF[i] = definteg1(tnewn,Jsn['jacobian'][:,i]/newx[:,i]**2)

	#--- Extrapolate q / F into the axis
	if axis0:
		dpsi0	= psiv[0]-psiv[1]
		dpsi1	= psiv[2]-psiv[1]
		dqof1	= qoverF[2]-qoverF[1]
		qoverF[0]	= dqof1/dpsi1 * dpsi0 + qoverF[1]

	return {'x':newx,'z':newz,'theta':newtheta,'psivec':psiv,'qoverF':qoverF}

#------------------------------------------------------------------------
def iterate_flux_coordinates(xfit,zfit,xaxis2d,zaxis2d,pd,psia,psib,rhonorm,miters,dconv):
	nthe	= np.size(xfit[:,0])
	nrad	= np.size(xfit[0,:])

	relax	= 1.0
	small	= np.float64(1.0e-20)
	iters	= 0

	nugget = 1
	while (nugget == 1):
		iters = iters+1
		# Store old values for later comparison
		xfold	= xfit
		zfold	= zfit

		#--- Compute psi at present x,z coordinates
		pf	= interpolate_psi_only(xfit.reshape(nthe*nrad),zfit.reshape(nthe*nrad),pd)
		# Compute normalized flux on x,z grid
		pfn	= (pf-psia)/(psib-psia)

		#  ing	= WHERE(pfn LT 0.0d0)		; Make sure psihat GE 0
		#  IF (ing[0] NE -1) THEN pfn[ing] = 0.d

		rpfn = pfn.reshape(nthe,nrad)

		# Compute dx,dz referenced to axis
		dxfit = xfit-xaxis2d
		dzfit = zfit-zaxis2d
		ndxfit = dxfit
		ndzfit = dzfit

		# SQRT(psi) is best for interpolation
		rrpfn = np.sqrt(np.float64(np.abs(rpfn)))
		for i in range(nthe-1):
			ndxfitinterp = scinterp.interp1d(dxfit[i,:],rrpfn[i,:])
			ndxfit[i,:] = ndexfitinterp(rhonorm)
		for i in range(nthe-1):
			ndzfitinterp = scinterp.interp1d(dzfit[i,:],rrpfn[i,:])
			ndzfit[i,:] = ndzfitinterp(rhonorm)

		#  FOR i=0,nthe-1 DO ndxfit[i,*] = INTERPOL(dxfit[i,*],rrpfn[i,*],rhonorm,/SPLINE)
		#  FOR i=0,nthe-1 DO ndzfit[i,*] = INTERPOL(dzfit[i,*],rrpfn[i,*],rhonorm,/SPLINE)

		dxfit	= relax*ndxfit + (1.0-relax)*dxfit
		dzfit	= relax*ndzfit + (1.0-relax)*dzfit

		#Add back axis to get x,z coordinates
		xfit	= dxfit+xaxis2d
		zfit	= dzfit+zaxis2d
		# Compute convergence error
		delta  = np.sqrt((xfit-xfold)**2+(zfit-zfold)**2)
		radius = np.sqrt((xfit-zaxis2d)**2+(zfit-zaxis2d)**2)
		ratio  = delta/(radius+small)
		mratio = ratio.max()
		print ('Iteration #, convergence error = {:3i},{:12.6e}'.format(iters,mratio))
		if((mratio > dconv) and (iters < miters)):
			nugget = 1
		else:
			nugget = 0
			return xfit,zfit

#------------------------------------------------------------------------
#  Given the reformed structure "psidata", boundary and axis info,
#  and interpolation parameters, return flux coordinates X & Z (and a
#  bunch of other stuff)
def compute_flux_coords(psidata,r_bnd,z_bnd,r_axis,z_axis,radexp0,radexp1,
						fluxfrac,nradius,ntheta,rscale,dconv,miters,
						**kwds):
	# optional kwds = {coords,empty,rhomin,rhomax,full_bnd}
	
	# psidata	= psidata(psistruc(g))
	# r_bnd	= [...]		; vector of boundary major radius points
	# z_bnd	= [...]		; vector of boundary Z points
	# r_axis	= 1.0	; major radius of magnetic axis
	# z_axis	= 0.0	; vertical position of magnetic axis
	# radexp0	= 2.0	; psi scales as rminor^radexp0 near axis
	# radexp1	= 0.25	; psi scales as rminor^radexp1 near edge
	# fluxfrac	= 0.997	; fraction of total flux spanned by surfaces
	# nradius	= 101	; number of flux surfaces
	# ntheta	= 101	; number of theta points on surface
	# rscale	= 1.02	; extrapolate past boundary this far when interpolating
	# dconv	= 1.0E-4	; flux surface convergence parameter
	# miters	= 10	; maximum number of iterations
	#coords	= 'ea'      ; coordinate system choices

	tstart	= systime.time()

	#--- Flux fraction
	dfluxfrac	= fluxfrac
	if 'rhomax' in kwds:
		dfluxfrac = kwds['rhomax']**2

	#--- Coordinate system choice
	defcor	= 'lr'
	if 'coords' in kwds:
		defcor = kwds['coords']

	#--- Initialize data
	nrad	= np.int32(nradius)
	nthe	= np.int32(ntheta)

	if 'empty' in kwds:
		xfit	= np.zeros((nthe,nrad),dtype='float64')
		zfit	= np.zeros((nthe,nrad),dtype='float64')
		psifit	= np.zeros(nrad,dtype='float64')
		psinorm	= np.zeros(nrad,dtype='float64')
		xb_ea	= np.zeros(nthe,dtype='float64')
		zb_ea	= np.zeros(nthe,dtype='float64')
		thefit	= np.zeros(nthe,dtype='float64')
		angle	= np.zeros(nthe,dtype='float64')
		qoverF	= np.zeros(nrad,dtype='float64')
	else:
		#--- Array for exponent
		unitv0	= np.arange(nrad,dtype='float64')/(nrad-1)
		vector	= np.sin(unitv0*np.pi/2.0)**2
		radexp	= vector*(radexp1-radexp0)+radexp0

		#--- rho minimum
		umin = np.float64(0.0)
		if 'rhomin' in kwds:
			umin = (kwds['rhomin']**2/dfluxfrac)^(1/radexp[0])
		unitv	= unitv0*(1.0-umin)+umin
		#vector	= unitv0

		pd	= psidata
		rb	= np.float64(r_bnd)
		zb	= np.float64(z_bnd)
		xaxis	= np.float64(r_axis)
		zaxis	= np.float64(z_axis)
		
		#--- Use bi-cubic splines to best-fit the axis position
		###########xaxis,zaxis = refine_null_position([xaxis],[zaxis],pd)

		#--- Compute 1D and 2D arrays for axis position
		xaxis1d	= xaxis*np.ones(nthe)
		zaxis1d	= zaxis*np.ones(nthe)
		xaxis2d	= xaxis*np.ones((nthe,nrad))
		zaxis2d	= zaxis*np.ones((nthe,nrad))

		#--- Compute psi at the refined magnetic axis
		psias	= interpolate_psi(xaxis,zaxis,pd)
		psia	= psias['psi'][0]

		#--- Fit the EFIT boundary from [0,2*PI] with nthe theta points
		iuni	= np.int32('full_bnd' in kwds)
		unib	= 1-iuni
		fb	= fit_boundary(rb,zb,nthe,uniform=unib)
		xb_fit	= fb['xfit']
		zb_fit	= fb['zfit']

		#--- Re-map the boundary to be equal-arcs
		xb_ea	= xb_fit
		zb_ea	= zb_fit
		#fbea	= remap_eal(xb_fit,zb_fit,nthe)
		#xb_ea	= fbea.xea
		#zb_ea	= fbea.zea
		#oplot,xb_fit,zb_fit,psym=4,color=mk_color('red')

		#--- Compute mean psi at the new equal-arc boundary
		ip 	= interpolate_psi(xb_ea,zb_ea,pd)
		psib	= np.mean(ip['psi'])

		#--- Flux function which flux coordinates will lie on
		psinrm	= unitv**radexp
		psifit	= dfluxfrac*(psib-psia)*psinrm+psia
		psinorm = (psifit-psia)/(psib-psia)
		rhonorm	= np.sqrt(np.float64(psinorm))

		#--- Compute the radius and angle of the boundary points from the axis
		ra	= rho_angle((xb_ea-xaxis1d),(zb_ea-zaxis1d))
		#--- Generate psin fitting grid (xfit,zfit)
		rnorm0	= rscale * unitv**(radexp/2.0)
		rnorm	= np.zeros((nthe,nrad),dtype='float32')
		rho		= np.zeros((nthe,nrad),dtype='float32')
		angle	= np.zeros((nthe,nrad),dtype='float32')
		for i in range(nthe-1):
			rnorm[i,:]	= rnorm0
		for i in range(nrad-1):
			rho[:,i]	= ra['rho']
		for i in range(nrad-1):
			angle[:,i]	= ra['angle']
		xfit = xaxis2d+rnorm*rho*np.cos(angle)
		zfit = zaxis2d+rnorm*rho*np.sin(angle)

		#--- Adjust coordinates until largest spatial shift is smaller
		#    than tolerance parameter dconv
		#
		#  NEED TO FIX THIS - DOESN'T WORK FOR HAMADA, PEST-THETA...
		#
		for ifc in range(0):
			xfit,zfit = iterate_flux_coordinates(xfit,zfit,xaxis2d,zaxis2d,pd,psia,psib,
				rhonorm,miters,dconv)

		#--- Construct uniform theta vector to pass back with f.c. solution
		thefit	= np.arange(nthe,dtype='float64')/np.float64(nthe-1)*np.float64(2.0*np.pi)

		#--- Make X,Z have specified Jacobian
		# available kwds = coords & ntheta
		kwds={'coords':defcor}
		ctc		= change_theta_coordinate(xfit,zfit,thefit,psifit,**kwds)
		xfit	= ctc['x']
		zfit	= ctc['z']
		xb_ea	= ctc['x'][:,nrad-1]
		zb_ea	= ctc['z'][:,nrad-1]
		qoverF	= ctc['qoverF']

	struc	= {'xfit':xfit,'zfit':zfit,'psifit':psifit,'psinorm':psinorm,'xbnd':xb_ea,
				'zbnd':zb_ea,'theta':thefit,'angle':ra['angle'],'nradius':nrad,
				'ntheta':nthe,'qoverF':qoverF}
	tstop	= systime.time()
	dtime	= str(tstop-tstart)
	dtime = dtime[:-2]
	print('Flux coordinates computed in '+dtime+' seconds.')
	return	struc

#------------------------------------------------------------------------
#  Given an EFIT g structure, compute the full equilibrium solution
#  in flux coordinates
def efitg_flux_coords(g,*quiet,**kwds):
	# optional kwds = {rexp0,rexp1,psifrac,nr,nt,ctol,maxit,field_subset,linear,
	#					fast,rexpm,coords,fc,qedge,pedge,rhomin,rhomax,full_bnd,
	#					eindex}
	# kwds must be in dictionary style
	# -- default args setup
	rscale	= 1.01			# extrapolate past boundary this far
	radexp0	= 2.0			# psi scales as rminor^radexp0 near axis
	radexp1	= 0.25			# psi scales as rminor^radexp1 near edge
	flxfrac	= 0.997			# fraction of psi spanned by flux surfaces
	nrad	= 101			# number of flux surfaces
	nthe	= 101			# number of theta points on surface
	dconv	= 1.0e-4		# flux surface convergence parameter
	miters	= 10			# maximum number of iterations
	defcor	= 'lr'			# default coordinate system
	if 'fc' in kwds:
		fc	= kwds['fc']	# pass flux coordinates through keyword

	deindex	= np.int32(0)
	if 'eindex' in kwds:
		deindex = np.int32(kwds['eindex'])

	#--- Execution time
	twopi	= 2.0*np.pi
	tstart	= systime.time()

	#--- Keywords
	if not quiet:
		iverbose  = 1

	if 'rexp0' in kwds:
		radexp0 = kwds['rexp0']
	if 'rexp1' in kwds:
		radexp1 = kwds['rexp1']
	if 'psifrac' in kwds:
		flxfrac = kwds['psifrac']

	if 'nr' in kwds:
		nrad = kwds['nr']
	if 'nt' in kwds:
		nthe = kwds['nt']
	if 'ctol' in kwds:
		dconv = kwds['ctol']
	if 'maxit' in kwds:
		miters = kwds['maxit']
	if 'coords' in kwds:
		defcor = kwds['coords']

	if 'linear' in kwds:
		radexp0 = 1.0
		radexp1 = 1.0
		print ('--> Radial coordinate is linear in poloidal flux.')

	stext	= str(g['shot']).strip()
	stime	= stext+', t='+str(g['time'])+' msec...'
	if iverbose:
		print ('Computing flux coordinate solution for shot='+stime)

	#--- R and Z of magnetic axis from EFIT
	xaxis	= np.float64(g['rmaxis'])
	zaxis	= np.float64(g['zmaxis'])

	#--- R and Z of boundary from EFIT
	rbe	= np.array(g['bdry'][0,0:g['nbdry']-1],dtype='float64').squeeze()
	zbe	= np.array(g['bdry'][1,0:g['nbdry']-1],dtype='float64').squeeze()

	#--- Interpolate F, FF', p' from EFIT onto new flux array
	#Get subarrays
	iactive	= np.where(g['fpol'] != 0)
	n_	= np.size(iactive)
	F_	= g['fpol'][iactive]
	FFp_= g['ffprim'][iactive]
	q_	= g['qpsi'][iactive]
	pp_	= g['pprime'][iactive]
	p_	= g['pres'][iactive]

	#--- Set the edge pressure value if this keyword is set
	n	= np.size(p_)
	if 'pedge' in kwds:
		p_ = (p_ - p_[n-1]) + kwds['pedge']*np.ones(n)

	# Compute EFIT psi
	norm	= np.arange(n_,dtype='float64')/(n_-1)
	psi0_	= g['ssimag']
	psi1_	= g['ssibry']
	psiefit	= norm*(psi1_-psi0_)+psi0_

	# Normalized psi(s) for interpolation (monotonically increasing)
	x0	= psiefit[0]
	x1	= psiefit[np.size(psiefit)-1]
	psien	= np.abs((psiefit-x0)/(x1-x0))

	# Change flux fraction based on specified q-edge?
	# If so, use largest value allowable between [0,1]
	if 'qedge' in kwds:
		flxfrcqinterp = scinterp.UnivariateSpline(psien,np.abs(q_))
		flxfrcq = flxfrcqinterp(kwds['qedge'])
		flxfrcq = min(flxfrcq,0.99990)
		flxfrcq = max(flxfrcq,0.00100)
		flxfrac = min(flxfrac,flxfrcq)
		print ('Flux fraction changed to '+str(flxfrac)+' to attempt match of specified edge q = {}'.format(kwds['qedge']))

	#--- Compute the flux coordinates X,Z and related quantities
	ps	= psistruc(g)
	pd 	= psidata(ps)

	#--- Use only boundary points that match boundary flux value to within tolerance
	psnbtol	= 0.0035
	psnbchk	= np.abs(((interpolate_psi(rbe,zbe,pd))['psi']-psi0_)/(psi1_-psi0_)-1.0)
	#jpsnbch	= 1-((psnbchk EQ MAX(psnbchk)) AND ((zbe EQ MAX(zbe)) OR (zbe EQ MIN(zbe))))
	jpsnbch	= 1
	ipsnbch	= np.where((psnbchk.any() < psnbtol) and jpsnbch)

	if(ipsnbch[0] != -1):
		print ('Maximum psin error on boundary = {}'.format(psnbchk.max()))
		rbe	= rbe[ipsnbch]
		zbe	= zbe[ipsnbch]

	if 'fc' not in kwds:
		kwds2={}
		if 'coords' in kwds:
			kwds2['coords']=defcor
		if 'rhomin' in kwds:
			kwds2['rhomin']=kwds['rhomin']
		if 'rhomax' in kwds:
			kwds2['rhomax']=kwds['rhomax']
		if 'full_bnd' in kwds:
			kwds2['full_bnd']=kwds['full_bnd']
		fc = compute_flux_coords(pd,rbe,zbe,xaxis,zaxis,radexp0,radexp1,
									flxfrac,nrad,nthe,rscale,dconv,miters,
									**kwds2)
	xb	= fc['xbnd']
	zb	= fc['zbnd']
	x	= fc['xfit']
	z	= fc['zfit']
	psiv	= fc['psifit']
	psinorm = fc['psinorm']
	nr	= fc['nradius']
	nt	= fc['ntheta']
	qoverF	= fc['qoverF']

	#Interpolate onto new psi
	psivn	= np.abs((psiv-x0)/(x1-x0))
	F_vinterp = scinterp.UnivariateSpline(psien,F_)
	F_v	= F_vinterp(psivn)
	FFp_vinterp = scinterp.UnivariateSpline(psien,FFp_)
	FFp_v	= FFp_vinterp(psivn)
	q_vinterp = scinterp.UnivariateSpline(psien,q_)
	q_v	= q_vinterp(psivn)
	pp_vinterp = scinterp.UnivariateSpline(psien,pp_)
	pp_v	= pp_vinterp(psivn)
	p_vinterp = scinterp.UnivariateSpline(psien,p_)
	p_v	= p_vinterp(psivn)
	#p_v	= antideriv1(psiv,pp_v)
	#p_v	= ABS(p_v-p_v[N_ELEMENTS(p_v)-1])

	# Compute new 2D arrays of F, FF', p', q
	F	= np.zeros((nt,nr),dtype='float64')
	FFp	= F.copy()
	pp	= F.copy()
	p	= F.copy()
	q	= F.copy()
	for i in range(nt-1):
		F[i,:]   = F_v
	for i in range(nt-1):
		FFp[i,:] = FFp_v
	for i in range(nt-1):
		pp[i,:]  = pp_v
	for i in range(nt-1):
		q[i,:]   = q_v
	for i in range(nt-1):
		p[i,:]   = p_v
	Fp	= FFp/F

	#--- Re-compute q-profile from EFIT
	qctc	= np.zeros((nt,nr),dtype='float64')
	for i in range(nt-1):
		qctc[i,:] = qoverF * F_v
	q_orig	= q
	q	= qctc

	#--- Compute Jacobian and related derivatives
	thenorm = np.arange(nt,dtype='float64')/np.float(nt-1)
	the = thenorm*2.0*np.pi
	jacs = flux_coordinate_jacobian(x,z,the,psiv)
	jac	= jacs['jacobian']
	xpsi	= jacs['xpsi']
	zpsi	= jacs['zpsi']
	xthe	= jacs['xthe']
	zthe	= jacs['zthe']

	#-- Compute poloidal arc-length
	parclen = np.zeros(nr,dtype='float64')
	for j in range(nt-2):
		dx	= x[j+1,:]-x[j,:]
		dz	= z[j+1,:]-z[j,:]
		parclen = parclen + np.sqrt(dx**2+dz**2)

	#--- Compute time array
	time = g['time']/1000.0

	#--- Compute Grad-psi at (x,z)
	nels = np.int32(nr*nt)
	x1d = x.reshape(nels)
	z1d = z.reshape(nels)
	pss = interpolate_psi(x1d,z1d,pd)

	#--- Compute flux and derivatives
	psi	= pss['psi'].reshape(nt,nr)
	dpsidr	= pss['dpsidr'].reshape(nt,nr)
	dpsidz	= pss['dpsidz'].reshape(nt,nr)

	#--- 2D arrays of normalized flux and "helically distorted" normalized flux
	psin	= np.zeros((nt,nr),dtype='float32')
	for i in range(nt-1):
		psin[i,:] = psinorm
	psinh	= psin

	#--- Force psi to be constant on a surface of constant flux
	for i in range(nr-1):
		psi[:,i] = np.mean(psi[0:nt-2,i])*np.ones(nt)

	#--- Compute B-field arrays
	mu0 = np.float64(4.0*np.pi*1.0e-7)
	Bphi= F/x
	Br	= -dpsidz/x
	Bz	= +dpsidr/x
	Bpol	= np.sqrt(Br**2+Bz**2)
	B	= np.sqrt(Bphi**2+Bpol**2)

	#--- Compute current arrays
	Jphi	= x*pp + FFp/(mu0*x)
	Jpol	= Fp*Bpol/mu0

	#--- Define volume
	dVdpsi	= np.zeros(nr,dtype='float64')
	for i in range(nr-1):
		dVdpsi[i] = 2.0*np.pi*definteg1(the,jac[:,i])
	volume	= antideriv1(psi[0,:].squeeze(),dVdpsi)

	#--- Compute the toroidal flux using internal small version of efc structure
	intefc	= {'nt':nt,'nr':nr,'Jacobian':jac,'theta':the,'psivec':psi[0,:].squeeze(),'x':x}
	torflux	= area_integrate(Bphi,intefc)

	#--- Compute <1/R^2>
	saRm2	= fs_average_array(1.0/x**2,jac,the)

	#--- Compute what should be unity at (x,z)
	# (This is a good test of the derivative calculations)
	derchk	= dpsidr*xpsi+dpsidz*zpsi

	#--- Compute pest-theta coordinates in the new coordinate system
	tnorm	= (the-the[0]*np.ones(nt)) / (the[nt-1]-the[0])
	ptarg	= jac / x**2
	integ	= np.zeros((nt,nr),dtype='float64')
	ninteg	= np.zeros((nt,nr),dtype='float64')
	for i in range(nr-1):
		integ[:,i] = antideriv1(the,ptarg[:,i])
	for i in range(nr-1):
		ninteg[:,i] = np.abs(integ[:,i])/np.max(np.abs(integ[:,i]))
	print (integ.max())
	ptheta 	= np.float64(2.0*np.pi)*ninteg

	#--- Compute flux surface averages of J.B (F' and p' components)
	saJBfp	= fs_average_array(FFp/F*B**2,jac,the)/mu0
	saJBpp	= fs_average_array(F*pp,jac,the)
	saJBtot	= saJBfp+saJBpp

	#--- Compute <J.B>/<B.Grad-phi>(psi) for J-SOLVER
	Jjsolv 	= saJBtot/(F[0,:]*saRm2)

	#--- Compute various minor radial coordinates
	#    rho = SQRT(normalized flux or volume)
	rhopol	= np.sqrt(np.abs(psinorm)/np.max(np.abs(psinorm)))
	rhotor	= np.sqrt(np.abs(torflux)/np.max(np.abs(torflux)))
	rhovol	= np.sqrt(np.abs(volume)/np.max(np.abs(volume)))

	#--- Setup variables for 3D representation
	null	= np.zeros((nt,nr),dtype='float64')
	null_nr	=np.zeros(nr,dtype='float64')
	null_nt	= np.zeros(nt,dtype='float64')
	y	= null
	yb	= null_nt
	ypsi	= null
	ythe	= null
	R	= x
	Rb	= xb
	Rpsi	= xpsi
	Rthe	= xthe
	phirad	= np.float64(0.0)
	phideg	= np.float64(0.0)

	#--- Min and max R on each flux surface
	Rmin	= null_nr
	Rmax	= null_nr
	for j in range(nr-1):
		Rmin[j] = R[:,j].min()
	for j in range(nr-1):
		Rmax[j] = R[:,j].max()

	#--- Min and max B on each flux surface
	Bmin	= null_nr
	Bmax	= null_nr
	for j in range(nr-1):
		Bmin[j] = B[:,j].min()
	for j in range(nr-1):
		Bmax[j] = B[:,j].max()

	#--- Inverse aspect ratio profiles
	# Global
	epsg = (Rb.max()-Rb.min())/(Rb.max()+Rb.min()) * rhopol

	# Local
	epsl = (Rmax-Rmin)/(Rmax+Rmin)

	# Mod-B equivalent
	epsb = (Bmax-Bmin)/(Bmax+Bmin)

	#--- M2 = toroidal Mach number squared, M2p = d(M(psi))^2/dpsi
	M2	= np.zeros((nt,nr),dtype='float64')
	M2p	= np.zeros((nt,nr),dtype='float64')

	#--- Compute <B^2>
	saBphi2	= fs_average_array(Bphi**2,jac,the)
	saBpol2	= fs_average_array(Bpol**2,jac,the)
	saB2	= fs_average_array(   B**2,jac,the)

	#--- Skip time-dependent stuff?
	if 'fast' in kwds:
		tstop	= systime.time()
		dtime	= str(tstop-tstart)[:-2]
		if iverbose:
			print ('--> Full solution computed in '+dtime+' seconds.')

		return {'time':time,'shot':g['shot'],'eindex':deindex,'x':x,'y':y,
					'z':z,'r':R,'xb':xb,'yb':yb,'zb':zb,'rb':Rb,'psi':psi,
					'psin':psin,'psinh':psinh,'nr':nr,'nt':nt,'Jacobian':jac,
					'phirad':phirad,'phideg':phideg,'xpsi':xpsi,'ypsi':ypsi,
					'zpsi':zpsi,'Rpsi':Rpsi,'xthe':xthe,'ythe':ythe,'zthe':zthe,
					'Rthe':Rthe,'Rmin':Rmin,'Rmax':Rmax,'Bmin':Bmin,'Bmax':Bmax,
					'epsg':epsg,'epsl':epsl,'epsb':epsb,'parclen':parclen,'saB2':saB2,
					'coords':defcor,'derchk':derchk,'pp':pp,'FFp':FFp,'Fp':Fp,'F':F,
					'q':q,'q_orig':q_orig,'B':B,'p':p,'theta':the,
					'psivec':psi[0,:].squeeze(),'psinorm':psinorm,'saJBfp':saJBfp,
					'saJBpp':saJBpp,'saJBtot':saJBtot,'pesttheta':ptheta,'Bphi':Bphi,
					'Br':Br,'Bz':Bz,'Bpol':Bpol,'Jjsolv':Jjsolv,'volume':volume,
					'dVdpsi':dVdpsi,'Jphi':Jphi,'Jpol':Jpol,'saRm2':saRm2,
					'rhopol':rhopol,'rhotor':rhotor,'rhovol':rhovol,'M2':M2,'M2p':M2p}

	#--- Compute components of electric field
	Er	= np.zeros((nt,nr),dtype='float64')
	Ez	= np.zeros((nt,nr),dtype='float64')
	Ephi	= np.zeros((nt,nr),dtype='float64')
	Epol	= np.zeros((nt,nr),dtype='float64')
	dBzdt	= np.zeros((nt,nr),dtype='float64')
	dBrdt	= np.zeros((nt,nr),dtype='float64')
	dBphidt	= np.zeros((nt,nr),dtype='float64')

	if 'field_subset' in kwds:
		fs 	= field_subset
		x1v	= fs.r[:,0].squeeze()
		x2v	= fs.z[0,:].squeeze()

		Us		= {'U':fs['dpsidt'],'x1v':x1v,'x2v':x2v}
		pint	= auto_interpolate_U(x1d,z1d,Us)
		Ephi	= -1*(pint['Ui']/x1d).reshape(nt,nr)
		dBzdt	=  (pint['dUdx1i']/x1d).reshape(nt,nr)
		dBrdt	= -1*(pint['dUdx2i']/x1d).reshape(nt,nr)

		Us		= {'U':fs['dBphidt'],'x1v':x1v,'x2v':x2v}
		dBint	= auto_interpolate_U(x1d,z1d,Us)
		# integrand
		intgr	= (dBint['Ui']/x1d).reshape(nt,nr)*jac
		for i in range(nt-1):
			Epol[i,:] = antideriv1(psiv,intgr[i,:])
		Epol	= Epol/(jac*Bpol)
		dBphidt	= dBint['Ui'].reshape(nt,nr)

		Us		= {'U':fs['Er'],'x1v':x1v,'x2v':x2v}
		Erint	= auto_interpolate_U(x1d,z1d,Us)
		Er		= Erint['Ui'].reshape(nt,nr)

		Us		= {'U':fs['Ez'],'x1v':x1v,'x2v':x2v}
		Ezint	= auto_interpolate_U(x1d,z1d,Us)
		Ez		= Ezint['Ui'].reshape(nt,nr)

	#--- Compute flux surface averages of E.B (phi and poloidal comps.)
	saEBphi	= fs_average_array(Ephi*Bphi,jac,the)
	saEBpol	= fs_average_array(Epol*Bpol,jac,the)

	#--- Sanity check on E.B from E.Bpoloidal
	#saEBpol	= fs_average_array(Er*Br+Ez*Bz,jac,the)

	saEBtot	= saEBphi + saEBpol

	vloopt	= (twopi * saEBphi / (F[0,:] * saRm2 )).squeeze()
	vloopp	= (twopi * saEBpol / (F[0,:] * saRm2 )).squeeze()
	vloop	= vloopt + vloopp

	#--- Compute conductivity
	sigma	= np.zeros(nr,dtype='float64')
	if(field_subset):
		sigma	= saJBtot/saEBtot

	#--- All done
	tstop	= systime.time()
	dtime	= str(tstop-tstart)[:-2]
	if iverbose:
		print ('--> Full solution computed in '+dtime+' seconds.')

	return {'time':time,'shot':g['shot'],'eindex':deindex,'x':x,
				'y':y,'z':z,'r':r,'xb':xb,'yb':yb,'zb':zb,'rb':rb,
				'psi':psi,'psin':psin,'psinh':psinh,'nr':nr,'nt':nt,
				'Jacobian':jac,'phirad':phirad,'phideg':phideg,
				'xpsi':xpsi,'ypsi':ypsi,'zpsi':zpsi,'Rpsi':Rpsi,
				'xthe':xthe,'ythe':ythe,'zthe':zthe,'Rthe':Rthe,
				'Rmin':Rmin,'Rmax':Rmax,'Bmin':Bmin,'Bmax':Bmax,
				'epsg':epsg,'epsl':epsl,'epsb':epsb,'parclen':parclen,
				'coords':defcor,'derchk':derchk,'pp':pp,'FFp':FFp,
				'Fp':Fp,'F':F,'q':q,'q_orig':q_orig,'B':B,'p':p,
				'theta':the,'psivec':psi[0,:].squeeze(),'psinorm':psinorm,
				'pesttheta':ptheta,'Bphi':Bphi,'Br':Br,'Bz':Bz,'Bpol':Bpol,
				'volume':volume,'rhopol':rhopol,'rhotor':rhotor,
				'rhovol':rhovol,'M2':M2,'M2p':M2p,'dVdpsi':dVdpsi,'Jphi':Jphi,
				'Jpol':Jpol,'dBphidt':dBphidt,'dBzdt':dBzdt,'dBrdt':dBrdt,
				'saEBphi':saEBphi,'saEBpol':saEBpol,'saEBtot':saEBtot,
				'vloop':vloop,'vloopt':vloopt,'vloopp':vloopp,
				'saJBfp':saJBfp,'saJBpp':saJBpp,'saJBtot':saJBtot,
				'sigma':sigma,'saRm2':saRm2,'Er':Er,'Ez':Ez,'Ephi':Ephi,
				'Epol':Epol,'saBphi2':saBphi2,'saBpol2':saBpol2,
				'saB2':saB2,'Jjsolv':Jjsolv}
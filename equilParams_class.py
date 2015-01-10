import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import EFIT.geqdsk as gdsk
import EFIT.equilibrium as eq
from Misc.deriv import deriv
#reload(eq)

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
            self.dpsidZ, self.dpsidR = np.gradient(self.PSIdict['psi2D'], self.RZdict['dZ'], self.RZdict['dR'])
            self.B_R = self.dpsidZ / self.RZdict['Rs2D']
            self.B_Z = -self.dpsidR / self.RZdict['Rs2D']
            self.Bp_2D = np.sqrt(self.B_R**2 + self.B_Z**2)
            self.theta = np.linspace(0.0, 2.*np.pi, self.thetapnts)

            psiN2D = self.PSIdict['psiN_2D'].flatten()
            idx = np.where(psiN2D <= 1)[0]
            Fpol2D = np.ones(psiN2D.shape) * self.data.get('bcentr') * self.data.get('rcentr')
            Fpol2D[idx] = self.PROFdict['ffunc'](psiN2D[idx])
            Fpol2D = Fpol2D.reshape(self.PSIdict['psiN_2D'].shape)
            self.Bt_2D = Fpol2D / self.RZdict['Rs2D']

            #---- more Functions ----
            self.psiFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'], self.PSIdict['psiN_2D'].T)
            self.BpFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'], self.Bp_2D.T)
            self.BtFunc = interp.RectBivariateSpline(self.RZdict['Rs1D'], self.RZdict['Zs1D'], self.Bt_2D.T)


        # --------------------------------------------------------------------------------
        # Interpolation function handles for all 1-D fields in the g-file
        def profiles(self):
            #---- Profiles ----
            fpol = self.data.get('fpol')
            ffunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(fpol)),fpol,s=0)
            fprime = self.data.get('ffprime')/fpol
            fpfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(fprime)),fprime,s=0)
            ffprime = self.data.get('ffprime')
            ffpfunc = interp.UnivariateSpline(np.linspace(0.,1.,np.size(ffprime)),ffprime,s=0)
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
            dR = (self.Rmax - self.Rmin)/(self.nw - 1)
            Rs1D = np.linspace(self.Rmin, self.Rmax, self.nw)
            dZ = (self.Zmax - self.Zmin)/(self.nh - 1)
            Zs1D = np.linspace(self.Zmin, self.Zmax, self.nh)
            Rs2D,Zs2D = np.meshgrid(Rs1D,Zs1D)

            return {'Rs1D':Rs1D,'dR':dR,'Zs1D':Zs1D,'dZ':dZ,'Rs2D':Rs2D,'Zs2D':Zs2D}


        # --------------------------------------------------------------------------------
        # 1-D and 2-D poloidal flux, normalized and regular
        # compared to the integral definition of psipol (psipol = 2pi integral_Raxis^Rsurf(Bpol * R * dR))
        # regular is shifted by self.siAxis and missing the factor 2*pi !!!
        # so: psipol = psi1D = 2pi * (self.siBry-self.siAxis) * psiN1D
        def getPsi(self):
            psiN1D = np.linspace(0.0 ,1.0, self.nw)
            psi1D = 2*np.pi * (self.siBry - self.siAxis) * psiN1D
            psi2D = self.data.get('psirz')
            psiN_2D = (psi2D - self.siAxis) / (self.siBry - self.siAxis)
            # psiN_2D[np.where(psiN_2D > 1.2)] = 1.2

            return {'psi2D':psi2D,'psiN_2D':psiN_2D,'psi1D':psi1D,'psiN1D':psiN1D}


        # --------------------------------------------------------------------------------
        # 1-D normalized toroidal flux
        # dpsitor/dpsipol = q  ->  psitor = integral(q(psipol) * dpsipol)
        # dummy input variable for backward compatability <-> this input is unused!!!
        def getTorPsi(self, dummy = None):
            dpsi = (self.siBry - self.siAxis)/(self.nw - 1) * 2*np.pi
            hold = integ.cumtrapz(self.PROFdict['q_prof'], dx = dpsi) * np.sign(self.data.get('bcentr'))
            psitor = np.append(0, hold)
            psitorN1D = (psitor - psitor[0])/(psitor[-1] - psitor[0])

            return {'psitorN1D':psitorN1D, 'psitor1D':psitor}


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

            Bp_hold = self.BpFunc.ev(R_hold, Z_hold)
            fpol_psiN = self.PROFdict['ffunc'](psiNVal)*np.ones(np.size(Bp_hold))
            fluxSur = eq.FluxSurface(fpol_psiN[0:-1], R_hold[0:-1], Z_hold[0:-1], Bp_hold[0:-1], self.theta[0:-1])
            Bt_hold = np.append(fluxSur._Bt, fluxSur._Bt[0])    # add last point = first point
            Bmod = np.append(fluxSur._B, fluxSur._B[0])         # add last point = first point

            return {'Rs':R_hold, 'Zs':Z_hold, 'Bp':Bp_hold, 'Bt':Bt_hold, 'Bmod':Bmod,
                    'fpol_psiN':fpol_psiN, 'FS':fluxSur}


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

            for i, psiNVal in enumerate(self.PSIdict['psiN1D']):
                R_hold = R[i,:]
                Z_hold = Z[i,:]

                Bp_hold = self.BpFunc.ev(R_hold, Z_hold)
                fpol_psiN = self.PROFdict['fpol'][i] * np.ones(self.thetapnts)

                # eq.FluxSurface requires theta = [0:2pi] without the last point
                fluxSur = eq.FluxSurface(fpol_psiN[0:-1], R_hold[0:-1], Z_hold[0:-1], Bp_hold[0:-1], self.theta[0:-1])
                Bt_hold = np.append(fluxSur._Bt, fluxSur._Bt[0])    # add last point = first point
                Bmod = np.append(fluxSur._B, fluxSur._B[0])         # add last point = first point

                FluxSur = {'Rs':R_hold, 'Zs':Z_hold, 'Bp':Bp_hold, 'Bt':Bt_hold,
                           'Bmod':Bmod, 'fpol_psiN':fpol_psiN, 'FS':fluxSur}

                FluxSurList.append(FluxSur)

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

            for i in xrange(self.nw):
                R_hold = FluxSurfList[i]['Rs']
                Z_hold = FluxSurfList[i]['Zs']

                Rs_hold2D[i,:] = R_hold
                Zs_hold2D[i,:] = Z_hold
                Btot_hold2D[i,:] = FluxSurfList[i]['Bmod']
                Bp_hold2D[i,:] = FluxSurfList[i]['Bp']
                Bt_hold2D[i,:] = FluxSurfList[i]['Bt']
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

            if(FluxSurfList == None):
                FluxSurfList = self.get_allFluxSur()
            if(Bdict == None):
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
                curvGeo_2D[i, :] = -1*fpol_psiN*(Bdict['Rs_2D'][i, :]*(kap2t1 - kap2t2 + kap2t3))/(R**5 * Bp * B**3)

                coeft1 = fpol_psiN/(R**4 * Bp**2 * B**2)
                coeft2 = ((Bdict['d2psidR2_2D'][i, :] -
                          Bdict['d2psidZ2_2D'][i, :])*(Bdict['dpsidR_2D'][i, :]**2 -
                          Bdict['dpsidZ_2D'][i, :]**2) +
                            (4*Bdict['dpsidR_2D'][i,:]*Bdict['d2psidRdZ_2D'][i,:]*Bdict['dpsidZ_2D'][i,:]))
                sht2 = fpol_psiN*Bdict['dpsidR_2D'][i,:]/(R**3 * B**2)
                sht3 = fprint_psiN*Bp**2/(B**2)
                shear_fl[i, :] = -1*(coeft1*coeft2 + sht2 - sht3)

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
            jpar1D = (PROFdict['fprime']*Bsqrd_prof/mu0 + PROFdict['pprime']*PROFdict['fpol']) / self.bcentr/1.e6 * np.sign(self.data.get('current'))

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
            jtor1D = np.abs(jtor1D)/1.e6 * np.sign(self.data.get('current'))
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
            #         'Bp_2D', 'Bt_2D', 'Btot_2D', 'Rs_2D', 'Zs_2D'
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
            Jdict = self.cur_density(FluxSurList)   # returns 'jpar', 'jtor'
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
        def flux_surface(self, psi0, N, theta = None):
            if(theta == None):
                theta = np.linspace(0,2*np.pi,N + 1)[0:-1]
            else :
                N = len(theta)

            r = np.zeros(theta.shape)

            # get maximum r from separatrix in g-file
            Rsep = self.data.get('rbbbs')
            Zsep = self.data.get('zbbbs')
            idxup = Zsep.argmax()
            idxdwn = Zsep.argmin()
            rup = np.sqrt((Rsep[idxup] - self.rmaxis)**2 + (Zsep[idxup] - self.zmaxis)**2)
            rdwn = np.sqrt((Rsep[idxdwn] - self.rmaxis)**2 + (Zsep[idxdwn] - self.zmaxis)**2)
            rmax = max([rup, rdwn])

            # iterate
            for i in xrange(N):
                r[i] = self.__bisec__(psi0, theta[i], b = rmax)

            R = r*np.cos(theta) + self.rmaxis
            Z = r*np.sin(theta) + self.zmaxis

            return R, Z

        # ----------------------------------------------------------------------------------------
        def get_j2D(self):
            mu0 = 4*np.pi*1e-7
            jR = np.zeros(self.RZdict['Rs2D'].shape)
            jZ = np.zeros(self.RZdict['Rs2D'].shape)
            jtor = np.zeros(self.RZdict['Rs2D'].shape)

            for i in xrange(self.nw):
                jR[:,i] = -deriv(self.Bt_2D[:,i], self.RZdict['Zs2D'][:,i])
                jtor[:,i] = deriv(self.B_R[:,i], self.RZdict['Zs2D'][:,i])

            for i in xrange(self.nh):
                jZ[i,:] = deriv(self.Bt_2D[i,:], self.RZdict['Rs2D'][i,:])
                jtor[i,:] -= deriv(self.B_Z[i,:], self.RZdict['Rs2D'][i,:])

            jZ += self.Bt_2D/self.RZdict['Rs2D']

            jR /= mu0
            jZ /= mu0
            jtor /= mu0

            idx = np.where((self.PSIdict['psiN_2D'] > 1.0) | (abs(self.RZdict['Zs2D']) > 1.3))
            jR[idx] = 0
            jZ[idx] = 0
            jtor[idx] = 0

            jpar = (jR*self.B_R + jZ*self.B_Z + jtor*self.Bt_2D) * np.sign(self.data.get('current'))
            jpar /= np.sqrt(self.B_R**2 + self.B_Z**2 + self.Bt_2D**2)
            jtot = np.sqrt(jR**2 + jZ**2 + jtor**2)
            jtor *= np.sign(self.data.get('current'))

            return {'R':self.RZdict['Rs2D'], 'Z':self.RZdict['Zs2D'], 'j2D':jtot, 'jpar2D':jpar,
                    'jR2D':jR, 'jZ2D':jZ, 'jtor2D':jtor}


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
            radexp0 = 2.0       # psi scales as r**radexp0 near axis
            radexp1 = 0.25      # psi scales as r**radexp1 near edge
            unitv = np.linspace(0.0, 1.0, npsi)
            vector = np.sin(unitv*np.pi/2.0)**2
            radexp = vector*(radexp1 - radexp0) + radexp0
            rnorm = unitv**(radexp/2.0)

            # get r at last closed flux surface
            R_lcfs = self.data.get('rbbbs')
            Z_lcfs = self.data.get('zbbbs')
            r_lcfs = self.__get_r__(R_lcfs, Z_lcfs)[1::]        # here first == last, so skip first
            th_lcfs = self.__get_theta__(R_lcfs, Z_lcfs)[1::]   # here first == last, so skip first
            index = np.argsort(th_lcfs) # th-lcfs needs to be monotonically increasing

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
                psi_now[np.abs(psi_now) < 1e-7] = 0     # check for small values (no negatives!)
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
            psi = self.psiFunc.ev(R, Z)
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

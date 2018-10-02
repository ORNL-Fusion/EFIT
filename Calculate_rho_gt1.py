# NAME: CALCULATE_RHO
#
# PURPOSE:
#
#      Calculate square root of normalized toroidal flux, rho, at a given (R,Z)
#      location or for a given value of normalized poloidal flux, psin.  The
#      variable rho is often referred to as the normalized radius.

import numpy as np
import scipy.interpolate as scinter

import EFIT.load_gfile_d3d as loadg
import MDSplus as mds

def main(shot,time,Rin,Zin,psin=None,tree='EFIT01',server='atlas.gat.com'):
    pi = np.pi

    MDSplusCONN = mds.Connection(server)
    parmDICT = loadg.read_g_file_mds(shot, time, tree=tree, connection=MDSplusCONN,
                                     write2file=False)

    if psin is None:
        Rs, Zs = np.meshgrid(parmDICT['R'], parmDICT['Z'])
        R_axis = parmDICT['RmAxis']
        Rs_trunc = Rs > R_axis
        f_psiN = scinter.Rbf(Rs[Rs_trunc], Zs[Rs_trunc], parmDICT['psiRZn'][Rs_trunc],
                             function='linear')

        psin = f_psiN(Rin,Zin)

#   Calculat toroidal flux and its integral
    tFlx = 0.0
    tFlux = 0.0
    rhox = np.zeros(parmDICT['NR'])
    dpsi = np.abs((parmDICT['psiAxis'] - parmDICT['psiSep']) / (parmDICT['NR']-1.))

    for j in range(int(parmDICT['NR'])):
        tFlx = 2.*pi*(parmDICT['qpsi'][j-1]+parmDICT['qpsi'][j])*(dpsi/2.)
        tFlux = tFlux + tFlx
        rhox[j] = np.sqrt(tFlux/(pi*np.abs(parmDICT['Bt0'])))

#   Normalize rho
    rhox = rhox/np.max(rhox)

#   now use a spline to determine eho at input positions

    xvect = np.arange(parmDICT['NR']) / (parmDICT['NR']-1.)
    rhospl = scinter.interp1d(xvect, rhox,fill_value='extrapolate',kind='slinear')

    rho = rhospl(psin)

    return {'rhoN': rho, 'psiN': psin, 'rhox':rhox,'xvect':xvect}

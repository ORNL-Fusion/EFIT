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


def standalone(shot, time, Rin, Zin, psin=None, tree='EFIT01', server='atlas.gat.com'):
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

        psin = f_psiN(Rin, Zin)

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
    rhospl = scinter.interp1d(xvect, rhox, fill_value='extrapolate', kind='slinear')

    rho = rhospl(psin)

    return {'rhoN': rho, 'psiN': psin}


def main(shot, time, Rin, Zin, psin=None, tree='EFIT01', server='atlas.gat.com'):
    shot = int(shot)
    import h5py

    dict = standalone(shot, time, Rin, Zin, psin=None, tree='EFIT01', server=server)
    tmpNAM = 'pyEFIT_rhoN_psiN_'+str(shot)+'.h5'
    fstore = h5py.File(tmpNAM, 'w')
    ds = fstore.create_dataset('rhoN', np.shape(dict['rhoN']), 'f')
    ds[...] = dict['rhoN']

    ds = fstore.create_dataset('psiN', np.shape(dict['psiN']), 'f')
    ds[...] = dict['psiN']

    fstore.close()
    print("hdf5 saved")


# --- Launch main() ----------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import textwrap
    parser = argparse.ArgumentParser(description='Collects an EFIT gfile and calcs rhoN',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent('''\
                                     Examples: 166025 2500 --Rin 2.20 2.21 --Zin -.207 -.207 '''))

    parser.add_argument('shot', help='DIII-D shot number', type=float)
    parser.add_argument('time', help='EFIT time', type=float)
    parser.add_argument('--Rin', help='R locations to calculate rhoN, psiN', required=True, nargs='+', default=[])
    parser.add_argument('--Zin', help='Z locations to calculate rhoN, psiN', required=True, nargs='+', default=[])
    parser.add_argument('-psin', help='list of psiN values to use instead of Rin & Zin', type=float, default=None)
    parser.add_argument('-t', '--tree', help='DIII-D EFIT tree', type=str, default='EFIT01')
    parser.add_argument('-s', '--server', help='how to access the altas database', type=str, default='atlas.gat.com')
    args = parser.parse_args()

    main(args.shot, args.time, args.Rin, args.Zin, psin=args.psin, tree=args.tree, server=args.server)

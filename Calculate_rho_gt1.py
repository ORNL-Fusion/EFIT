# NAME: CALCULATE_RHO
#
# PURPOSE:
#
#      Calculate square root of normalized toroidal flux, rho, at a given (R,Z)
#      location or for a given value of normalized poloidal flux, psin.  The
#      variable rho is often referred to as the normalized radius.
import numpy as np
import scipy.interpolate as scinter

import EFIT.equilParams_class as eqCL


def load_EFIT(shot, time, tree=None, server='atlas.gat.com'):
    gfileNam = './g' + format(shot, '06d') + '.' + format(time, '05d')
    paramClass = eqCL.equilParams(gfileNam, tree=tree, server=server)
    paramDICT = paramClass.get_all()
    paramDICT['NR'] = paramClass.g['NR']
    paramDICT['psiAXIS'] = paramClass.g['psiAxis']
    paramDICT['psiSEP'] = paramClass.g['psiSep']
    paramDICT['Bt0'] = paramClass.g['Bt0']

    return paramDICT


def getRHON(paramDICT, Rin=0.0, Zin=0.0, psin=None):

    if psin is None:
        Rs, Zs = np.meshgrid(paramDICT['R'], paramDICT['Z'])
        R_axis = paramDICT['RmAxis']
        Rs_trunc = Rs > R_axis
        f_psiN = scinter.Rbf(Rs[Rs_trunc], Zs[Rs_trunc], paramDICT['psiRZn'][Rs_trunc],
                             function='linear')
        psin = f_psiN(Rin, Zin)

#   Calculat toroidal flux and its integral
    tFlx = 0.0
    tFlux = 0.0
    rhox = np.zeros(paramDICT['NR'])
    dpsi = np.abs((paramDICT['psiAXIS'] - paramDICT['psiSEP']) / (paramDICT['NR'] - 1.))

    for j in range(int(paramDICT['NR'])):
        tFlx = 2. * np.pi * (paramDICT['qprof1D'][j - 1] + paramDICT['qprof1D'][j]) * (dpsi / 2.)
        tFlux = tFlux + tFlx
        rhox[j] = np.sqrt(tFlux / (np.pi * np.abs(paramDICT['Bt0'])))

#   Normalize rho
    rhoN = rhox / np.max(rhox)

#   now use a spline to determine rho at input positions
    xvect = np.arange(paramDICT['NR']) / (paramDICT['NR'] - 1.)
    rhospl = scinter.interp1d(xvect, rhox, fill_value='extrapolate', kind='slinear')
    rhoNspl = scinter.interp1d(xvect, rhoN, fill_value='extrapolate', kind='slinear')
    rho = rhospl(psin)
    rhoN = rhoNspl(psin)

    return {'rhoN': rhoN, 'rho': rho, 'psiN': psin}


def main(shot, time, Rin, Zin, psin=None, tree='EFIT01', server='atlas.gat.com'):
    shot = int(shot)
    import h5py

    EFITdic = load_EFIT(shot, time, tree='EFIT01', server=server)
    dict = getRHON(EFITdic, Rin, Zin, psin=None)
    tmpNAM = 'pyEFIT_rhoN_psiN_' + str(shot) + '.h5'
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
    parser.add_argument('--Rin', help='R locations to calculate rhoN, psiN', required=True,
                        nargs='+', default=[])
    parser.add_argument('--Zin', help='Z locations to calculate rhoN, psiN', required=True,
                        nargs='+', default=[])
    parser.add_argument('-psin', help='list of psiN values to use instead of Rin & Zin',
                        type=float, default=None)
    parser.add_argument('-t', '--tree', help='DIII-D EFIT tree', type=str, default='EFIT01')
    parser.add_argument('-s', '--server', help='how to access the altas database', type=str,
                        default='atlas.gat.com')
    args = parser.parse_args()

    main(args.shot, args.time, args.Rin, args.Zin, psin=args.psin, tree=args.tree,
         server=args.server)

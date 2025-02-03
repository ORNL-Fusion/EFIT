# IMAS_EQ.py
# description:  reads a netcdf, hdf5, json, equilibrium file formatted per IMAS/OMAS
# engineer:     T Looby
# date:         20241030

import os
import numpy as np


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
    def __init__(self):
        """
        Constructor
        """
        self.data = {}
        return

    def readJSON(self, filename, time, psiMult=1.0, BtMult=1.0, IpMult=1.0):
        """
        reads from IMAS JSON and assigns to the parameters we use in the equilParams_class ep object
        """
        #example multipliers 
        #psiMult = -1.0 / (2.0*np.pi)
        #BtMult = -1.0
        #IpMult = -1.0

        import json
        with open(filename, 'r') as file:
            data = json.load(file) 
        
        try:
            tIdx = np.where(np.round(np.array(data['equilibrium']['time']), 8)==time)[0][0]
        except:
            print("Could not find timestep " + str(time) + " in JSON equilibrium dict.  Aborting.")
            return

        eqt = data['equilibrium']['time_slice'][tIdx]
        wall = data['wall']        

        d = {}
        #ep object name left of '='
        d['R1D'] = eqt['profiles_2d'][0]['grid']['dim1']
        d['Z1D'] = eqt['profiles_2d'][0]['grid']['dim2']
        d['nw'] = len(d['R1D'])
        d['nh'] = len(d['Z1D']) 
        d['rcentr'] = data['equilibrium']['vacuum_toroidal_field']['r0']
        d['bcentr'] = data['equilibrium']['vacuum_toroidal_field']['b0'][tIdx] * BtMult
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
        return d
    
    def writeGEQDSK(self, file, g, shot=None, time=None, ep=None):
        """
        writes a new gfile.  user must supply
        file: name of new gfile
        g:    dictionary containing all GEQDSK parameters
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
            f.write('  EFIT    xx/xx/xxxx    #' + str(shot) + '  ' + str(time) + 'ms        ')
            f.write('   3 ' + str(g['NR']) + ' ' + str(g['NZ']) + '\n')
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



    #=====================================================================
    #                       private functions
    #=====================================================================
    # --- _write_array -----------------------
    # write numpy array in format used in g-file:
    # 5 columns, 9 digit float with exponents and no spaces in front of negative numbers
    def _write_array(self, x, f):
        N = len(x)
        rows = int(N/5)  # integer division
        rest = N - 5*rows
        for i in range(rows):
            for j in range(5):
                    f.write('% .9E' % (x[i*5 + j]))
            f.write('\n')
        if(rest > 0):
            for j in range(rest):
                f.write('% .9E' % (x[rows*5 + j]))
            f.write('\n')

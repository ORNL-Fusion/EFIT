# IMAS_EQ.py
# description:  reads a netcdf (or hdf5) equilibrium file formatted per IMAS
# engineer:     T Looby
# date:         20241030

import os
import numpy as np


class netCDF:
    def __init__(self):
        """
        Constructor
        """
        self.data = {}
        return

    def readNetCDF(self, filename, time):
        """
        reads from IMAS netCDF and assigns to the parameters we use in the equilParams_class ep object
        """
        import netCDF4

        nc = netCDF4.Dataset(filename)
        #commenting for now as the netcdfs do not contain equilibrium.time
        #tIdx = np.where(nc['equilibrium']['time_slice']==time)[0]
        tIdx = str(time)

        eqt = nc['equilibrium']['time_slice'][tIdx]

        wall = nc['wall']

        d = {}

        #ep object name left of '='
        d['rcentr'] = nc['equilibrium']['vacuum_toroidal_field'].variables['r0'][...].item()
        d['bcentr'] = np.array(nc['equilibrium']['vacuum_toroidal_field'].variables['b0'][...])[0]
        d['rmaxis'] = eqt['global_quantities']['magnetic_axis']['r'][...].item()
        d['zmaxis'] = eqt['global_quantities']['magnetic_axis']['z'][...].item()

        print(d)


        #current version
        d['Rmin'] = np.min(eqt['coordinate_system']['grid']['dim1'])
        d['Rmax'] = np.max(eqt['coordinate_system']['grid']['dim1'])
        #future version
        #d['Rmin'] = np.min(eqt['profiles_2d'][0]['r'])
        #d['Rmax'] = np.max(eqt['profiles_2d'][0]['r'])
        d['Rlcfs'] = eqt.boundary.lcfs.r #should be changed to boundary.outline.r
        d['Zlcfs'] = eqt.boundary.lcfs.z #should be changed to boundary.outline.z
        d['Rbdry'] = np.max(d['Rlcfs'])
        d['Zmin'] = np.min(eqt.coordinate_system.grid.dim2)
        d['Zmax'] = np.max(eqt.coordinate_system.grid.dim2)
        d['Zlowest'] = np.min(d['Zbdry'])
        d['siAxis'] = eqt.global_quantities.psi_axis
        d['siBry'] = eqt.global_quantities.psi_boundary
        d['fpol'] = eqt.profiles_1d.f
        d['ffprime'] = eqt.profiles_1d.f_df_dpsi
        d['pprime'] = eqt.profiles_1d.dpressure_dpsi
        d['pres'] = eqt.profiles_1d.pressure
        d['qpsi'] = eqt.profiles_1d.q
        d['psirz'] = eqt.profiles_2d[0].psi
        d['lcfs'] = np.vstack((d['Rlcfs'], d['Zlcfs'])).T
        d['Rwall'] = wall.description_2d[:].limiter.unit[:].outline.r
        d['Zwall'] = wall.description_2d[:].limiter.unit[:].outline.z
        d['wall'] = np.vstack((d['Rwall'], d['Zwall'])).T
        d['rGrid'] = eqt.coordinate_system.r
        d['zGrid'] = eqt.coordinate_system.z 
        d['rdim'] = np.max(d['rGrid']) - np.min(d['rGrid'])
        d['zdim'] = np.max(d['zGrid']) - np.min(d['zGrid'])
        d['R0'] = eqt.global_quantities.magnetic_axis.r
        d['R1'] = np.min(eqt.coordinate_system.grid.dim1)
        d['Zmid'] = 0.0
        d['Ip'] = eqt.global_quantities.ip
        d['nw'] = len(d['rGrid'][:,0])
        d['nh'] = len(d['rGrid'][0,:])
        d['thetapnts'] = 2*d['nw']

        nc.close()

        return d
    


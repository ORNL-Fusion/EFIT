# IMAS_netCDF.py
# description:  reads a netcdf equilibrium file formatted per IMAS
# engineer:     T Looby
# date:         20241030

import os
import numpy as np
import netCDF4

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
        nc = netCDF4.Dataset(filename)
        tIdx = np.where(nc.equilibrium.time==time)[0]
        eqt = nc['equilibrium'].time_slice[tIdx]
        wall = nc['wall']

        d = {}

        #ep object name left of '='
        d['rcentr'] = eqt.vacuum_toroidal_field.r0[tIdx]
        d['bcentr'] = eqt.vacuum_toroidal_field.b0[tIdx]
        d['rmaxis'] = eqt.global_quantities.magnetic_axis.r
        d['zmaxis'] = eqt.global_quantities.magnetic_axis.z
        d['Rmin'] = np.min(eqt.coordinate_system.grid.dim1)
        d['Rmax'] = np.max(eqt.coordinate_system.grid.dim1)
        d['Rlcfs'] = eqt.boundary.lcfs.r
        d['Zlcfs'] = eqt.boundary.lcfs.z
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
        d['psirz'] = eqt.ggd.psi.values
        d['lcfs'] = np.vstack((d['Rlcfs'], d['Zlcfs'])).T
        d['Rwall'] = wall.description_2d[:].limiter.unit[:].outline.r
        d['Zwall'] = wall.description_2d[:].limiter.unit[:].outline.r
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

        return d
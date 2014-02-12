import numpy as np
import EFIT.equilParams_class as ePc
reload(ePc)

def equilParams(gfileNam, nw = 0, nh = 0, thetapnts = 0, grid2G = True):
	# initiate class ans store R, Z and psi
	ep = ePc.equilParams(gfileNam, nw, nh, thetapnts, grid2G)
	paramDICT={'Rs1D':ep.RZdict['Rs1D'], 'Zs1D':ep.RZdict['Zs1D'], 'theta':ep.theta,
			   'psi2D':ep.PSIdict['psi2D'], 'psiN1D':ep.PSIdict['psiN1D'], 'psiN_2D':ep.PSIdict['psiN_2D']}
	
	# Prepare all flux surfaces
	FluxSurList = ep.get_allFluxSur()
	
	# 2-D B-field properties
	Bdict = ep.getBs_2D(FluxSurList)	
	paramDICT['Rs_hold2D'] = Bdict['Rs_2D']
	paramDICT['Zs_hold2D'] = Bdict['Zs_2D']
	paramDICT['Btot_2D'] = Bdict['Btot_2D']
	paramDICT['Bp_2D'] = Bdict['Bp_2D']
	paramDICT['Bt_2D'] = Bdict['Bt_2D']
	
	# 2-D local shear
	SHEARdict = ep.get_Curv_Shear(FluxSurList, Bdict) 
	paramDICT['curvNorm_2D'] = SHEARdict['curvNorm_2D']
	paramDICT['curvGeo_2D'] = SHEARdict['curvGeo_2D']
	paramDICT['S_l_2D'] = SHEARdict['localShear_2D']

	# 1-D current density
	Jdict = ep.cur_density(FluxSurList)
	paramDICT['jpar1D'] = Jdict['jpar']
	paramDICT['jtor1D'] = Jdict['jtor']

	# q profile
	qprof1D = ep.PROFdict['qfunc'](ep.PSIdict['psiN1D'])
	paramDICT['qprof1D'] = qprof1D

	# pressure profile
	press1D = ep.PROFdict['pfunc'](ep.PSIdict['psiN1D'])
	paramDICT['press1D'] = press1D

	# toroidal field profile
	btor1D = ep.PROFdict['ffunc'](ep.PSIdict['psiN1D'])/ep.Rsminor
	paramDICT['btor1D'] = btor1D

	# toroidal flux: psitor from Canik's g3d.pro
	paramDICT['psitorN1D'] = ep.getTorPsi(qprof1D)

	return paramDICT

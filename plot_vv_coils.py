import numpy as np

base = 100.

# original locations
RVVIN_o = np.array([95.748, 95.748, 112.210, 168.285, 216.931, 242.824, 242.824, 242.824, 216.931, 183.657, 112.006, 95.748, 95.748]) / base

ZVVIN_o = np.array([0.0001,  126.083, 142.545, 142.545, 103.195, 40.310, 0.0, -39.916, -102.799, -142.342, -142.342, -126.083, -0.0001]) / base

RVVOUT_o = np.array([92.573,  92.573, 110.632, 169.634, 220.083, 245.364, 245.364, 245.364, 220.228, 185.430, 110.429,  92.573,  92.573]) / base

ZVVOUT_o = np.array([0.0001,  128.295, 146.355, 146.355, 105.545,  44.148,  0.0,	-43.754,-104.798,-146.152,-146.152, -128.295, -0.0001]) / base


# # updated locations
RVVIN = np.array([95.748, 112.210, 168.285, 216.931, 242.824, 242.824, 216.931, 183.657, 112.006, 95.748, 95.748]) / base

ZVVIN = np.array([126.083, 142.545, 142.545, 103.195, 40.310, -39.916, -102.799, -142.342, -142.342, -126.083, 126.083]) / base

RVVOUT = np.array([92.573, 110.632, 169.634, 220.083, 245.364, 245.364, 220.228, 185.430, 110.429,  92.573,  92.573]) / base

ZVVOUT = np.array([128.295, 146.355, 146.355, 105.545,  44.148, -43.754,-104.798,-146.152,-146.152, -128.295, 128.295]) / base

# Icoils
RIC_up = np.array([2.269-.2753*np.sin(22.4*np.pi/180), 2.269+.2753*np.sin(22.4*np.pi/180)])
ZIC_up = np.array([0.7575+.2753*np.cos(22.4*np.pi/180), 0.7575-.2753*np.cos(22.4*np.pi/180)])

RIC_dw = np.array([2.269+.2753*np.sin(22.4*np.pi/180), 2.269-.2753*np.sin(22.4*np.pi/180)])
ZIC_dw = np.array([-0.7575+.2753*np.cos(22.4*np.pi/180), -0.7575-.2753*np.cos(22.4*np.pi/180)])

# Ccoils
RCC = np.array([3.2, 3.2])
ZCC = np.array([0.8, -0.8])

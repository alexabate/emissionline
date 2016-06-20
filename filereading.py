from pylab import *
import numpy as np 
import scipy.spatial as ssp


def read_ems_basis(PCAdir):
	PCAbasis_ems = np.zeros((11,5))
	with open(PCAdir) as f:
	    lines = [line for line in f if not line.startswith('#')]
	    FH = np.loadtxt(lines)
	PCAbasis_ems[0,:] = FH[7,:] #H alpha
	PCAbasis_ems[1,:] = FH[3,:] #H beta
	PCAbasis_ems[2,:] = FH[9,:] #SII 6718
	PCAbasis_ems[3,:] = FH[10,:]#SII 6733
	PCAbasis_ems[4,:] = FH[2,:] #H gamma
	PCAbasis_ems[5,:] = FH[0,:] #OII 3727
	PCAbasis_ems[6,:] = FH[5,:] #O 5008
	PCAbasis_ems[7,:] = FH[4,:] #O 4960
	PCAbasis_ems[8,:] = FH[8,:] #N 6585
	PCAbasis_ems[9,:] = FH[6,:] #N 6550
	PCAbasis_ems[10,:] = FH[1,:]#OII 3730
	return PCAbasis_ems

def read_file(datadir, cols):
	"""

	:Parameters:
	-'datadir':data directory
	-cols:selected feature space
	"""
	total_samples = np.loadtxt(datadir)
	subsample = total_samples[:,cols]
	return subsample


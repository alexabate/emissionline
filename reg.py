from pylab import *
import numpy as np 
import scipy.spatial as ssp
from scipy.spatial.distance import pdist
from scipy import interp
from filereading import *
from compare import *
import scipy.ndimage.filters as flt
import sys


"""
An ancillary function which is to normalize the spectra according to Beck's method.
xhat: wavelength
yhat: flux
"""
def norm_rules(xhat, yhat):
	medlist = [[4250,4300], [4600,4800], [5400,5500], [5600,5800]]
	medrec = np.zeros(len(medlist))
	for jj in np.arange(len(medlist)):
		m = medlist[jj]
		mmin = m[0]
		mmax = m[1]
		sinds = np.logical_and((xhat>mmin) , (xhat<mmax))
		medrec[jj] = np.median(yhat[sinds])
	tarmean = np.mean(medrec)
	yhat = 1.0/tarmean * yhat
	return yhat

"""
Process the original spectra in the folder Spectra/, seperate the continua and degrade its resolution. Save them into folder spectra_proc/
wlinp.txt: wavelength grid to be intepolated. Say, given the SDSS spectra, we may want to intepolate along the wavelength of Brown SEDs.
galaxy_sample.txt: 13k SDSS galaxies. 

NOTICE: this function is not actually used in this demo, since the Spectra folder is too large. 
"""

def inp_spec():
	nn = 13788
	xhat = np.loadtxt('wlinp.txt')	
	nx = len(xhat)
	lx = np.max(xhat) - np.min(xhat)
	sigma   = 8.0/2.5*nx/lx
	# the mask size is 20A
	filer = [[3716,3738],[4330,4350],[4851,4871],[4949,4969],[4997,5017],[6540,6600],[6706,6726],[6721,6741]]
	names = np.loadtxt('galaxy_sample.txt')[:,0]
	snmatrix = np.zeros((nn,len(peakws)))
	nsmatrix = np.zeros((nn, len(peakws)))

	for ii in 1+np.arange(nn):
		if(ii%100==0):
			print "finishing ",100.0*float(ii)/nn,"%"

		fstr = 'spec_' + '%020d'%(names[ii-1]) + '.txt'
		fsavstr = 'spec_' + '%06d'%(ii) + '.txt'
		sampledir = 'Spectra/' + fstr
		with open(sampledir) as f:
		    lines = [line for line in f if not line.startswith('#')]
		    FH = np.loadtxt(lines)
		    cpspec = FH
		cpspec[:,1] = flt.gaussian_filter1d(cpspec[:,1], sigma=sigma)
		for f in filer:
			fmin = f[0]
			fmax = f[1]
			sinds = np.logical_or((cpspec[:,0]<fmin) , (cpspec[:,0]>fmax))
			cpspec = cpspec[sinds,:]	
			finp = finp[sinds]	
		cpspec[:,1] = flt.median_filter(cpspec[:,1], size = 10)

		yout = cpspec[:,1]
		xin = cpspec[:,0]
		yhat = interp(xhat, xin, yout)
		yhat = norm_rules(xhat,yhat)
		np.savetxt('spectra_proc/'+fsavstr,np.hstack((np.transpose(np.array([xhat])),np.transpose(np.array([yhat])))))

"""
Read the processed/nommalized continua in the spectra_proc/ and save them as one matrix: spec_all.txt
spec_all n*m matrix, n is the number of spectra, m is the flux 

NOTICE: this function is not actually used in this demo. I provide a copy of spec_all.txt directly.
"""

def sum_spec():
	nn = 13788
	# the following line is subjected to change.
	xhat = np.loadtxt('wlinp.txt')

	nx = len(xhat)
	spec_all = np.zeros((nn,nx))
	for ii in 1+np.arange(nn):
		if(ii%10==0):
			print "finishing ",100.0*float(ii)/nn,"%"

		fstr = 'spec_' + '%06d'%(ii) + '.txt'
		sampledir = 'spectra_proc/' + fstr
		with open(sampledir) as f:
		    lines = [line for line in f if not line.startswith('#')]
		    FH = np.loadtxt(lines)
		spec_all[ii-1,:] = FH[:,1]
	np.savetxt('spec_all.txt',spec_all)

"""
compute the eigenvector for the continua matrix
save them into mycontPCA.txt
xmean: mean luminosity
U,s,V: X = UsV^T, save as s.txt, v.txt
"""

def compute_eig_vec():
	X = np.loadtxt('spec_all.txt')
	print 'loading finished'
	Xorg = X
	xmean = np.mean(X,axis = 0)
	#centerize
	X  = X - xmean
	print 'starting SVD'
	U,s, V = np.linalg.svd(X, full_matrices=True)
	print 'finishing'
	#np.savetxt('s.txt',s)
	#v is the principle direction.
	#np.savetxt('v.txt',V)
	ndims = 50
	proj = np.dot(X,np.transpose(V))
	proj = proj[:,0:ndims]
	np.savetxt('mycontPCA.txt',proj)

	return proj


"""
following 3 functions are used as tools local linear regression. We can also use scipy/sklearn to avoid writing our own.
"""

"""
Define guassian kernel: exp{(x-x0)^2/2c^2}
"""
def k_guassian(x, x0, c, a=1.0):
    dif= x - x0
    return a * np.exp(dif * dif.T/(-2.0 * c**2))


"""
Calculte the weight matrix for between certain datapoint and the inputs data (or, the training data X)
inputs: n*m matrix, n is the total number of data, m is the dimension of the data
datapoint: query test datapoint
weights: returned weight matrix
"""
def get_weights(inputs, datapoint, c=1.0):
	x = np.mat(inputs)
	n_rows = x.shape[0]
	#create a small value to avoid underflow
	small_underflow = np.nextafter(0,1)
	weights = np.mat(np.eye(n_rows))
	for i in xrange(n_rows):
	    #weights[i, i] = k_guassian(datapoint, x[i], c)+ small_underflow
	    weights[i,i] = 1/np.sqrt((datapoint-x[i])*(datapoint-x[i]).T)
	return weights


"""
Local weighted regressuon function
training_inputs: training data X
training_outputs: training data Y
datapoint: query test datapoint X
yhat: predicted test datapoint Y
sumw: not important, just sum of the weights
"""
def lwr_fun(training_inputs, training_outputs, datapoint, c=1.0):
    weights = get_weights(training_inputs, datapoint, c=c)
    nn = weights.shape[0]
    sumw = 0
    maxw = 0
    for i in xrange(nn):
    	if weights[i,i]>maxw:
    		maxw = weights[i,i]
    	sumw = sumw + weights[i,i]
    for i in xrange(nn):
    	weights[i,i] = weights[i,i]/sumw
    x = np.mat(training_inputs)
    y = np.mat(training_outputs)
    xt = x.T * (weights * x)
    if (np.max(weights)>0.999):
    	ind_1D = np.argmax(weights)
    	ind_2D = np.unravel_index(np.argmax(weights),(nn,nn))
    	yhat = y[ind_2D[0]]
    else:
    	xtI = np.linalg.pinv(xt)
    	betas = xtI * (x.T * (weights * y))   
    	yhat = datapoint * betas

    return yhat,sumw


"""
main function for regression.
galaxy_sample.txt: a file provided by Beck which gives the logEW for SDSS galaxies
est_ems.txt: estimation for the emission lines
real_ems.txt: real value for the emission lines
test_ems.txt: Beck's estimation for the emission lines, use as a sanity check 
estwt.txt: not important, saved weight
"""

def main():
	cols= np.arange(59)
	datadir = 'galaxy_sample.txt'
	subsample = read_file(datadir, cols)
	inds = 19 + 4*np.arange(10)
	contPCA = compute_eig_vec()
	contPCA = contPCA[:,0:5]

	tempPCA = np.ones((contPCA.shape[0],contPCA.shape[1]+1))
	tempPCA[:,0:contPCA.shape[1]] = contPCA
	contPCA = tempPCA
	ems = subsample[:,inds]
	testem = subsample[:,inds+1]
	nn = len(contPCA)
	mytree = ssp.KDTree(contPCA)
	i = 0
	k = 30
	estem = np.zeros((nn,10))
	estwt= np.zeros(nn)

	for i in np.arange(nn):
		tempPCA = contPCA[i,:]
		_,tempind = mytree.query(tempPCA, k+1)
		nb_contPCA = contPCA[tempind[1:k+1],:]
		nb_ems = ems[tempind[1:k+1],:]
		estem[i],estwt[i] = lwr_fun(nb_contPCA, nb_ems, tempPCA, c=0.1)
		#estem[i] = rhks_predict(nb_contPCA, nb_ems, tempPCA, l=0.1)
		if(i%1000==0):
			print "finishing ",100.0*float(i)/nn,"%"
	
	np.savetxt('est_ems.txt',estem)
	np.savetxt('real_ems.txt',ems)
	np.savetxt('test_ems.txt',testem)
	print np.mean(estwt)





if __name__ == '__main__':
	datadir = 'galaxy_sample.txt'
	nsample = len(read_file(datadir, [1]))
	main()
	#plot the emission line comparison and BPT diagram
	all_compare_11(nsample)







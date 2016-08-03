# emissionline
This code is to predict the emission line log(EW) from the continuum.

Execute reg.py to start the demo.

Input files:
wlinp.txt: wavelength grid to be intepolated. Say, given the SDSS spectra, we may want to intepolate along the wavelength of Brown SEDs.
galaxy_sample.txt: 13k SDSS galaxies by Beck et al (2016). Check http://www.vo.elte.hu/papers/2015/emissionlines/ for datails.
spec_all.txt: all the spectra n*m matrix, n is the number of spectra (13k), m is the flux corresponding to the wavelength grid.

Output files:
est_ems.txt: estimation for the emission lines
real_ems.txt: real value for the emission lines
test_ems.txt: Beck's estimation for the emission lines, use as a sanity check 

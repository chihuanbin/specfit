import numpy as np
import matplotlib.pyplot as plt 
import model_fit_tools_v2 as mft
import emcee 
from glob import glob

def redres(wl, spec, factor):

	new_wl = []
	new_spec = []

	for n in range(len(wl)):
		if n % factor == 0:
			new_wl.append(wl[n])
			new_spec.append(spec[n])
		else:
			idx = int((n - (n % factor))/factor)
			new_spec[idx] += spec[n]
	return new_wl, new_spec

for n in range(10):
	fname = glob('run4/specs/spec_{}.txt'.format(n))[0]

	data = np.loadtxt(fname)
	fname = fname.split('/')[-1]
	if len(data) > 0:

		data_wl = data[:,0]
		data_spec = data[:,1]

		ndim, nburn, nsteps, nwalkers = 2, 100, 400, 40

		p0  = emcee.utils.sample_ball([4000, 4], [200, 0.2], size=(nwalkers))

		mft.run_emcee(fname, nwalkers, nsteps, ndim, nburn, p0, 2, 0, data, 3000, [min(data_wl), max(data_wl)], [0.1, 10], no = False)

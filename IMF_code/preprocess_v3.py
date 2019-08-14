import numpy as np
import matplotlib.pyplot as plt 
import model_fit_tools_v3 as mft
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

for n in range(5):
	fname = glob('run4/specs/spec_{}.txt'.format(n))[0]

	data = np.loadtxt(fname)

	data_wl = data[:,0]
	data_spec = data[:,1]

	cen_wl = data_wl[int(len(data_wl)/2)]

	p0 = [4000, 4.5]

	nspec, ndust = 1, 0

	nwalkers, t, tm, alpha, niter = 4, 1, 0.1, 0.1, 4

	mft.run_sim_anneal(nwalkers, p0, t, tm, alpha, niter, nspec, ndust, [data_wl, data_spec], [0.25, cen_wl], 3000, [min(data_wl), max(data_wl)], w = 'aa', pys = False, du = False, no = False)
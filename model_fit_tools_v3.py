import model_fit_tools_v2 as mft
import emcee
import numpy as np
import matplotlib.pyplot as plt
from random import random
import multiprocessing as mp

def acc_prob(current_sol, test_sol, temp):
	if np.isnan(test_sol) or np.isinf(np.abs(test_sol)):
		return 0
	else:
		p = np.exp((test_sol - current_sol)/ temp)
		if p > 1:
			return 1
		else:
			return p

def neighbor(p0, T, bl, ul, seed):
	'''
	bl is bottom limit array (same length as p0)
	ul is upper limit array (same length as p0)
	(for parameters)
	'''
	np.random.seed(seed)
	r = np.random.randint(len(p0))
	p0[r] = np.random.uniform(bl[r], ul[r])
	return p0


#cost function is mft.logposterior(p0, nspec, ndust, data, flux_ratio, broadening, r, wu = 'aa', pysyn = False, dust = False, norm = True)
#which returns a chi square value. P0 is the input parameters, which you can draw randomly

def sim_anneal(walker, p0, T, T_min, alpha, niter, nspec, ndust, data, flux_ratio, broadening, r, wu = 'aa', pysyn = False, dust = False, norm = True):
	'''
	input: sol is the input set of parameters to produce a spectrum
			then we need to compute the chi square using mft.logposterior.
			Also input a starting temp, minimum temp, an alpha factor,
			and a number of iterations per test
	Then: compute chi square using input parameters -- 
			sol should be an array containing all the inputs for logposterior.
			Then compute a neighbor using a randomly produced neighbor, and compute the cs there.
			then test the acceptance probability. 

	eventually return the final answer.
	'''
	seed = int(np.random.uniform(1927018 * walker))

	cs = mft.logposterior(p0, nspec, ndust, data, flux_ratio, broadening, r, wu = wu, pysyn = pysyn, dust = dust, norm = norm)
	if np.isnan(cs) or np.isinf(np.abs(cs)):
		cs = np.inf
	best_chi = cs
	best_p0 = p0

	p = p0
	c = [cs]
	temps = [T]

	while T > T_min:
		i = 0
		while i <= niter:
			new_p0 = neighbor(p0, T, [2000, 2000, 2, 2], [10000, 10000, 5.5, 5.5], seed)
			new_cs = mft.logposterior(new_p0, nspec, ndust, data, flux_ratio, broadening, r, wu = wu, pysyn = pysyn, dust = dust, norm = norm)


			if np.isnan(new_cs) or np.isinf(np.abs(new_cs)):
				new_cs = np.inf

			if new_cs < cs:
				p0 = new_p0
				cs = new_cs
			else:
				ap = acc_prob(cs, new_cs, T)
				if ap > np.random.uniform():
					p0 = new_p0
					cs = new_cs
			if cs < best_chi:
				best_chi = cs
				best_p0 = p0  
			np.vstack((p, p0))
			c.append(cs)
			temps.append(T)            
			i += 1
		T = T*alpha

	np.savetxt('params_walker{}.txt'.format(walker), np.vstack((temps, c, p)), header='# Temperature, chi-square, T1, T2, log g 1, log g 2')
	return np.hstack((best_chi, best_p0))

def run_sim_anneal(nwalkers, p0, T, T_min, alpha, niter, nspec, ndust, data,\
	flux_ratio, broadening, r, w = 'aa', pys = False, du = False, no = True):
	
	pool = mp.Pool()
	results = [pool.apply_async(sim_anneal, (n, p0, T, T_min, alpha, niter, nspec, ndust, data, flux_ratio, broadening, r), dict(wu = w, pysyn = pys, dust = du, norm = no)) for n in range(nwalkers)]
	out = [p.get() for p in results]

	#print('Writing file')
	np.savetxt('results/model_fit_pars.txt', out, fmt = '%.8f')
	

data_wl, data_spec = np.genfromtxt('Data/Spectra_for_Kendall/fftau_wifes_spec.csv', delimiter=',', unpack = True)

data_wl = data_wl[1:-1]/1e4
data_spec = data_spec[1:-1]

cen_wl = data_wl[int(len(data_wl)/2)]

p0 = [4100, 3600, 3.2, 3.5]

nspec, ndust = 2, 0

nwalkers, t, tm, alpha, niter = 4, 1, 0.1, 0.9, 10

run_sim_anneal(nwalkers, p0, t, tm, alpha, niter, nspec, ndust, [data_wl, data_spec], [0.25, cen_wl], 3000, [min(data_wl), max(data_wl)], w = 'um', pys = False, du = False, no = True)

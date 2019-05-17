#Kendall Sullivan

#EMCEE VERSION OF MCMC CODE

#TO DO: Write add_disk function, disk/dust to possible fit params

import numpy as np
#import pysynphot as ps
import matplotlib.pyplot as plt
from astropy.io import fits
import os 
from glob import glob
from astropy import units as u
#from matplotlib import rc
from itertools import permutations 
import time, sys
import scipy.stats
import multiprocessing as mp
import timeit
from PyAstronomy import pyasl
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy import ndimage
import emcee
import corner

def update_progress(progress):
	'''
	update_progress() : Displays or updates a console progress bar
	Accepts a float between 0 and 1. Any int will be converted to a float.
	A value under 0 represents a 'halt'.
	A value at 1 or bigger represents 100%
	'''
	barLength = 10 # Modify this to change the length of the progress bar
	status = ""
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"
	block = int(round(barLength*progress))
	text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()

def bccorr(wl, bcvel, radvel):
	'''
	input: wavelength vector, a barycentric or heliocentric velocity, and a systemic radial velocity.
	**Velocities in km/s**
	If system RV isn't known, that value can be zero.
	outputs: a wavelength vector corrected for barycentric and radial velocities.
	'''
	lam_corr = []
	for w in wl:
		lam_corr.append(w * (1. + (bcvel - radvel)/3e5))
	return lam_corr
	
def plots(wave, flux, l, lw=1, labels=True, xscale='log', yscale='log', save=False):
	'''
	make a basic plot - input a list of wave and flux arrays, and a label array for the legend
	if you want to label your axes, set labels=True and enter them interaactively
	you can also set xscale and yscale to what you want, and set it to save if you'd like
	Natively creates a log-log plot with labels but doesn't save it.
	'''
	fig, ax = plt.subplots()
	for n in range(len(wave)):
		ax.plot(wave[n], flux[n], label = l[n], linewidth=lw)
	if labels == True:
		ax.set_xlabel(r'{}'.format(input('xlabel? ')), fontsize=13)
		ax.set_ylabel(r'{}'.format(input('ylabel? ')), fontsize=13)
		ax.set_title(r'{}'.format(input('title? ')), fontsize=15)
	ax.tick_params(which='both', labelsize='larger')
	ax.set_xscale(xscale)
	ax.set_yscale(yscale)
	ax.legend()

	plt.show()
	if save == True:
		plt.savefig('{}.pdf'.format(input('title? ')))

def find_nearest(array, value):
	'''
	finds index in array such that the array component at the returned index is closest to the desired value.
	Input: array, value
	output: index at which array is closest to value
	'''
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def shift(wl, spec, rv, bcarr, **kwargs):
	'''
	for bccorr, use bcarr as well, which should be EITHER:
	1) the pure barycentric velocity calculated elsewhere OR
	2) a dictionary with the following entries (all as floats, except the observatory name code, if using): 
	{'ra': RA (deg), 'dec': dec (deg), 'obs': observatory name or location of observatory, 'date': JD of midpoint of observation}
	The observatory can either be an observatory code as recognized in the PyAstronomy.pyasl.Observatory list of observatories,
	or an array containing longitude, latitude (both in deg) and altitude (in meters), in that order.

	To see a list of observatory codes use "PyAstronomy.pyasl.listobservatories()".
	'''
	if len(bcarr) == 1:
		bcvel = bcarr[0]
	if len(bcarr) > 1:
		if isinstance(bcarr['obs'], str):
			try:
				ob = pyasl.observatory(bcarr['obs'])
			except:
				print('This observatory code didn\'t work. Try help(shift) for more information')
			lon, lat, alt = ob['longitude'], ob['latitude'], ob['altitude']
		if np.isarray(bcarr['obs']):
			lon, lat, alt = bcarr['obs'][0], bcarr['obs'][1], bcarr['obs'][2]
		bcvel = pyasl.helcorr(lon, lat, alt, bcarr['ra'], bcarr['dec'], bcarr['date'])[0]

	wl = bccorr()

	return ''

def broaden(even_wl, modelspec_interp, res, vsini, limb, plot = True):
	'''
	input: model wavelength vector, model spectrum vector, star vsin(i), the limb darkening coeffecient, 
	the spectral resolution, and the maximum sigma to extend the resolution gaussian broadening along 
	the spectrum

	output: a tuple containing an evenly spaced wavelength vector spanning the width of the original wavelength 
	range, and a corresponding flux vector
	'''

	#sig = np.mean(even_wl)/res

	broad = pyasl.instrBroadGaussFast(even_wl, modelspec_interp, res, maxsig=5)

	if vsini != 0 and limb != 0:
		rot = pyasl.rotBroad(even_wl, broad, limb, vsini)#, edgeHandling='firstlast')
	else:
		rot = broad

	modelspec_interp = [(modelspec_interp[n] / max(modelspec_interp))  for n in range(len(modelspec_interp))]
	broad = [broad[n]/max(broad) for n in range(len(broad))]
	rot = [(rot[n]/max(rot))  for n in range(len(rot))]

	if plot == True:

		plt.figure()
		plt.plot(even_wl, modelspec_interp, label = 'model')
		plt.plot(even_wl, broad, label = 'broadened')
		plt.plot(even_wl, rot, label = 'rotation')
		plt.legend(loc = 'best')
		plt.xlabel('wavelength (angstroms)')
		plt.ylabel('normalized flux')
		plt.savefig('rotation.pdf')

	return(even_wl, rot)

def rmlines(wl, spec, **kwargs):
	'''
	kwargs: add_lines: to add more lines to the linelist (interactive)
			buff: to change the buffer size, input a float here
				otherwise the buffer size defaults to 15 angstroms
			uni: specifies unit for input spectrum wavelengths (default is microns) [T/F]
			conv: if unit is true, also specify conversion factor (wl = wl * conv) to microns
	'''
	names, transition, wav = np.genfromtxt('linelist.txt', unpack = True, autostrip = True)
	space = 1.5e-3 #15 angstroms -> microns

	for key, value in kwargs.items():
		if key == add_lines:
			wl.append(input('What wavelengths (in microns) do you want to add? '))
		if key == buff:
			space = value
		if key == uni:
			wl = wl * value

	diff = wl[10] - wl[9]

	for line in wav:
		end1 = find_nearest(wl, line-space)
		end2 = find_nearest(wl, line+space)
		if wl[end1] > min(wl) and wl[end2] < max(wl) and (end1, end2)> (0, 0) and (end1, end2) < (len(wl), len(wl)):
			for n in range(len(wl)):
				if wl[n] > wl[end1] and wl[n] < wl[end2] and spec[n] > (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2:
					spec[n] = (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2
	print(len(spec), len(wl))
	return spec

def make_reg(wl, flux, waverange):
	'''
	given some wavelength range as an array, output flux and wavelength vectors within that range 
	input: wavelength and flux vectors, wavelength range array
	output: wavlength and flux vectors within the given range
	TO DO: interpolate instead of just pulling the closest indices
	'''
	min_wl = find_nearest(wl, min(waverange))
	max_wl = find_nearest(wl, max(waverange))
	wlslice = wl[min_wl:max_wl]
	fluxslice = flux[min_wl:max_wl]
	return wlslice, fluxslice

def interp_2_spec(spec1, spec2, ep1, ep2, val, verbose = True):
	'''
	input: two spectrum arrays (fluxes, no wavelengths)
	two endpoints, ep1 and ep2, which are the endpoints of the parameter we're interpolating between
	a value between ep1 and ep2 that we wish to interpolate to
	and verbose can print an error message if the interpolation gets messed up, if true - default is False

	returns: a spectrum without a wavelength parameter
	'''
	ret_arr = []
	for n in range(len(spec1)):
		v = spec1[n] + (spec2[n]-spec1[n]) * (val-ep1) / (ep2-ep1)
		#if verbose == True:
		if np.isnan(v) or np.isinf(v) or v < 0:
			v = 0
				#print('There are undefined values in the interpolation. Here are the input parameters: \n', spec1[n], spec2[n], ep1, ep2, v)
		ret_arr.append(v)
	return ret_arr

def make_varied_param(init, sig):
	'''
	randomly varies a parameter within a gaussian range based on given std deviation
	input: initial value, std deviation of gaussian to draw from
	output: the variation
	'''
	var = np.random.normal(init, sig)
	return var

def get_spec(temp, log_g, reg, metallicity = 0, normalize = True, wlunit = 'aa', pys = False, plot = False):
	#need to add contingency in the homemade interpolation for if metallicity is not zero
	'''
	Creates a spectrum from given parameters, either using the pysynphot utility from STScI or using a homemade interpolation scheme.
	Pysynphot may be slightly more reliable, but the homemade interpolation is more efficient (by a factor of ~2).
	input: temperature value, log(g) value, region array ([start, end]), metallicity (defaults to 0), normalize (defaults to True), 
	wavelength unit (defaults to angstroms ('aa'), also supports microns ('um')), pys (use pysynphot) defaults to False, plot (defaults to False)

	returns: a wavelength array and a flux array, in the specified units, as a tuple. Flux is in units of F_lambda (I think)
	Uses the Phoenix models as the base for calculations. TO DO: add a path variable so that this is more flexible

	'''

	if pys == True:
	#grabs a phoenix spectrum using Icat calls via pysynphot (from STScI) defaults to microns
	#get the spectrum
		sp = ps.Icat('phoenix', temp, metallicity, log_g)
		#put it in flambda units
		sp.convert('flam')
		#make arrays to eventually return so we don't have to deal with subroutines or other types of arrays
		spflux = np.array(sp.flux, dtype='float')
		spwave = np.array(sp.wave, dtype='float')

	if pys == False:
		#we have to:
		#read in the synthetic spectra
		#pick our temperature and log g values (assume metallicity is constant for now)
		#pull a spectrum 

		files = glob('phoenix/phoenixm00/*.fits')
		temps = [int(files[n].split('.')[0].split('_')[1]) for n in range(len(files))]
		temp2 = np.sort(temps)
		idx = find_nearest(temp2, temp)

		if idx + 1 < len(temp2):
			if temp2[idx] - temp == 0:
				idx2 = idx
			elif (temp2[idx-1] - temp) < (temp2[idx +1] - temp):
				idx2 = idx - 1
			else:
				idx2 = idx + 1
		else:
			idx2 = idx

		lgs = np.arange(0, 6, 0.5)
		lg = find_nearest(lgs, log_g)
		if lg + 1 < len(lgs):
			if lgs[lg] - log_g == 0:
				lg2 = lg
			if (lgs[lg-1] - log_g) > (lgs[lg + 1] - log_g):
				lg2 = lg - 1
			else:
				lg2 = lg + 1
		else:
			lg2 = lg

		lg = lgs[lg]
		lg2 = lgs[lg2]

		file1 = []
		if idx != idx2:
			file1 = fits.open(files[np.where(temps == temp2[min(idx, idx2)])[0][0]])[1]
		else:
			file1 = fits.open(files[np.where(temps == temp2[idx])[0][0]])[1]
		file1 = Table.read(file1)
		wl1 = file1['WAVELENGTH']
		t1l1 = file1['g{}'.format(str(int(min(lg, lg2) * 10)).zfill(2))]
		t1l2 = file1['g{}'.format(str(int(max(lg, lg2) * 10)).zfill(2))]

		tl = []
		if idx == idx2 and lg == lg2:
			tl = file1['g{}'.format(str(int(lg * 10)).zfill(2))]

		if idx == idx2 and lg != lg2:
			tl = interp_2_spec(t1l1, t1l2, min(lg, lg2), max(lg, lg2), log_g)


		if idx != idx2 and lg != lg2:
			file2 = fits.open(files[np.where(temps == temp2[max(idx, idx2)])[0][0]])[1]
			file2 = Table.read(file2)
			wl2 = file2['WAVELENGTH']
			t2l1 = file2['g{}'.format(str(int(min(lg, lg2) * 10)).zfill(2))]
			t2l2 = file2['g{}'.format(str(int(max(lg, lg2) * 10)).zfill(2))]


			t1l = interp_2_spec(t1l1, t1l2, min(lg, lg2), max(lg, lg2), log_g)
			#print('t1l: ', t1l)
			t2l = interp_2_spec(t2l1, t2l2, min(lg, lg2), max(lg, lg2), log_g)
			##print('t2l: ', t2l)

			tl = interp_2_spec(t1l, t2l, temps[min(idx, idx2)], temps[max(idx, idx2)], temp)

		if plot == True:
			wl1a, tla = make_reg(wl1, tl, [1e4, 1e5])
			wl1a, t1l1a = make_reg(wl1, t1l1, [1e4, 1e5])
			wl1a, t1l2a = make_reg(wl1, t1l2, [1e4, 1e5])
			plt.loglog(wl1a, tla, label = 'tl')
			plt.loglog(wl1a, t1l1a, label = 't1l1')
			plt.loglog(wl1a, t1l2a, label = 't1l2')
			plt.legend()
			plt.show()
		
		spwave = wl1
		spflux = tl


	reg = [reg[n] * 1e4 for n in range(len(reg))]
	spwave, spflux = make_reg(spwave, spflux, reg)

	#you can choose to normalize
	if normalize == True:
		if len(spflux) > 0:
			if max(spflux) > 0:
				spflux = [spflux[n]/max(spflux) for n in range(len(spflux))]
		else:
			spflux = np.ones(len(spflux))
	#and depending on if you want angstroms ('aa') or microns ('um') returned for wavelength
	#return wavelength and flux as a tuple
	if wlunit == 'aa': #return in angstroms
		return spwave, spflux
	if wlunit == 'um':
		spwave = spwave * 1e-4
		return spwave, spflux
		
def add_spec(wl, spec, flux_ratio, normalize = True):#, waverange):
	'''
	add spectra together given an array of spectra and flux ratios
	TO DO: handle multiple flux ratios in different spectral ranges

	input: wavelength array (of vectors), spectrum array (of vectors), flux ratio array with len = len(spectrum_array) - 1, whether or not to normalize (default is True)
	output: spectra added together with the given flux ratio
	'''
	spec1 = spec[0]
	for n in range(len(flux_ratio)):
		spec2 = [spec[n+1][k] * flux_ratio[n] for k in range(len(spec1))]
		spec1 = [spec1[k] + spec2[k] for k in range(len(spec1))]
	if normalize == True:
	#normalize and return
		spec1 = spec1/max(spec1)
	return spec1

def make_bb_continuum(wl, spec, dust_arr, wl_unit = 'um'):
	'''

	'''
	h = 6.6261e-34 #J * s
	c = 2.998e8 #m/s
	kb = 1.3806e-23 # J/K

	if wl_unit == 'um':
		wl = [wl[n] * 1e-6 for n in range(len(wl))] #convert to meters
	if wl_unit == 'aa':
		wl = [wl[n] * 1e-10 for n in range(len(wl))]

	if type(dust_arr) == float or type(dust_arr) == int:
		pl = [(2 * h * c**2) /((wl[n]**5) * (np.exp((h*c)/(wl[n] * kb * dust_arr)) - 1)) for n in range(len(wl))]

	if type(dust_arr) == np.isarray():
		for temp in dust_arr:
			pl = [(2 * h * c**2) /((wl[n]**5) * (np.exp((h*c)/(wl[n] * kb * temp)) - 1)) for n in range(len(wl))]

			spec = [spec[n] + pl[n] for n in range(len(pl))]
	return spec

def fit_spec(n_walkers, wl, flux, reg, fr, guess_init, sig_init = {'t':[200, 200], 'lg':[0.2, 0.2], 'dust': [100]}, wu='um', burn = 100, cs = 10, steps = 200, pysyn=False, conv = True, dust = False):
	##print(guess_init)
	#does an MCMC to fit a combined model spectrum to an observed single spectrum
	#guess_init and sig_init should be dictionaries of component names and values for the input guess and the 
	#prior standard deviation, respectively. 
	#assumes they have the same metallicity
	#the code will expect an dictionary with values for temperature ('t'), log g ('lg'), and dust ('dust') right now
	#TO DO: add line broadening, disk/dust to possible fit params

	if 'm' in guess_init:
		metal = guess_init['m']
	else:
		metal = 0

	#make some initial guess' primary and secondary spectra, then add them
	wave1, spec1 = get_spec(guess_init['t'][0], guess_init['lg'][0], reg, metallicity = metal, wlunit = wu, pys = pysyn)
	wave2, spec2 = get_spec(guess_init['t'][1], guess_init['lg'][1], reg, metallicity = metal, wlunit = wu, pys = pysyn)

	init_cspec = add_spec(wave1, wave2, spec1, spec2, fr)

	if dust == True:
		init_cspec = add_dust(init_cspec, guess_init['dust'][0])

	#calculate the chi square value of that fit
	init_cs, pval = scipy.stats.chisquare(flux, init_cspec)
	#that becomes your comparison chi square
	chi = init_cs
	#make a random seed based on your number of walkers
	np.random.seed(n_walkers + np.random.randint(2000))

	#savechi will hang on to the chi square value of each fit
	savechi = []

	#sp will hang on to the tested set of parameters at the end of each iteration
	sp = []
	for key in guess_init:
		for l in range(len(guess_init[key])):
			sp.append(guess_init[key][l])

	gi = sp

	si = []
	for key in sig_init:
		for m in range(len(sig_init[key])):
			si.append(sig_init[key][m])

	var_par = sp

	n = 0
	#print('Starting MCMC walker {}....(this might take a while)'.format(n_walkers + 1))
	while n < steps:
		vp = np.random.randint(0, len(var_par))
		var_par[vp] = make_varied_param(var_par[vp], si[vp])
		try:
			if n <= burn:
				n = n + 1

			#make spectrum from varied parameters
			test_wave1, test_spec1 = get_spec(var_par[0], var_par[2], reg, wlunit = wu, pys = pysyn)
			test_wave2, test_spec2 = get_spec(var_par[1], var_par[3], reg, wlunit = wu, pys = pysyn)

			test_cspec = add_spec(test_wave1, test_wave2, test_spec1, test_spec2, fr)

			if dust == True:
				test_cspec = add_dust(test_cspec, var_par[4])

			#calc chi square between data and proposed change
			test_cs, pval = scipy.stats.chisquare(test_cspec, flux)

			lh = np.exp(-1 * (init_cs)/2 + (test_cs)/2)

			u = np.random.uniform(0, 1)

			if chi > test_cs and lh > u:
				gi[vp] = var_par[vp]
				chi = test_cs 

			if n > burn:
				sp = np.vstack((sp, gi))
				savechi.append(chi)
				print(n, chi)
				if conv == True:
					if savechi[-1] <= cs:
						n = steps + burn
						print("Walker {} is done.".format(n_walkers + 1))
					elif savechi[-1] > cs:
						n = burn + 5
					else:
						print('something\'s happening')
		except:
			pass;
	np.savetxt('results/params{}.txt'.format(n_walkers), sp)
	np.savetxt('results/chisq{}.txt'.format(n_walkers), savechi)

	return sp[np.where(savechi == min(savechi))][0]


def run_mcmc(walk, w, flux, regg, fr, temp_vals, lg_vals):
	#use multiple walkers and parallel processing:
	pool = mp.Pool()
	results = [pool.apply_async(fit_spec, args = (walker_num, w, flux,regg, fr, {'t': temp_vals, 'lg':lg_vals} )) for walker_num in range(walk)]
	out = [p.get() for p in results]

	#print('Writing file')
	np.savetxt('results/multi_walkers.txt', out, fmt = '%.8f')

	return

def loglikelihood(p0, nspec, ndust, data, flux_ratio, broadening, r, w = 'aa', pysyn = False, dust = False, norm = True):
	"""
	The natural logarithm of the joint likelihood. 
	Set to the chisquare value. (we want uniform acceptance weighted by the significance)
	
	Possible kwargs are reg (region), wlunit ('um' or 'aa'), dust (defaults to False), \
		normalize (defaults to True), pysyn (defaults to False), 
		For more help 
	Args:
		p0 (list): a sample containing individual parameter values
		Then p0[0: n] = temp, p0[n : 2n] = lg, p0[2n : -1] = dust temps
		data (list): the set of data/observations
	Note:
		We do not include the normalisation constants (as discussed above).

	current options: arbitrary stars, dust (multi-valued). 
	To do: fit for broadening or vsini,  
	"""
	le = len(data[:][0])

	wl = np.zeros(le)
	spec = np.zeros(le)

	for n in range(nspec):
		ww, spex = get_spec(p0[n], p0[nspec + n], normalize = norm, reg = r, wlunit = w, pys = pysyn)

		wl1 = np.linspace(min(ww), max(ww), le)

		if len(spex) == 0:
			spex = np.ones(le)

		intep = scipy.interpolate.interp1d(ww, spex)
		spec1 = intep(wl1)

		wl = np.vstack((wl, wl1))
		spec = np.vstack((spec, spec1))

	test_spec = add_spec(wl, spec, flux_ratio)

	if dust == True:
		test_spec = make_bb_continuum([wl[:][1], test_spec], p0[2 * nspec : -1], wl_unit = w)

	#test_spec = broaden(wl[:][1], test_spec, broadening, 0, 0, plot=False)

	init_cs, pval = scipy.stats.chisquare(data[:][-1], test_spec)
	#print(init_cs)
	if np.isnan(init_cs):
		init_cs = -np.inf

	return init_cs

# WE ASSUME A UNIFORM PRIOR -- add something more sophisticated eventually

def logprior(p0, nspec, ndust):
	temps = p0[0:nspec]
	lgs = p0[nspec:2 * nspec]
	if ndust > 0:
		dust = p0[2 * nspec : 2 * nspec + ndust]
	for p in range(nspec):
		if 2000 < temps[p] < 15000 and 0 < lgs[p] < 5.5:
			return 0.0
		else:
			return -np.inf

def logposterior(p0, nspec, ndust, data, flux_ratio, broadening, r, wu = 'aa', pysyn = False, dust = False, norm = True):
	"""
	The natural logarithm of the joint posterior.

	Args:
		p0 (list): a sample containing individual parameter values/
		data (list): the set of data/observations
	Assuming a uniform prior for now
	"""
	if p0[nspec] <= 5.5 and p0[nspec + 1] <= 5.5 and p0[nspec] > 0 and p0[nspec + 1] > 0:
		lp = logprior(p0, nspec, ndust)
	else:
		lp = -np.inf

	# if the prior is not finite return a probability of zero (log probability of -inf)
	if not np.isfinite(lp):
		return -np.inf

	lh = loglikelihood(p0, nspec, ndust, data, flux_ratio, broadening, r, w = wu, pysyn = False, dust = False, norm = True)
	# return the likeihood times the prior (log likelihood plus the log prior)
	return lp + lh


def run_emcee(nwalkers, nsteps, ndim, pos, nspec, ndust, data, flux_ratio, broadening, r, w = 'aa', pys = False, du = False, no = True):
	'''
	p0 is a dictionary containing the initial guesses for temperature and log g.
	data is the spectrum to fit to
	flux ratio is self-explanatory, but is an array
	r is the region to fit within
	'''

	sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(nspec, ndust, data, flux_ratio, broadening, r), kwargs ={'wu': w, 'pysyn': pys, 'dust': du, 'norm':no}, threads = 4)

	p, prob, state = sampler.run_mcmc(pos, nsteps)
	
	f = open("results/chain.txt", "w")
	f.close()

	for result in sampler.sample(p, iterations=100, storechain=False):
		position = result[0]
		f = open("results/chain.txt", "w")
		for k in range(position.shape[0]):
			f.write("{} {}\n".format(k, str(position[k])))
		f.close()


	for i in range(ndim):
		plt.figure(i)
		plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
		plt.title("Dimension {0:d}".format(i))
		plt.savefig('results/plots/{}.pdf'.format(i))
		plt.close()

		plt.figure(i)
		for n in range(nwalkers):
			plt.plot(np.arange(nsteps),sampler.chain[n, :, i], color = 'k')
		plt.savefig('results/plots/chain_{}.pdf'.format(i))
		plt.close()
	chain = sampler.chain[:, :, 0].T
	N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
	try:
		new = np.empty(len(N))
		for i, n in enumerate(N):
			new[i] = emcee.autocorr.integrated_time(chain[:, :n])

		plt.loglog(N, new, "o-", label="DFM 2017")
		ylim = plt.gca().get_ylim()
		plt.plot(N, N / 50.0, "--k", label="tau = N/50")
		plt.ylim(ylim)
		plt.xlabel("number of samples, N")
		plt.ylabel("tau estimates")
		plt.legend(fontsize=14);
		plt.savefig('results/plots/autocorr.pdf')
		plt.close()
	except:
		pass;
	samples = sampler.chain[:, :, :].reshape((-1, ndim))
	fig = corner.corner(samples)
	fig.savefig("results/plots/triangle.pdf")
	plt.close()

	return("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

"""
.. module:: IMF
   :platform: Unix, Windows
   :synopsis: Synthetic population production

.. moduleauthor:: Kendall Sullivan <kendallsullivan@utexas.edu>

Dependencies: numpy, matplotlib, astropy, scipy, model_fit_tools_v2, emcee
"""

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os 
from glob import glob
from astropy import units as u
from matplotlib import rc
rc('text', usetex=True)
from scipy.stats import chisquare
from scipy.interpolate import interp1d, SmoothBivariateSpline, interp2d, griddata
from scipy.integrate import trapz, simps
import model_fit_tools_v2 as mft
import multiprocessing as mp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.cm as cm

def redres(wl, spec, factor):
	"""Calculates a barycentric velocity correction given a barycentric and/or a radial velocity (set the unused value to zero)

	Args: 
		wl (list): wavelength vector.
		bcvel (float): a barycentric or heliocentric velocity.
		radvel (float): a systemic radial velocity.

	Note:
		Velocities are in km/s.
		If system RV isn't known, that value can be zero.

	Returns: 
		lam_corr (list): a wavelength vector corrected for barycentric and radial velocities.

	"""
	wlnew = []
	specnew = []

	for i in range(len(wl)):
		if i%factor == 0:
			wlnew.append(wl[i])
			specnew.append(spec[i])
		else:
			idx = int((i - i%factor)/factor)
			specnew[idx] += spec[i]
	return wlnew, specnew

def make_salpeter_imf(massrange, exponent):
	imf = []
	for n, m in enumerate(massrange):
		imf.append(4.1e-2 * m**-1.3)
	#imf = [imf[n] / (max(massrange)-min(massrange)) for n in range(len(imf))]
	#imf = [i/max(imf) for i in imf]
	return np.array(imf)

def make_chabrier_imf(massrange):
	#from Chabrier 2003, PASP
	p = []
	for m in massrange:
		if m <= 1:
			#do lognormal
			p.append(0.093*np.exp(-((np.log(m) - np.log(0.3))**2)/(2 * 0.55**2)))
		else:
			#do Salpeter power law: x = 1.3 for the log version
			p.append(9e-3*m**-1.3)
	#p = [p[n]/(max(massrange) - min(massrange)*massrange[n] * np.log(10)) for n in range(len(p))]
	#p = [pn/max(p) for pn in p]
	return np.array(p)

def calc_pct(imf, wh = 'chabrier'):
	x = np.arange(0.08, 100, 0.05)

	total_chab = make_chabrier_imf(x)
	total_sal = make_salpeter_imf(x, 1.3)

	total_chab = [tc/total_chab[19] for tc in total_chab]
	total_sal = [ts/total_sal[19] for ts in total_sal]

	chab = np.trapz(total_chab[0:19], x[0:19])/np.trapz(total_chab, x)
	sal = np.trapz(total_sal[19:-1], x[19:-1])/np.trapz(total_sal, x)
	total = chab+sal

	if imf == 'c':
		return x, total_chab/np.trapz(total_chab, x)
	elif imf == 's':
		return x, total_sal/np.trapz(total_chab, x)
	
	elif imf == 'pct':
		if wh == 'chabrier':
			return chab/total,
		elif wh == 'salpeter':
			return sal/total, 
		else:
			return "You messed something up" 

def get_params(mass, age):
	'''
	input: mass (solar masses) and age (megayears, 1e6). 
	requires evolutionary models in a folder called "isochrones" and atmosphere models in a folder called "phoenix_isos", at the moment
	Uses input mass and age to get physical parameters (luminosity, radius) from Baraffe isochrones, 
	then uses those physical parameters to get a temperature and log g from the phoenix BT-Settl isochrones (to match the model spectra)
	'''
	a = str(int(age * 10)).zfill(5)
	if mass <= 1.4 and age>=1:
		isos = glob('isochrones/*.txt')
		ages = []
		for file in isos:
			ages.append(int((file.split('_')[1]))/10)
		ages = np.sort(ages)

		a1 = mft.find_nearest(ages, age)

		if ages[a1] > age:
			a2 = a1 - 1
		else:
			a2 = a1 + 1

		aa1 = ages[min(a1, a2)]
		aa2 = ages[max(a1, a2)]

		m1, lum1, radius1 = np.genfromtxt(glob('isochrones/*{}*.txt'.format(str(int(aa1 * 10)).zfill(5)))[0], usecols =(0, 2, 4), comments = '!', unpack = True, autostrip = True)
		m2, lum2, radius2 = np.genfromtxt(glob('isochrones/*{}*.txt'.format(str(int(aa2 * 10)).zfill(5)))[0], usecols =(0, 2, 4), comments = '!', unpack = True, autostrip = True)

		aaa1, aaa2 = np.full(len(m1), aa1), np.full(len(m2), aa2)

		lum1, lum2 = [10**l for l in lum1], [10**l for l in lum2]

		a_l = griddata((np.hstack((m1, m2)), np.hstack((aaa1, aaa2))), np.hstack((lum1, lum2)), (mass, age))#interp_2d(mass, age, np.hstack((m1, m2)), np.hstack((aaa1, aaa2)), np.hstack((lum1, lum2)))
		a_r = griddata((np.hstack((m1, m2)), np.hstack((aaa1, aaa2))), np.hstack((radius1, radius2)), (mass, age))#interp_2d(mass, age, np.hstack((m1, m2)), np.hstack((aaa1, aaa2)), np.hstack((radius1, radius2)))
		a_l = np.log10(a_l)

		#units: solar masses, kelvin, solar luminosity, log(g), giga-centimeters (NOT SOLAR RADII)
		#THUS assume solar radius = 6.957e5 km = 6.957e10 cm = 69.75 Gcm
		m_real1, teff1, lu1, logg1, rad1 = np.genfromtxt(glob('phoenix_isos/*{}*.txt'.format(str(int(aa1*10)).zfill(5)))[0], \
			usecols = (0, 1, 2, 3, 4), autostrip = True, unpack = True)

		m_real2, teff2, lu2, logg2, rad2 = np.genfromtxt(glob('phoenix_isos/*{}*.txt'.format(str(int(aa2*10)).zfill(5)))[0], \
			usecols = (0, 1, 2, 3, 4), autostrip = True, unpack = True)

		teff1, lu1, logg1 = teff1[1:-1], lu1[1:-1], logg1[1:-1]
		rad1 = [np.around(r/69.75, 2) for r in rad1] #convert to solar radius

		teff2, lu2, logg2 = teff2[1:-1], lu2[1:-1], logg2[1:-1]
		rad2 = [np.around(r/69.75, 2) for r in rad2] #convert to solar radius

		aaa1, aaa2 = np.full(len(lu1), aa1), np.full(len(lu2), aa2)

		if a_l >= lu1[0] and a_l <= lu1[-1] and a_l >= lu2[0] and a_l <= lu2[-1]:
			a_l, lu1, lu2 = 10**a_l, [10**l for l in lu1], [10**l for l in lu2]
			#temp, log_g = interp_2d(a_l, age, np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2)), np.hstack((teff1, teff2))), interp_2d(a_l, age, np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2)), np.hstack((logg1, logg2)))
			temp = griddata((np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2))), np.hstack((teff1, teff2)), (a_l, age))
			log_g = griddata((np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2))), np.hstack((logg1, logg2)), (a_l, age))
			#a_l = np.log10(a_l)
			return temp, log_g, a_l

		else:
			if a_l > lu1[0] and a_l < lu1[-1]:
				idx = mft.find_nearest(a_l, lu1)
				temp, log_g = teff1[idx], logg1[idx]
			elif a_l > lu2[0] and a_l < lu2[-1]:
				idx = mft.find_nearest(a_l, lu2)
				temp, log_g = teff2[idx], logg2[idx]
			else:
				print('luminosity is out of range, using maximum')
				idx = np.where(np.hstack((lu1, lu2)) == max(np.hstack((lu1, lu2))))
				temp, log_g = np.hstack((teff1, teff2))[idx], np.hstack((logg1, logg2))[idx]
			return temp, log_g, a_l



	else:
		age = np.log10(age * 1e6)
		matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
		ages = matrix[:, 1]
		age_new = ages[mft.find_nearest(ages, age)]
		matrix = matrix[np.where(matrix[:, 1] == age_new)]

		mat = matrix[:,4:8]
		#mat columns are:
		#mass (0), logL (1), logTe (2), logg (3)
		ma = mat[:, 0]
		l_int = interp1d(ma, mat[:, 1])
		t_int = interp1d(ma, mat[:, 2])
		lg_int = interp1d(ma, mat[:, 3])
		a_m = mass
		a_l = 10**l_int(a_m)
		a_t = 10**t_int(a_m)
		logg = lg_int(a_m)
		return a_t, logg, a_l

def fn_pair(mat1, mat2, val1, val2):
	ret1 = [np.abs(float(m) - val1) for m in mat1]
	ret2 = [np.abs(float(m) - val2) for m in mat2]

	min_sum = [ret1[n] + ret2[n] for n in range(len(ret1))]

	idx = np.where(min_sum == min(min_sum))[0][0]
	return idx

def match_pars(temp1, temp2, lf):
	lg = 4

	wl1, spec1 = mft.get_spec(temp1, lg, [0.55, 0.95], normalize = False)
	wl2, spec2 = mft.get_spec(temp2, lg, [0.55, 0.95], normalize = False)

	inte1 = interp1d(wl1, spec1)
	inte2 = interp1d(wl2, spec2)

	wl = np.linspace(max(min(wl1), min(wl2)), min(max(wl1), max(wl2)), max(len(wl2), len(wl2)))
	spec1 = inte1(wl)
	spec2 = inte2(wl)

	wl1, spec1 = mft.broaden(wl, spec1, 3000, 0, 0)
	wl2, spec2 = mft.broaden(wl, spec2, 3000, 0, 0)

	wl1, spec1 = redres(wl1, spec1, 6)
	wl2, spec2 = redres(wl2, spec2, 6)

	spec2 *= lf

	spec = [spec1[n] + spec2[n] for n in range(len(spec1))]

	return wl, spec

def get_secondary_mass(pri_mass):
	#from raghavan 2010 Fig. 16
	rn = np.random.random_sample(size=1)[0]
	if rn < (5/110):
		mr = np.random.uniform(low=0.05, high = 0.2)
	elif rn > 0.9:
		mr = np.random.uniform(low=0.95, high = 1)
	else:
		mr = np.random.uniform(low=0.2, high = 0.95)
	sec_mass = pri_mass * mr
	return sec_mass

def get_distance(sfr):
	if sfr == 'taurus':
		return np.random.randint(low=140, high=160)

def make_mass(n_sys):
	r = np.random.RandomState()#seed = 234632)
	masses = np.linspace(0.08, 3, 400)
	prob_dist = make_chabrier_imf(masses)
	pd = [prob_dist[n]/np.sum(prob_dist) for n in range(len(prob_dist))]
	cumu_dist = [np.sum(pd[:n]) for n in range(len(pd))]

	r_num = r.random_sample(size = n_sys)
	mass = []
	for rn in r_num:
		idx = mft.find_nearest(cumu_dist, rn)
		m = masses[idx]
		mass.append(m)
	return mass

def make_binary_sys(n, n_sys, multiplicity, mass_bins, age, av, run, sfr = 'taurus'):
	#from Raghavan + 2010, eccentricity can be pulled from a uniform distribution
	#From the same source, the period distribution can be approximated as a Gaussian with
	#a peak at log(P) = 5.03 and sigma_log(P) = 2.28

	#decide if my mass will end up in the chabrier or salpeter regime by drawing a random number and comparing to the percentage
	#of the total cmf that is taken by the Chab. portion of the imf
	r = np.random.RandomState()#seed = 234632)

	age = np.random.uniform(low = age[0], high = age[-1])

	pri_array_keys = ['num', 'p_mass', 'multiplicity', 'age', 'av', 'temp', 'logg', 'luminosity', 'distance']
	p_arr_keys2 = ['a multiplicity of 1 indicates a multiple at all - it\'s a flag, not a number of stars in the system\n #distance is in pc']

	kw = ['num', 's_mass', 'sep', 'age', 'eccentricity', 'period', 'temp', 'logg', 'luminosity']

	pri_pars = np.empty(len(pri_array_keys))
	sec_pars = np.empty(len(kw))
	mass = make_mass(n_sys)
	# while n < n_sys:
	# 	print('System:', n)
		# try:
	spec_file = open(os.getcwd() + '/' + run + '/specs/spec_{}.txt'.format(n), 'w')
	ptemp, plogg, plum = get_params(mass[n_sys - 1], age)

	# ptemp = (np.round(ptemp / 100) * 100)
	# plogg = np.round(plogg/0.5) * 0.5

	pri_par = [n, mass[n_sys -1], 0, age, av, ptemp, plogg, plum, get_distance(sfr)] #etc. - can add more later
	pri_pars = np.vstack((pri_pars, np.array(pri_par)))
	if type(pri_par[1]) != None:
		print('initial system parameters: ptemp: ', ptemp, 'plogg: ', plogg, 'mass: ', mass[n_sys -1])
		pri_wl, pri_spec = mft.get_spec(ptemp, plogg, [0.55, 0.95], normalize = False)

		comb_spec = pri_spec
		if mass[n_sys-1] < mass_bins[0]:
			mf = multiplicity[0]
		elif mass[n_sys-1] > mass_bins[-1]:
			mf = multiplicity[-1]
		else:
			for bn in range(1, len(mass_bins) - 1):
				mf = multiplicity[bn + 1]

		num_rand = r.random_sample()

		if mf >= num_rand:
			pri_pars[n_sys][2]= 1

			sec_par = np.empty(len(kw))
			sec_par[0] = n
			sec_par[1] = get_secondary_mass(mass[n_sys - 1])
			sec_par[2] = r.uniform(10, 1e3)
			sec_par[3] = age
			sec_par[4] = r.uniform(0, 1)
			sec_par[5] = r.normal(5.03, 2.28)

			stemp, slogg, slum = get_params(sec_par[1], age)
			sec_par[6:9] = stemp, slogg, slum

			sec_pars = np.vstack((sec_pars, sec_par))

			sec_wl, sec_spec = mft.get_spec(stemp, slogg, [0.55, 0.95], normalize = False)
			comb_spec = [comb_spec[t] + sec_spec[t] for t in range(len(sec_spec))]

		for k in range(len(comb_spec)):
			spec_file.write(str(pri_wl[k]) + ' ' + str(comb_spec[k]) + '\n')
		
		n += 1

	else:
		mass[n] = make_mass(1)

	# except:
	# 	n += 1
	# 	pass;
	return pri_pars, sec_pars

def run_bin(massrange, dm, binary_fractions, mass_div, num_particles, age):
	#use multiple walkers and parallel processing:

	pool = mp.Pool()
	results = [pool.apply_async(make_binary_pop, args = (array(massrange[n-1], massrange[n], massrange[n+1]), \
		binary_fractions, mass_div, num_particles, age)) for n in range(1, len(massrange)-1)]
	out = [p.get() for p in results]

	#print('Writing file')
	np.savetxt('results/multi_walkers.txt', out, fmt = '%.8f')

	return

def plot_imf(mass_range, num_particles, age, binary_fractions, mass_div, new_pop = False, multip = True):
	salpeter = 1.35

	if new_pop == True and multip == True:
		run_bin(mass_range, binary_fractions, mass_div, num_particles, age) 
	if new_pop == True and multip != True:
		make_binary_pop(mass_div, binary_fractions, mass_div, num_particles, age)

	binary_files = glob('binary_files/binary_parameters*.txt')  

	masses = []
	ptemp = []
	stemp = []
	pmass = []
	smass = []
	maxn = 0
	mm = 0

	for f in binary_files:
		p_mass, s_mass, p_temp, s_temp, p_logg, s_logg, log_period, eccentricity, flux_ratio =\
		np.genfromtxt(f, dtype = 'str', autostrip = True, unpack = True, deletechars = '][, ')

		number = len(p_mass)
		'''
		if number > maxn:
			maxn = number
			try:
				mm = float(p_mass[0].strip('[] ,'))
			except:
				pass
		'''

		for n in range(len(p_mass)):
			try: 
				mass_sum = float(p_mass[n].strip(' ][, ')) + float(s_mass[n].strip(' ][, '))

				if mass_sum != 0 and float(p_temp[n].strip(' [], ')) != 0 and float(p_mass[n].strip(' [], ')) != 0:
					masses.append(mass_sum)


					ptemp.append(float(p_temp[n].strip('[], ')))
					pmass.append(float(p_mass[n].strip('[], ')))
					if float(s_mass[n].strip('[],')) != 0:
						stemp.append(float(s_temp[n].strip('[] ,')))
						smass.append(float(s_mass[n].strip('[] ,')))
			except:
				pass;

	mm = np.median(pmass)
	nmr = np.linspace(mass_range[0], mass_range[-1], len(mass_range) * 100)
	cm = make_chabrier_imf(nmr)
	primary_numbers = [int(np.rint(num_particles * p)) * 13 for p in cm]

	fig, ax = plt.subplots()
	ax.hist(masses, bins = np.logspace(np.log10(min(masses)), np.log10(max(masses)), 15), label = 'Single + binary stars', color = 'b')
	ax.hist(pmass, bins = np.logspace(np.log10(min(pmass)), np.log10(max(pmass)), 15), label = 'Single stars', facecolor = 'cyan', alpha = 0.7)
	#ax.hist(smass, bins = np.logspace(np.log10(min(smass)), np.log10(max(smass)), 20), label = 'Secondary', alpha = 0.5, color = 'teal')
	ax.plot(nmr, primary_numbers, label = 'Theoretical IMF', linestyle = '--', color = 'tab:orange')
	plt.axvline(x = mm, label = 'Median stellar mass', color='red')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel(r'Mass ($M_{\odot}$)', fontsize = 14)
	ax.set_ylabel('Number of stars', fontsize = 14)
	ax.set_xlim(0.08, 2.1)
	#ax.set_ylim(1e2, 1e3)
	#ax.set_yscale('log')
	ax.set_xscale('log')
	ax.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig('masshist.pdf')
	plt.show()
	plt.close(fig)

	return

def find_norm(model, data):
	init_guess = max(data)/max(model)

	step = 0.1

	while step > 0.001:
		model_minus = [model[n] * (init_guess - (init_guess * step)) for n in range(len(model))]
		model_init = [model[n] * init_guess for n in range(len(model))]
		model_plus = [model[n] * (init_guess + (init_guess * step)) for n in range(len(model))]

		minus_var = np.mean([m * 0.01 for m in model_minus])
		init_var = np.mean([m * 0.01 for m in model_init])
		plus_var = np.mean([m * 0.01 for m in model_plus])
		
		xs_minus = mft.chisq(data, model_minus, minus_var)
		xs_init = mft.chisq(data, model_init, init_var)
		xs_plus = mft.chisq(data, model_plus, plus_var)

		#print(step, xs_minus, xs_init, xs_plus, init_guess)

		if xs_init < xs_minus and xs_init < xs_plus:
			step *= 0.5
		elif xs_init > xs_minus and xs_init < xs_plus:
			init_guess = init_guess - (init_guess * step)
		elif xs_init < xs_minus and xs_init > xs_plus:
			init_guess = init_guess + (init_guess * step)
		else:
			if xs_minus < xs_plus:
				init_guess = init_guess - (init_guess * step)
			else:
				init_guess = init_guess + (init_guess * step)

	return init_guess

def even_simpler(filename, t_guess, lg_guess, t_range, lg_range):
	wl, spec = np.genfromtxt(filename, unpack = True)

	wle = np.linspace(wl[0], wl[-1], len(wl))
	intep = interp1d(wl, spec)
	spece = intep(wle)

	wle, spece = mft.broaden(wle, spece, 3000, 0, 0)
	#wle, spece = redres(wle, spece, 6)

	xs = []
	temp = []
	logg = []
	norm = []
	st = []

	for l in lg_range:
		t_init = t_guess

		step = 200
		niter = 0

		while step >= 25:
			print(step, t_init)
			t_minus = t_init - step
			t_plus = t_init + step

			ww_minus, ss_minus = mft.get_spec(t_minus, l, [0.55, 0.95], normalize = False)
			ww_init, ss_init = mft.get_spec(t_init, l, [0.55, 0.95], normalize = False)
			ww_plus, ss_plus = mft.get_spec(t_plus, l, [0.55, 0.95], normalize = False)																													

			wwe_minus = np.linspace(ww_minus[0], ww_minus[-1], len(wle))
			wwe_init = np.linspace(ww_init[0], ww_init[-1], len(wle))
			wwe_plus = np.linspace(ww_plus[0], ww_plus[-1], len(wle))

			intep_minus = interp1d(ww_minus, ss_minus)
			intep_init = interp1d(ww_init, ss_init)
			intep_plus = interp1d(ww_plus, ss_plus)

			sse_minus = intep_minus(wwe_minus)
			sse_init = intep_init(wwe_init)
			sse_plus = intep_plus(wwe_plus)

			wwe_minus, sse_minus = mft.broaden(wwe_minus, sse_minus, 3000, 0, 0)
			# wwe_minus, sse_minus = redres(wwe_minus, sse_minus, 8)

			wwe_init, sse_init = mft.broaden(wwe_init, sse_init, 3000, 0, 0)
			# wwe_init, sse_init = redres(wwe_init, sse_init, 8)

			wwe_plus, sse_plus = mft.broaden(wwe_plus, sse_plus, 3000, 0, 0)
			# wwe_plus, sse_plus = redres(wwe_plus, sse_plus, 8)


			second_wl = np.linspace(max(wle[0], wwe_init[0], wwe_minus[0], wwe_plus[0]), min(wle[-1], wwe_init[-1], wwe_minus[-1], wwe_plus[-1]), max(len(wle), len(wwe_init), len(wwe_minus), len(wwe_plus)))

			di2 = interp1d(wle, spece)
			mi2_minus = interp1d(wwe_minus, sse_minus)
			mi2_init = interp1d(wwe_init, sse_init)
			mi2_plus = interp1d(wwe_plus, sse_plus)

			spece = di2(second_wl)
			sse_minus = mi2_minus(second_wl)
			sse_init = mi2_init(second_wl)
			sse_plus = mi2_plus(second_wl)

			n_minus = find_norm(sse_minus, spece)
			n_init = find_norm(sse_init, spece)
			n_plus = find_norm(sse_plus, spece)

			sse_minus = [s/n_minus for s in sse_minus]
			sse_init = [s/n_init for s in sse_init]
			sse_plus = [s/n_plus for s in sse_plus]

			var = np.mean([sp * 0.01 for sp in spece])
			
			cs_minus = mft.chisq(sse_minus, spece, var)
			cs_init = mft.chisq(sse_init, spece, var)
			cs_plus = mft.chisq(sse_plus, spece, var)

			niter += 1

			# print(step, cs_minus, cs_init, cs_plus, t_init)
			st.append(step)
			xs.append(cs_init)
			temp.append(t_init)
			logg.append(l)
			norm.append(n_init)

			if cs_init < cs_minus and cs_init < cs_plus:
				step *= 0.5
				niter = 0
			elif cs_init > cs_minus and cs_init < cs_plus:
				t_init = t_init - step
			elif cs_init < cs_minus and cs_init > cs_plus:
				t_init = t_init + step
			else:
				if cs_minus < cs_plus:
					t_init = t_init - step
				else:
					t_init = t_init + step

			if niter > 10:
				step *= 0.5
	'''
	fig, [ax, ax1] = plt.subplots(ncols = 2)
	for n in range(len(st)):
		w, sp = mft.get_spec(temp[n], logg[n], [0.5, 1], normalize = False)
		print(w[0])
		#we = np.linspace(w[0], w[-1], 12000)
		intep = interp1d(w, sp)
		spe = intep(wle)
		print('broadening')
		we, spe = mft.broaden(wle, spe, 3000, 0, 0)
		#we, spe = redres(we, spe, 6)

		idx = np.where(spece == max(spece))
		norm_val = spe[mft.find_nearest(we, wle[idx])]
		specen = [spece[k]/max(spece) for k in range(len(spece))]
		spe = [(spe[k]/norm_val)*norm[n] for k in range(len(spe))]

		specen = [s - n for s in specen]
		spe = [s - n for s in spe]

		print('plotting')
		ax.plot(we, spe)
		ax.text(6000, (-1 *n)+0.2, 'Temp = {}, log(g) = {}, step size = {}'.format(temp[n], logg[n], st[n]))
		ax.plot(wle, specen, color = 'k')
		ax.set_xlim(6000, 9000)
		ax.set_title('Data: Temp = {}, log(g) = {}'.format(t_guess, lg_guess))

		resid = [specen[k] - spe[k] - n for k in range(len(specen))]
		ax1.plot(wle, resid)
		ax1.set_title('Residuals')
		ax1.set_xlim(6000, 9000)

	#fig.tight_layout()
	plt.savefig(filename.split('/')[0] + '/meeb_steps_{}.pdf'.format(filename.split('_')[-1].split('.')[0]))
	plt.close()
	'''
	print('saving')
	np.savetxt(filename.split('/')[0] + '/results/params_' + filename.split('/')[-1], np.column_stack((st, xs, temp, logg, norm)), header = '#step size, chi square, temperature, log(g), normalization')
	#np.savetxt('results/params_' + filename.split('/')[-1], np.column_stack((xs, temp, logg, norm)), header = '# chi square, temperature, log(g), normalization')

	return

def interp_2d(temp_point, lum_point, t_mat, lum_mat, zval_mat):
	lum1, lum2 = 0, 0
	try:
		diffs = []
		for l in lum_mat:
			diffs.append(l - lum_point)
		diffs = np.array(diffs)
		lp1 = max(diffs[np.where(diffs < 0)])
		lp2 = min(diffs[np.where(diffs > 0)])
		idx1 = np.where(diffs == lp1)[0]
		idx2 = np.where(diffs == lp2)[0]

		if len(idx1) > 1:
			idx1 = idx1[0]
		if len(idx2) > 1:
			idx2 = idx2[0]

		lum1, lum2 = min(lum_mat[int(idx1)], lum_mat[int(idx2)]), max(lum_mat[int(idx1)], lum_mat[int(idx2)])

	except:
		lp1 = mft.find_nearest(lum_mat, lum_point)
		if lum_mat[lp1] > lum_point:
			lp2 = lp1 - 1
		else:
			lp2 = lp1 + 1
		if lp1 < len(lum_mat) and lp2 < len(lum_mat):
			lum1, lum2 = lum_mat[min(lp1, lp2)], lum_mat[max(lp1, lp2)]
		elif max(lum1, lum2) > max(lum_mat):
			lum1 = lum_mat[lp1]
			lum2 = max(lum_mat) + 0.1
		else:
			lum1 = lum_mat[lp1]
			lum2 = min(lum_mat) - 0.1

	t1 = mft.find_nearest(t_mat, temp_point)
	if t_mat[t1] > temp_point:
		t2 = t1 - 1
	else:
		t2 = t1 + 1
	t1, t2 = min(t_mat[t1], t_mat[t2]), max(t_mat[t1], t_mat[t2])

		# print('z11: ', t1, lum1, z11)
		# print('z22: ', t2, lum2, z22)
		# print('z12: ', t1, lum2, z12)
		# print('z21: ', t2, lum1, z21)
		# print('temp, lum: ', temp_point, lum_point)
	# else:
	# 	tmat_diff, lmat_diff = np.ndarray(np.shape(t_mat)), np.ndarray(np.shape(t_mat))
	# 	for j in range(np.shape(t_mat)[0]):
	# 		for k in range(np.shape(t_mat)[1]):
	# 			tmat_diff[j][k] = t_mat[j][k] - temp_point
	# 			lmat_diff[j][k] = lum_mat[j][k] - lum_point


	# 	diffs_temp_pos = tmat_diff[np.where(tmat_diff >= 0)]
	# 	diffs_temp_neg = tmat_diff[np.where(tmat_diff < 0)]
	# 	diffs_lum_pos = lmat_diff[np.where(lmat_diff >= 0)]
	# 	diffs_lum_neg = lmat_diff[np.where(lmat_diff < 0)]

	# 	t1, t2, lum1, lum2 = t_mat[np.where(tmat_diff == min(diffs_temp_pos))], t_mat[np.where(tmat_diff == max(diffs_temp_neg))], lum_mat[np.where(lmat_diff == min(diffs_lum_pos))], lum_mat[np.where(lmat_diff == max(diffs_lum_neg))]

	# 	if len(t1) > 1:
	# 		t1 = t1[0]
	# 	if len(t2) > 1:
	# 		t2 = t2[0]
	# 	if len(lum1) > 1:
	# 		lum1 = lum1[0]
	# 	if len(lum2) > 1:
	# 		lum2 = lum2[0]

	# 	t1, t2, lum1, lum2 = float(t1), float(t2), float(lum1), float(lum2)

	# 	t_mat, lum_mat, zval_mat = t_mat.flatten(), lum_mat.flatten(), zval_mat.flatten()

	z11 = zval_mat[fn_pair(t_mat, lum_mat, t1, lum1)]
	z22 = zval_mat[fn_pair(t_mat, lum_mat, t2, lum2)]
	z12 = zval_mat[fn_pair(t_mat, lum_mat, t1, lum2)]
	z21 = zval_mat[fn_pair(t_mat, lum_mat, t2, lum1)]

	fxy1 = (((t2 - temp_point)/(t2 - t1))*z11) + (((temp_point - t1)/(t2 - t1))*z21)
	fxy2 = (((t2 - temp_point)/(t2-t1))*z12) + (((temp_point - t1)/(t2 - t1))*z22)
	fxy = (((lum2 - lum_point)/(lum2 - lum1))*fxy1) + (((lum_point - lum1)/(lum2 - lum1)) * fxy2)
	print('lums and temps: ', lum1, lum_point, lum2, t1, temp_point, t2)
	print('fxy pars', lum2 - lum_point, lum2 - lum1, fxy1, lum_point - lum1, fxy2, fxy)
	return fxy
	

def analyze_sys(runfolder):
	'''
	Args: 
		runfolder (string): path to follow
		age_guess (float): System age guess in Myr.

	Returns:
		Mass and Luminosity from teff and log(g).

	'''

	csqs = glob(runfolder + '/results/params*')

	pnum, pmass, multiplicity, page, av, ptemp, plogg, pluminosity, distance = np.genfromtxt(runfolder + '/ppar.txt', unpack = True)

	snum, s_mass, sep, sage, seccentricity, period, stemp, slogg, sluminosity = np.genfromtxt(runfolder + '/spar.txt', unpack = True)

	sl = np.zeros(len(pluminosity))
	for n in range(len(pluminosity)):
		if n in snum:
			sl[n] = pluminosity[n] + sluminosity[np.where(snum == pnum[n])]
		else:
			sl[n] = pluminosity[n]

	masses = np.zeros(len(csqs))
	lums = np.zeros(len(csqs))
	ages = np.zeros(len(csqs))
	num = np.zeros(len(csqs))
	inp_t = np.zeros(len(csqs))
	out_t = np.zeros(len(csqs))
	inp_age = np.zeros(len(csqs))

	isos = glob('isochrones/baraffe*.txt')
	nums = []
	for i in isos:
		nn = i.split('_')[1]
		nums.append(nn)

	nums = sorted(nums)

	mass, temps, lum, logg, rad, lilo, mj, mh, mk = np.genfromtxt(glob('isochrones/baraffe_{}*.txt'.format(nums[0]))[0], unpack = True, autostrip = True, comments = '!')
	lum = [10 ** l for l in lum]
	age = np.full(len(mass), int(glob('isochrones/baraffe_{}*.txt'.format(nums[0]))[0].split('_')[1])/10)

	for n in range(1, len(nums)):
		m, tt, lumi, llg, r, llo, mjj, mhh, mkk = np.genfromtxt(glob('isochrones/baraffe_{}*.txt'.format(nums[n]))[0], unpack=True, autostrip = True, comments = '!')
		a = np.full(len(m), int(glob('isochrones/baraffe_{}*.txt'.format(nums[n]))[0].split('_')[1])/10)
		lumi = [10**l for l in lumi]
		age = np.hstack((age, a))
		mass = np.hstack((mass, m))
		temps = np.hstack((temps, tt))
		lum = np.hstack((lum, lumi))
		logg = np.hstack((logg, llg))

	for k, file in enumerate(csqs):
		number = int(file.split('.')[0].split('_')[-1])
		num[k] = number
		st, cs, temp, lg, norm = np.genfromtxt(file, unpack = True, autostrip = True)
		ts = temp[np.where(cs == min(cs))][0]
		l = lg[np.where(cs == min(cs))]
		print('ts, l, log g (fitted pars)', ts, sl[number], l)
		out_t[k] = ts
		inp_t[k] = ptemp[number]
		inp_age[k] = page[number]
		luminosity = sl[number]
		if max(temps) > ts and max(lum) > luminosity and min(temps) < ts and min(lum) < luminosity:	
			print('baraffe')
			a = griddata((temps, lum), age, (ts, luminosity))#interp_2d(ts, luminosity, temps, lum, age)
			m = griddata((temps, lum), mass, (ts, luminosity)) #interp_2d(ts, luminosity, temps, lum, mass)

			masses[k] = m
			lums[k] = luminosity
			ages[k] = a

			print('age: ', a, 'temp: ', ts, 'lum: ', luminosity, 'mass: ', m)

		else: #max(temps) < ts or max(lum) < luminosity or min(lum) > luminosity or min(temps) > ts or m < 0 or a < 0:
			print('parsec')
			matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
			aage = matrix[:, 1]
			aage = [(10**a)/1e6 for a in aage]
			#mat columns are:
			#mass (0), logL (1), logTe (2), logg (3)
			mat = matrix[:,4:8]
			ma = mat[:,0]
			lll = mat[:, 1]#[10**(ll) for ll in mat[:, 1]]
			teff = [10 ** (tt) for tt in mat[:, 2]]
			llog = mat[:, 3]

			a = griddata((teff, lll), aage, (ts, luminosity))#interp_2d(ts, luminosity, teff, lll, aage)

			m = griddata((teff, lll), ma, (ts, luminosity))#interp_2d(ts, luminosity, teff, lll, ma)

			masses[k] = m
			lums[k] = sl[number]
			ages[k] = a
			luminosity = 10 ** luminosity

			print('age: ', a, 'temp: ', ts, 'lum: ', luminosity, 'mass: ', m)

		if luminosity < 0:
			print('age fit parameters: ', ts, luminosity, m, a)

	print('max age: ', max(ages))

	t_pct = [abs((out_t[n] - inp_t[n])/inp_t[n]) * 100 for n in range(len(inp_t))]
	l_pct = [abs((lums[n] - sl[n])/sl[n]) * 100 for n in range(len(sl))]
	total_pct = t_pct + l_pct

	colors = cm.winter(total_pct)

	fig, ax = plt.subplots()
	for n in range(len(ages)):
		ax.scatter(page[n], ages[n], marker = '.', s = total_pct[n]*1000, edgecolors = 'k', color = colors[n], cmap = 'plasma')
	if len(snum > 0):
		s_fit_ages = []
		for n, idx in enumerate(snum):
			s_fit_ages.append(ages[int(idx)])
		ax.scatter(sage, s_fit_ages, marker = 'v', color='xkcd:sky blue', label = 'Secondary stars')

	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input age (Myr)')
	ax.set_ylabel('Output age (Myr)')
	# ax.set_aspect('equal', 'box')
	ax.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_plot.pdf')
	plt.close()

	fig, ax = plt.subplots()
	ax.scatter(inp_t, out_t-inp_t, marker = '.', s = 20, color = 'navy', label = 'Primary stars')
	ax.plot([min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t))], [0, 0], linestyle = ':')#[min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t))], linestyle=':', label = '1:1')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input Temp (K)')
	ax.set_ylabel('Output Temp difference (K)')
	ax.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/temp_plot.pdf')
	plt.close()

	fig, ax = plt.subplots()
	ax.scatter(inp_age, ages, marker = '.', s = 20, color = 'navy', label = 'Primary stars')
	ax.plot([min(min(inp_age), min(ages)), max(max(inp_age), max(ages))], [min(min(inp_age), min(ages)), max(max(inp_age), max(ages))], linestyle=':', label = '1:1')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input age (Myr)')
	ax.set_ylabel('Output age (Myr)')
	ax.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_plot2.pdf')
	plt.close()
	np.savetxt(runfolder + '/results/mass_fit_results.txt', np.column_stack((num, ages, masses, lums, out_t)), header= '#number, age (myr), Mass, Luminosity, fitted temperature')
	return masses, lums

def plot_specs(num, run):
	cwd = os.getcwd()
	wl, spec = np.genfromtxt(run + '/specs/spec_{}.txt'.format(num), unpack = True)

	ewl = np.linspace(wl[0], wl[-1], len(wl))
	inte = interp1d(wl, spec)
	espec = inte(ewl)

	ewl, espec = mft.broaden(ewl, espec, 3000, 0, 0)
	wl, spec = redres(ewl, espec, 6)

	p_num, p_mass, mul, p_age, p_av, p_temp, p_logg, p_luminosity, distance = np.genfromtxt(run + '/ppar.txt', unpack = True)
	pnum, multiplicity, p_temp, p_logg = int(p_num[np.where(p_num == num)[0]]), mul[np.where(p_num == num)[0]], p_temp[np.where(p_num == num)[0]], p_logg[np.where(p_num == num)[0]]

	st, xs, temp, lg, norm = np.genfromtxt(cwd + '/' + run+'/results/params_spec_{}.txt'.format(num), unpack = True)

	idx = np.where(xs == min(xs))

	t, l, n = temp[idx], lg[idx], norm[idx]

	w, s = mft.get_spec(t[0], l[0], [0.55, 0.95], normalize = False)
	w_ = np.linspace(w[0], w[-1], len(w))

	model_intep = interp1d(w, s)
	s_ = model_intep(w_)

	w_, s_ = mft.broaden(w_, s_, 3000, 0, 0)
	w_, s_ = redres(w_, s_, 6)

	s_ *= n

	fig1, ax1 = plt.subplots()

	if int(multiplicity) == 0:
		ax1.plot(wl, spec, color = 'navy', label = 'Input: T = {}, log(g) = {}'.format(p_temp, p_logg))
	else:
		s_num, s_mass, s_sep, s_age, eccentricity, period, s_temp, s_logg, ls_uminosity = np.genfromtxt(run + '/spar.txt', unpack = True)
		if np.size(s_num) > 1:
			sn = np.where(s_num == pnum)[0]
			s_temp, s_logg = s_temp[sn], s_logg[sn]

		ax1.plot(wl, spec, color = 'navy', label = 'Input: T1 = {}, T2 = {}, \nlog(g)1 = {}, log(g)2 = {}'.format(p_temp, s_temp, p_logg, s_logg))

	ax1.plot(w_, s_, color='xkcd:sky blue', label = 'Best fit model: \nT = {}, log(g) = {}, \nChi sq = {}'.format(t, l, str(xs[idx]).split('.')[0].split('[')[-1]), linestyle= ':')
	plt.minorticks_on()
	ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax1.tick_params(bottom=True, top =True, left=True, right=True)
	ax1.tick_params(which='both', labelsize = "large", direction='in')
	ax1.tick_params('both', length=8, width=1.5, which='major')
	ax1.tick_params('both', length=4, width=1, which='minor')
	ax1.set_xlabel(r'$\lambda$ (\AA)', fontsize = 14)
	ax1.set_ylabel(r'$L_{\lambda}$', fontsize = 14)
	ax1.set_xlim(6000, 9000)
	ax1.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(run + '/plot_spec_{}.pdf'.format(num))
	plt.close(fig1)

	fig2, ax2 = plt.subplots()
	if multiplicity == 0:
		ax2.plot(wl, spec, color = 'navy', label = 'Input: T = {}, log(g) = {}'.format(p_temp, p_logg))
	else:
		ax2.plot(wl, spec, color = 'navy', label = 'Input: T1 = {}, T2 = {}, \nlog(g)1 = {}, log(g)2 = {}'.format(p_temp, s_temp, p_logg, s_logg))
	ax2.plot(w_, s_, color='xkcd:sky blue', label = 'Best fit model: \nT = {}, log(g) = {}, \nChi sq = {}'.format(t, l, str(xs[idx]).split('.')[0].split('[')[-1]), linestyle= ':')
	plt.minorticks_on()
	ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax2.tick_params(bottom=True, top =True, left=True, right=True)
	ax2.tick_params(which='both', labelsize = "large", direction='in')
	ax2.tick_params('both', length=8, width=1.5, which='major')
	ax2.tick_params('both', length=4, width=1, which='minor')
	ax2.set_xlabel(r'$\lambda$ (\AA)', fontsize = 14)
	ax2.set_ylabel(r'$L_{\lambda}$', fontsize = 14)
	ax2.set_xlim(6850, 7250)
	ax2.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(run + '/plot_spec_TiO_{}.pdf'.format(num))
	plt.close(fig2)

	fig3, ax3 = plt.subplots()
	if multiplicity == 0:
		ax3.plot(wl, spec, color = 'navy', label = 'Input: T = {}, log(g) = {}'.format(p_temp, p_logg))
	else:
		ax3.plot(wl, spec, color = 'navy', label = 'Input: T1 = {}, T2 = {}, \nlog(g)1 = {}, log(g)2 = {}'.format(p_temp, s_temp, p_logg, s_logg))
	ax3.plot(w_, s_, color='xkcd:sky blue', label = 'Best fit model: \nT = {}, log(g) = {}, \nChi sq = {}'.format(t, l, str(xs[idx]).split('.')[0].split('[')[-1]), linestyle= ':')
	plt.minorticks_on()
	ax3.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax3.tick_params(bottom=True, top =True, left=True, right=True)
	ax3.tick_params(which='both', labelsize = "large", direction='in')
	ax3.tick_params('both', length=8, width=1.5, which='major')
	ax3.tick_params('both', length=4, width=1, which='minor')
	ax3.set_xlabel(r'$\lambda$ (\AA)', fontsize = 14)
	ax3.set_ylabel(r'$L_{\lambda}$', fontsize = 14)
	ax3.set_xlim(8400, 8950)
	ax3.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(run + '/plot_spec_CaIR_{}.pdf'.format(num))
	plt.close(fig3)

	return 

def run_pop(nsys, run, new_pop = False):

	cwd = os.getcwd()

	if new_pop == True:

		bf = [0, 0, 0, 0] #[0.35, 0.4, 0.45, 0.5]
		md = [0.8, 1, 1.5]

		if not os.path.exists(cwd + '/' + run):
			os.mkdir(cwd + '/' + run)
		if not os.path.exists(cwd + '/' + run + '/specs'):
			os.mkdir(cwd + '/' + run + '/specs')
		if not os.path.exists(cwd + '/' + run + '/results'):
			os.mkdir(cwd + '/' + run + '/results')

		print('Making systems')
		age_range = [0.99, 1.01]
		pool = mp.Pool()
		results = [pool.apply_async(make_binary_sys, args = (ns, 1, bf, md, age_range, 0, run)) for ns in range(nsys)]
		out = [p.get() for p in results]

		out1, out2 = np.array(out).T

		ppar = [out1[n][:][1] for n in range(nsys)]
		np.savetxt(cwd + '/' + run + '/ppar.txt', ppar)

		nbin = 0
		wh = []
		for k, n in enumerate(ppar):
			nbin += int(n[2])
			if int(n[2]) == 1:
				wh.append(k)
		if nbin > 0:
			spar = [out2[w][:][1] for w in wh]
			np.savetxt(cwd + '/' + run + '/spar.txt', spar)
		else:
			pass;

	print('Fitting now')
	t_g, lg_g = np.genfromtxt(cwd +'/' + run + '/ppar.txt', usecols = (5,6), unpack = True)

	ts = [np.round(t/25) * 25 for t in t_g]
	ls = [np.round(l / 0.1) * 0.1 for l in lg_g]
	trange = [np.arange(((np.round(t / 100) * 100) - 300), ((np.round(t / 100) * 100) + 700), 100) for t in ts]
	lgrange = [3.5]#np.arange(3.5, 4.1, 0.1)
	
	files = np.array([run + '/specs/spec_{}.txt'.format(n) for n in range(nsys)])

	# for n in range(nsys):
	# 	even_simpler(files[n], ts[n], ls[n], trange[n], lgrange)

	pool = mp.Pool()
	results = [pool.apply_async(even_simpler, args = (files[n], ts[n], ls[n], trange[n], lgrange)) for n in range(nsys)]
	o = [p.get() for p in results]
	print('done!')

	return

def plot_init_pars(run, pri_num, sec_num, pri_mass, sec_mass, sep, av, distance):
	mr = []
	for n in pri_num:
		if n in sec_num:
			mr.append(float(sec_mass[np.where(sec_num == n)]/pri_mass[np.where(pri_num == n)][0]))

	fig, [ax1, ax2] = plt.subplots(nrows = 2)
	ax1.hist(sep, color = 'navy')
	ax1.set_xlabel('Separation (AU)')
	ax1.set_ylabel('Number')

	ax2.hist(mr, bins = 20, color='xkcd:sky blue')
	ax2.set_xlabel(r'Mass ratio (secondary/primary), M$_{\odot}$')
	ax2.set_ylabel('Number')

	plt.minorticks_on()
	ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax1.tick_params(bottom=True, top =True, left=True, right=True)
	ax1.tick_params(which='both', labelsize = "large", direction='in')
	ax1.tick_params('both', length=8, width=1.5, which='major')
	ax1.tick_params('both', length=4, width=1, which='minor')

	ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax2.tick_params(bottom=True, top =True, left=True, right=True)
	ax2.tick_params(which='both', labelsize = "large", direction='in')
	ax2.tick_params('both', length=8, width=1.5, which='major')
	ax2.tick_params('both', length=4, width=1, which='minor')

	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/mr_and_sep.pdf')
	plt.close(fig)

	fig = plt.figure(figsize=(6, 6))
	grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
	main_ax = fig.add_subplot(grid[:-1, 1:], xticklabels=[], yticklabels=[])
	y_hist = fig.add_subplot(grid[:-1, 0], xticklabels = [])#, sharey=main_ax)
	x_hist = fig.add_subplot(grid[-1, 1:], yticklabels = [])#, sharex=main_ax)

	main_ax.plot(distance, av, 'ok', alpha = 0.5)
	#main_ax.set_xlabel('Distance (pc)')
	#main_ax.set_ylabel(r'A$_{V}$, (mag)')

	x_hist.hist(distance, orientation = 'vertical')
	x_hist.set_xlabel('Distance (pc)')
	x_hist.invert_yaxis()

	y_hist.hist(av, orientation = 'horizontal')
	y_hist.set_ylabel('Extinction (AV, mag)')
	y_hist.invert_xaxis()

	plt.minorticks_on()
	main_ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	main_ax.tick_params(bottom=True, top =True, left=True, right=True)
	main_ax.tick_params(which='both', labelsize = "large", direction='in')
	main_ax.tick_params('both', length=8, width=1.5, which='major')
	main_ax.tick_params('both', length=4, width=1, which='minor')

	x_hist.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	x_hist.tick_params(bottom=True, top =True, left=True, right=True)
	x_hist.tick_params(which='both', labelsize = "large", direction='in')
	x_hist.tick_params('both', length=8, width=1.5, which='major')
	x_hist.tick_params('both', length=4, width=1, which='minor')

	y_hist.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	y_hist.tick_params(bottom=True, top =True, left=True, right=True)
	y_hist.tick_params(which='both', labelsize = "large", direction='in')
	y_hist.tick_params('both', length=8, width=1.5, which='major')
	y_hist.tick_params('both', length=4, width=1, which='minor')

	# plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/dist_av.pdf')
	plt.close(fig)
	return


def final_analysis(nsys, run):
	print('Plotting!')

	#[plot_specs(n, run) for n in range(nsys)]

	print('Running mass analysis')

	analyze_sys(run)

	num, age, fit_mass, logg, fit_t = np.genfromtxt(run + '/results/mass_fit_results.txt', unpack = True)

	n, mass, multiplicity, page, av, ptemp, plogg, pluminosity, distance = np.genfromtxt(run + '/ppar.txt', unpack = True)

	sn, smass, sep, sage, seccentricity, period, stemp, slogg, sluminosity = np.genfromtxt(run + '/spar.txt', unpack = True)

	fig, ax = plt.subplots()
	ax.plot(ptemp, pluminosity, 'v', color = 'navy', label = 'Primary stars')
	ax.plot(stemp, sluminosity, 'o', color='xkcd:sky blue', label = 'Secondary stars')
	ax.set_xlabel('Temperature (K)')
	ax.set_ylabel('Luminosity (solar lum.)')
	ax.legend(loc = 'best')
	ax.invert_xaxis()
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/hrd_input.pdf')
	plt.close()

	plot_init_pars(run, n, sn, mass, smass, sep, av, distance)

	sys_lum = []

	mass_resid = [] 
	fm = []
	mm = []
	for k in n: 
		idx = np.where(num == k) 
		m = mass[idx]
		l = pluminosity[idx]
		if int(k) in sn:
			m += smass[np.where(sn == int(k))]
			l += sluminosity[np.where(sn == int(k))]
		test = m - fit_mass[idx] 
		fm.append(fit_mass[idx])
		mm.append(m[0])
		sys_lum.append(l)
		mass_resid.append(test[0]) 

	sys_flux = [((10**sys_lum[n]) * 3.9e33) /(4 * np.pi * (distance[n] * 3.086e18)**2) for n in range(len(num))]
	sys_mag_app = [-2.5 * np.log10(sf[0]/17180) for sf in sys_flux]

	mr = [mass_resid[n]/mm[n] for n in range(len(mass))]

	plt.figure(1)
	plt.hist(mm, label = 'input', color = 'navy')
	plt.hist(fit_mass, color = 'xkcd:sky blue', label = 'output') 
	plt.legend()
	plt.savefig(run + '/masshist.pdf')
	plt.close()

	plt.figure(2)
	plt.hist(mass_resid, color='navy')	
	plt.xlabel('Mass residual, solar masses')
	plt.tight_layout()
	plt.savefig(run + '/error_fit.pdf')
	plt.close()

	plt.figure(3)
	plt.hist(mr, color = 'navy', label = 'Avg. frac. error = {:.2f}\n 1 stdev = {:.2f}'.format(float(np.mean(mr)), float(np.std(mr))))
	plt.legend() 
	plt.xlabel('Fractional mass residual')
	plt.tight_layout()
	plt.savefig(run + '/error_fit_fractional.pdf')
	plt.close()

	one_one = np.arange(0, 4)
	plt.figure(4)
	plt.scatter(mm, fm)
	plt.plot(one_one, one_one, ':', label = 'One-to-one correspondence line')
	#plt.xlim(min(min(mass), min(fit_mass)) - 0.1, max(max(mass), max(fit_mass)) + 0.1)
	#plt.ylim(min(min(mass), min(fit_mass)) - 0.1, max(max(mass), max(fit_mass)) + 0.1)
	#plt.ylim(0, 3)
	plt.xlabel('Expected mass')
	plt.ylabel('Fitted mass')
	plt.legend(loc = 'best')
	plt.tight_layout()
	plt.savefig(run + '/mass_scatter.pdf')
	plt.close()

	plt.figure(5)
	plt.scatter(fit_t, sys_lum)
	plt.xlabel('Fitted Temperature')
	# plt.xscale('log')
	# plt.yscale('log')
	plt.legend(loc = 'best')
	plt.gca().invert_xaxis()
	plt.ylabel(r'System Luminosities, L$_{\odot}$')
	plt.tight_layout()
	plt.savefig(run + '/HRD_output.pdf')
	plt.close()

	plt.figure(6)
	plt.scatter(fit_t, sys_mag_app)
	plt.xlabel('Fitted Temperature')
	plt.ylabel(r'System apparent bolometric magnitude')
	plt.gca().invert_yaxis()
	plt.tight_layout()
	plt.savefig(run + '/HRD_output_flux.pdf')
	plt.close()

	return

def run_on_grid_points(n_sys):
	lg = 4
	files = glob('phoenix/lte*.7.dat.txt')
	t = sorted([int(files[n].split('-')[0].split('e')[2]) * 1e2 for n in range(len(files))])
	temps = [min(t)]

	for n, tt in enumerate(t):
		if tt > temps[-1]:
			temps.append(tt)

	mass = make_mass(n_sys)
	n = 0
	while n < n_sys:
		ptemp, plogg, plum = get_params(mass[n], 1)
		t1_idx = mft.find_nearest(temps, ptemp)
		tt = temps[t1_idx]

		pri_wl, pri_spec = mft.get_spec(tt, lg, [0.55, 0.95], normalize = False)
		np.savetxt('spec_{}_{}.txt'.format(n, tt), np.column_stack((pri_wl, pri_spec)))

		print('fitting')
		even_simpler(os.getcwd() + '/spec_{}_{}.txt'.format(n, tt), tt, lg, np.arange(tt - 200, tt + 300, 100), [3.5, 4])

		n += 1

	return


run_pop(2, 'run20', new_pop = True) 
final_analysis(2, 'run20')
#run_on_grid_points(4)
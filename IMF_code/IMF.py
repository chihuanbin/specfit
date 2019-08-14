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
from scipy.interpolate import interp1d
from scipy.integrate import trapz, simps
import model_fit_tools_v2 as mft
import multiprocessing as mp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

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
	if mass <= 1.4 and age>=0.5:
		m, lum, radius = np.genfromtxt(glob('isochrones/*{}*.txt'.format(a))[0], usecols =(0, 2, 4), comments = '!', unpack = True, autostrip = True)
		idx = mft.find_nearest(m, mass)
		a_m = m[idx] #should I interpolate? #solar masses
		a_l = lum[idx] #solar luminosity
		a_r = radius[idx] #solar radius

		#units: solar masses, kelvin, solar luminosity, log(g), giga-centimeters (NOT SOLAR RADII)
		#THUS assume solar radius = 6.957e5 km = 6.957e10 cm = 69.75 Gcm
		m_real, teff, lu, logg, rad = np.genfromtxt(glob('phoenix_isos/*{}*.txt'.format(str(int(age)).zfill(5)))[0], \
			usecols = (0, 1, 2, 3, 4), autostrip = True, unpack = True)

		rad = [np.around(r/69.75, 2) for r in rad] #convert to solar radius

		new_idx = mft.find_nearest(lu[1:-1], a_l)
		temp, log_g = teff[new_idx], logg[new_idx]
		return temp, log_g, lu[new_idx]

	else:
		age = np.log10(age * 1e6)
		matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
		ages = matrix[:, 1]
		age_new = ages[mft.find_nearest(ages, age)]
		matrix = matrix[np.where(matrix[:, 1] == age_new)]

		mat = matrix[:,4:8]
		#mat columns are:
		#mass (0), logL (1), logTe (2), logg (3)
		idx = mft.find_nearest(mass, mat[:, 0])
		a_m = mat[idx, 0]
		a_l = 10**mat[idx, 1]
		a_t = 10**mat[idx, 2]
		logg = mat[idx, 3]
		return a_t, logg, a_l

def fn_pair(mat1, mat2, val1, val2):
	ret1 = [np.abs(float(m) - val1)[0] for m in mat1]
	ret2 = [np.abs(float(m) - val2)[0] for m in mat2]

	min_sum = [ret1[n] + ret2[n] for n in range(len(ret1))]

	idx = np.where(min_sum == min(min_sum))

	return idx

def match_pars(temp, lum):
	isos = glob('phoenix_isos/*.txt')

	a = np.genfromtxt(isos[0], autostrip=True, unpack=True, usecols = (0, 1, 2, 3)) #pulls mass, teff, lum, logg

	for n, i in enumerate(isos):
		if n > 0:
			b = np.genfromtxt(i, autostrip=True, unpack=True, usecols = (0,1,2,3))
			a = np.vstack((a, b))

	matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
	mat = matrix[:,4:8]
	#mat columns are:
	#mass (0), logL (1), logTe (2), logg (3)
	a_m = mat[:, 0]
	a_l = 10**mat[:, 1]
	a_t = 10**mat[:, 2]
	logg = mat[:, 3]

	c = np.array((a_m, a_t, a_l, logg))

	a = np.vstack((a, c))

	idx = fn_pair(a[:, 1], a[:, 2], temp, lum)

	mass, temp, lum, logg = a[idx]

	wl, spec = mft.get_spec(temp, logg, [0.55, 0.95], normalize = False)

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

	r_num = r.random_sample(size = n_sys)
	mass = []
	for rn in r_num:
		if rn < calc_pct(imf = 'pct', wh='chabrier'):
			m = 10
			while m > 1:
				m = r.lognormal(np.log(0.3), 0.69)
		else:
			m = 10**((1/r.power(2.3)) * 4.43e-2)
		mass.append(m)
	return mass

def make_binary_sys(n, n_sys, multiplicity, mass_bins, age, av, run, sfr = 'taurus'):
	#from Raghavan + 2010, eccentricity can be pulled from a uniform distribution
	#From the same source, the period distribution can be approximated as a Gaussian with
	#a peak at log(P) = 5.03 and sigma_log(P) = 2.28

	#decide if my mass will end up in the chabrier or salpeter regime by drawing a random number and comparing to the percentage
	#of the total cmf that is taken by the Chab. portion of the imf
	r = np.random.RandomState()#seed = 234632)

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

	pri_par = [n, mass[n_sys -1], 0, age, av, ptemp, plogg, plum, get_distance(sfr)] #etc. - can add more later

	pri_pars = np.vstack((pri_pars, np.array(pri_par)))
	if type(pri_par[1]) != None:
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
			sec_par[2] = r.uniform(0, 1)
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

		# print(step, xs_minus, xs_init, xs_plus, init_guess)

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

	wle, spece = mft.broaden(wl, spece, 3000, 0, 0)
	wle, spece = redres(wle, spece, 8)
	xs = []
	temp = []
	logg = []
	norm = []

	for l in lg_range:
		for t in t_range:

			ww, ss = mft.get_spec(t, l, [0.55, 0.95], normalize = False)

			wwe = np.linspace(ww[0], ww[-1], len(wle))

			intep_model = interp1d(ww, ss)
			sse = intep_model(wwe)

			wwe, sse = mft.broaden(wwe, sse, 3000, 0, 0)
			wwe, sse = redres(wwe, sse, 8)

			second_wl = np.linspace(max(wle[0], wwe[0]), min(wle[-1], wwe[-1]), max(len(wle), len(wwe)))

			di2 = interp1d(wle, spece)
			mi2 = interp1d(wwe, sse)

			spece = di2(second_wl)
			sse = mi2(second_wl)

			n = find_norm(sse, spece)
			sse = [s * n for s in sse]
			var = np.mean([sp * 0.01 for sp in spece])
			cs = mft.chisq(sse, spece, var)

			xs.append(cs)
			temp.append(t)
			logg.append(l)
			norm.append(n)

	np.savetxt(filename.split('/')[0] + '/results/params_' + filename.split('/')[-1], np.column_stack((xs, temp, logg, norm)), header = '# chi square, temperature, log(g), normalization')

	return

def analyze_sys(runfolder, age_guess):
	'''
	Args: 
		runfolder (string): path to follow
		age_guess (float): System age guess in Myr.

	Returns:
		Mass and Luminosity from teff and log(g).

	'''

	csqs = glob(runfolder + '/results/params*')

	#pnum, pmass, multiplicity, page, av, ptemp, plogg, pluminosity, distance = np.genfromtxt(runfolder + '/ppar.txt', unpack = True)

	#snum, s_mass, sep, sage, seccentricity, period, stemp, slogg, sluminosity = np.genfromtxt(runfolder + '/spar.txt', unpack = True)

	masses = []
	lums = []
	num = []

	for file in csqs:
		number = int(file.split('.')[0].split('_')[-1])
		cs, temp, lg, norm = np.genfromtxt(file, unpack = True, autostrip = True)
		t = temp[np.where(cs == min(cs))]
		l = lg[np.where(cs == min(cs))]

		mass, temp, lum, logg, rad, lilo, mj, mh, mk = np.genfromtxt('isochrones/baraffe_{}_gyr.txt'.format(str(int(age_guess * 10)).zfill(5)), unpack = True, autostrip = True, comments = '!')

		if temp[-1] > t:
			idx = fn_pair(temp, logg, t, l)
			m = mass[idx]
			ll = lum[idx]
			masses.append(m)
			lums.append(ll)
			num.append(number)

		else:
			ag = np.log10(age_guess * 1e6)
			matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
			ages = matrix[:, 1]
			age_new = ages[mft.find_nearest(ages, ag)]
			matrix = matrix[np.where(matrix[:, 1] == age_new)]

			mat = matrix[:,4:8]
			#mat columns are:
			#mass (0), logL (1), logTe (2), logg (3)
			mass = mat[:,0]
			lum = [10**(ll) for ll in mat[:, 1]]
			teff = [10 ** (tt) for tt in mat[:, 2]]
			llog = mat[:, 3]
			idx = int(fn_pair(teff, llog, t, l)[0])

			masses.append(mass[idx])
			lums.append(lum[idx])
			num.append(number)
	np.savetxt(runfolder + '/results/mass_fit_results.txt', np.column_stack((num, masses, lums)), header= '# Mass, Luminosity')
	return masses, lums

def plot_specs(num, run):
	cwd = os.getcwd()
	wl, spec = np.genfromtxt(run + '/specs/spec_{}.txt'.format(num), unpack = True)

	ewl = np.linspace(wl[0], wl[-1], len(wl))
	inte = interp1d(wl, spec)
	espec = inte(ewl)

	ewl, espec = mft.broaden(ewl, espec, 3000, 0, 0)
	wl, spec = redres(ewl, espec, 8)

	p_num, p_mass, mul, p_age, p_av, p_temp, p_logg, p_luminosity, distance = np.genfromtxt(run + '/ppar.txt', unpack = True)
	pnum, multiplicity, p_temp, p_logg = int(p_num[np.where(p_num == num)[0]]), mul[np.where(p_num == num)[0]], p_temp[np.where(p_num == num)[0]], p_logg[np.where(p_num == num)[0]]

	xs, temp, lg, norm = np.genfromtxt(cwd + '/' + run+'/results/params_spec_{}.txt'.format(num), unpack = True)

	idx = np.where(xs == min(xs))

	t, l, n = temp[idx], lg[idx], norm[idx]

	w, s = mft.get_spec(t, l, [0.55, 0.95], normalize = False)
	w_ = np.linspace(w[0], w[-1], len(w))

	model_intep = interp1d(w, s)
	s_ = model_intep(w_)

	w_, s_ = mft.broaden(w_, s_, 3000, 0, 0)
	w_, s_ = redres(w_, s_, 8)

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

		bf = [0, 0, 0, 0]#[0.35, 0.4, 0.45, 0.5]
		md = [0.5, 1, 1.5]#[0.8, 1, 1.5]

		if not os.path.exists(cwd + '/' + run):
			os.mkdir(cwd + '/' + run)
		if not os.path.exists(cwd + '/' + run + '/specs'):
			os.mkdir(cwd + '/' + run + '/specs')
		if not os.path.exists(cwd + '/' + run + '/results'):
			os.mkdir(cwd + '/' + run + '/results')

		print('Making systems')

		pool = mp.Pool()
		results = [pool.apply_async(make_binary_sys, args = (ns, 1, bf, md, 1, 0, run)) for ns in range(nsys)]
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

	ts = [np.round(t/100) * 100 for t in t_g]
	ls = [np.round(l / 0.5) * 0.5 for l in lg_g]
	trange = [np.arange(((np.round(t / 100) * 100) - 300), ((np.round(t / 100) * 100) + 1000), 100) for t in ts]
	lgrange = [3.5, 4]
	
	files = np.array([run + '/specs/spec_{}.txt'.format(n) for n in range(nsys)])

	pool = mp.Pool()
	results = [pool.apply_async(even_simpler, args = (files[n], ts[n], ls[n], trange[n], lgrange)) for n in range(nsys)]
	o = [p.get() for p in results]

	print('Plotting!')

	#even_simpler(f, np.round(t / 100) * 100, np.round(l / 0.5) * 0.5, np.arange(((np.round(t / 100) * 100) - 300), ((np.round(t / 100) * 100) + 300), 100), [3.5, 4])

	[plot_specs(n, run) for n in range(nsys)]

	print('Running mass analysis')

	analyze_sys(run, 1)

	n, mass, logg = np.genfromtxt(run + '/ppar.txt', unpack = True)
	num, fit_mass, m1, m2, m3, m4, m5, m6, m7 = np.genfromtxt(run + '/results/mass_fit_results.txt', unpack = True)

	mass_resid = [] 
	for k in n: 
		idx = np.where(num == k) 
		test = mass[int(k)] - fit_mass[idx] 
		mass_resid.append(test[0]) 

	mr = [mass_resid[n]/mass[n] for n in range(len(mass))]

	plt.figure(1)
	plt.hist(mass, label = 'input')
	plt.hist(fit_mass, alpha = 0.7, label = 'output') 
	plt.legend()
	plt.savefig(run + '/masshist.pdf')
	plt.close()

	plt.figure(2)
	plt.hist(mass_resid, bins = 20)	
	plt.xlabel('Mass residual, solar masses')
	plt.tight_layout()
	plt.savefig(run + '/error_fit.pdf')
	plt.close()

	plt.figure(3)
	plt.hist(mr, bins = 20, label = 'Avg. frac. error = {:.2f}\n 1 stdev = {:.2f}'.format(np.mean(mr), np.std(mr)))
	plt.legend() 
	plt.xlabel('Fractional mass residual')
	plt.tight_layout()
	plt.savefig(run + '/error_fit_fractional.pdf')
	plt.close()
	return


run_pop(16, 'run4', new_pop = True) 

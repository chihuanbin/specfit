"""
.. module:: IMF
   :platform: Unix, Windows
   :synopsis: Synthetic population production

.. moduleauthor:: Kendall Sullivan <kendallsullivan@utexas.edu>

Dependencies: numpy, matplotlib, astropy, scipy, model_fit_tools_v2, MPI4py
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
from scipy.optimize import root, minimize, leastsq
from scipy.integrate import trapz, simps
import model_fit_tools_v2 as mft
import multiprocessing as mp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.cm as cm
import extinction
from mpi4py import MPI
from labellines import labelLine, labelLines

def redres(wl, spec, factor):
	"""Reduces spectral resolution to mimic pixellation at the surface of a CCD
	Args:
		wl (array): wavelength array 
		spec (array): flux array
		factor (int): factor by which to reduce the resolution

	Note:
		wl and spec must be the same length. For consistency in reducing the wavelength array it should be evenly spaced.

	Returns:
		wlnew, specnew (tuple): two lists, where the first is the reduced wavelength array and the second is the reduced-resolution flux array
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
			p.append(0.097*np.exp(-((np.log(m) - np.log(0.3))**2)/(2 * 0.55**2)))
		else:
			#do Salpeter power law: x = 1.3 for the log version
			p.append(0.0095*m**-1.3)
	#p = [p[n]/(max(massrange) - min(massrange)*massrange[n] * np.log(10)) for n in range(len(p))]
	#p = [pn/max(p) for pn in p]
	return np.array(p)

def calc_pct(imf, wh = 'chabrier'):
	x = np.arange(0.09, 100, 0.05)

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

def get_params(mass, age, which = 'parsec'):
	'''
	input: mass (solar masses) and age (megayears, 1e6). 
	requires evolutionary models in a folder called "isochrones" and atmosphere models in a folder called "phoenix_isos", at the moment
	Uses input mass and age to get physical parameters (luminosity, radius) from Baraffe isochrones, 
	then uses those physical parameters to get a temperature and log g from the phoenix BT-Settl isochrones (to match the model spectra)
	'''
	a = str(int(age * 10)).zfill(5)
	if which == 'baraffe':
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

		a_l = griddata((np.hstack((m1, m2)), np.hstack((aaa1, aaa2))), np.hstack((lum1, lum2)), (mass, age))#interp_2d(mass, age, np.hstack((m1, m2)), np.hstack((aaa1, aaa2)), np.hstack((lum1, lum2)))
		a_r = griddata((np.hstack((m1, m2)), np.hstack((aaa1, aaa2))), np.hstack((radius1, radius2)), (mass, age))#interp_2d(mass, age, np.hstack((m1, m2)), np.hstack((aaa1, aaa2)), np.hstack((radius1, radius2)))

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
			#temp, log_g = interp_2d(a_l, age, np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2)), np.hstack((teff1, teff2))), interp_2d(a_l, age, np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2)), np.hstack((logg1, logg2)))
			temp = griddata((np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2))), np.hstack((teff1, teff2)), (a_l, age))
			log_g = griddata((np.hstack((lu1, lu2)), np.hstack((aaa1, aaa2))), np.hstack((logg1, logg2)), (a_l, age))
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

	if which == 'parsec':
		age = np.log10(age * 1e6)
		matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
		ages = matrix[:, 1]
		aa = [ages[0]]
		for a in ages:
			if a != aa[-1]:
				aa.append(a)
		aa = np.sort(aa)
		ma, logl, logt, lg = matrix[:, 4], matrix[:, 5], matrix[:, 6], matrix[:,7]
		#mat columns are:
		#mass (0), logL (1), logTe (2), logg (3)
		a1 = mft.find_nearest(aa, age) 
		if aa[a1] > age:
			a2 = a1 - 1
		elif aa[a1] < age:
			a2 = a1 + 1
		else:
			a1 = a2

		if a1 != a2:
			age1 = aa[min(a1, a2)]
			age2 = aa[max(a1,a2)]

			ages1, ages2 = ages[np.where(ages == age1)], ages[np.where(ages == age2)]
			ma1, ma2 = ma[np.where(ages == age1)], ma[np.where(ages == age2)]
			lt1, lt2 = logt[np.where(ages == age1)], logt[np.where(ages == age2)]
			lg1, lg2 = lg[np.where(ages == age1)], lg[np.where(ages == age2)]
			logl1, logl2 = logl[np.where(ages == age1)], logl[np.where(ages == age2)]

			mag1 = [10 ** (-0.4 * matrix[:, n][np.where(ages == age1)]) for n in range(12, 18)] #VRIJHK
			mag2 = [10 ** (-0.4 * matrix[:, n][np.where(ages == age2)]) for n in range(12, 18)]

			a_t = griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((lt1, lt2)), (age, mass))
			a_t = 10 ** a_t
			logg = griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((lg1, lg2)), (age, mass))
			a_l = griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((logl1, logl2)), (age, mass))
			mm = [griddata((np.hstack((ages1, ages2)), np.hstack((ma1, ma2))), np.hstack((mag1[n], mag2[n])), (age, mass)) for n in range(len(mag1))]
			mm = [-2.5 * np.log10(mm[n]) for n in range(len(mm))]
			return a_t, logg, a_l, mm
		else:
			print('grid point! Fix this, Kendall')

def fn_pair(mat1, mat2, val1, val2):
	ret1 = [np.abs(float(m) - val1) for m in mat1]
	ret2 = [np.abs(float(m) - val2) for m in mat2]

	min_sum = [ret1[n] + ret2[n] for n in range(len(ret1))]

	idx = np.where(min_sum == min(min_sum))[0][0]
	return idx

def fn_trio(mat1, mat2, mat3, val1, val2, val3):
	ret1 = [np.abs(float(m) - val1[0]) for m in mat1]
	ret2 = [np.abs(float(m) - val2[0]) for m in mat2]
	ret3 = [np.abs(float(m) - val3[0]) for m in mat3]

	min_sum = [ret1[n] + ret2[n] + ret3[n] for n in range(len(ret1))]

	idx = np.where(min_sum == min(min_sum))[0][0]
	return idx

def match_pars(temp1, temp2, lf):
	lg = 4

	wl1, spec1 = mft.get_spec(temp1, lg, [0.45, 2.5], normalize = False, reduce_res = False, resolution = 3000)
	wl2, spec2 = mft.get_spec(temp2, lg, [0.45, 2.5], normalize = False, reduce_res = False, resolution = 3000)

	spec2 *= lf

	spec = [spec1[n] + spec2[n] for n in range(len(spec1))]

	return wl, spec

def get_secondary_mass(pri_mass):
	#from raghavan 2010 Fig. 16
	sm = 0
	while sm < 0.09:
		r = np.random.RandomState()
		rn = r.random_sample(size=1)[0]
		if rn < (5/110):
			mr = r.uniform(low=0.05, high = 0.2)
		elif rn > 0.9:
			mr = r.uniform(low=0.95, high = 1)
		else:
			mr = r.uniform(low=0.2, high = 0.95)
		sm = pri_mass * mr
	return sm

def get_distance(sfr):
	r = np.random.RandomState()
	if sfr == 'taurus':
		return r.randint(low=125, high=165)

def make_mass(n_sys):
	r = np.random.RandomState()#seed = 234632)
	masses = np.linspace(0.09, 2.5, 1000)
	prob_dist = make_chabrier_imf(masses)
	pd = [prob_dist[n]/np.sum(prob_dist) for n in range(len(prob_dist))]
	cumu_dist = [np.sum(pd[:n]) for n in range(len(pd))]
	r_num = r.random_sample(size = n_sys)
	mass = []
	for rn in r_num:
		idx = 0
		for n in range(0, len(cumu_dist) -1):
			if rn > cumu_dist[n] and rn <= cumu_dist[n + 1]:
				idx = n
		mass.append(masses[idx])
	return mass

def make_binary_sys(n, n_sys, multiplicity, mass_bins, age, av, run, sfr = 'taurus', model = 'parsec'):
	#from Raghavan + 2010, eccentricity can be pulled from a uniform distribution
	#From the same source, the period distribution can be approximated as a Gaussian with
	#a peak at log(P) = 5.03 and sigma_log(P) = 2.28

	#decide if my mass will end up in the chabrier or salpeter regime by drawing a random number and comparing to the percentage
	#of the total cmf that is taken by the Chab. portion of the imf
	r = np.random.RandomState()#seed = 234632)

	age = r.uniform(low = age[0], high = age[-1])

	pri_array_keys = ['num', 'p_mass', 'multiplicity', 'age', 'av', 'temp', 'logg', 'luminosity', 'distance', 'v mag', 'r mag', 'i mag', 'j mag', 'h mag', 'k mag']
	p_arr_keys2 = ['#a multiplicity of 1 indicates a multiple at all - it\'s a flag, not a number of stars in the system\n #distance is in pc\
	\n #magnitudes are apparent at the distance assigned']

	kw = ['num', 's_mass', 'sep', 'age', 'eccentricity', 'period', 'temp', 'logg', 'luminosity', 'v mag', 'r mag', 'i mag', 'j mag', 'h mag', 'k mag']

	pri_pars = np.empty(len(pri_array_keys))
	sec_pars = np.empty(len(kw))
	mass = make_mass(n_sys)
	print(mass)

	spec_file = open(os.getcwd() + '/' + run + '/specs/spec_{}.txt'.format(n), 'w')
	ptemp, plogg, plum, mags = get_params(mass, 1.01)#(mass[n_sys - 1], age, which = model)
	plogg = 4

	dist = get_distance(sfr)
	dm = 5 * np.log10(dist/10) - 5
	pri_par = [n, mass[n_sys -1], 0, age, av, float(ptemp), float(plogg), float(plum), dist] #etc. - can add more later
	[pri_par.append(float(mags[n])) for n in range(len(mags))]
	pri_pars = np.vstack((pri_pars, np.array(pri_par)))
	if type(pri_par[1]) != None:
		pw, ps = mft.get_spec(ptemp, plogg, [0.45, 0.9], normalize = False)
		pri_wl, pri_spec = [],[]
		for k in range(len(pw)):
			pri_wl.append(float(pw[k]))
			pri_spec.append(float(ps[k]))

		rad = (1 /(2*np.sqrt(np.pi * 5.67e-5))) * np.sqrt((10**plum) * 3.826e33)/ptemp**2
		pri_spec = [float(ps * 4 * np.pi * rad**2) for ps in pri_spec]

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
			print(n)
			sec_par[0] = n
			sec_par[1] = get_secondary_mass(mass[n_sys - 1])
			sec_par[2] = r.uniform(10, 1e3)
			sec_par[3] = age
			sec_par[4] = r.uniform(0, 1)
			sec_par[5] = r.normal(5.03, 2.28)

			stemp, slogg, slum, smags = get_params(sec_par[1], age, which = model)
			slogg = 4
			sec_par[6:9] = stemp, slogg, slum
			sec_par[9:] = smags

			sec_pars = np.vstack((sec_pars, sec_par))

			sw, ss = mft.get_spec(stemp, slogg, [0.45, 0.9], normalize = False)
			sec_wl, sec_spec = [],[]
			for n in range(len(pw)):
				sec_wl.append(float(sw[n]))
				sec_spec.append(float(ss[n]))

			rads = (1 /(2* np.sqrt(np.pi * 5.6704e-5))) * np.sqrt((10**slum) * 3.826e33)/stemp**2
			sec_spec = [float(ps * 4 * np.pi * rads**2) for ps in sec_spec]

			# s_factor = slum/plum
			# pflux_sum, sflux_sum = np.sum(comb_spec), np.sum(sec_spec)
			# flux_factor = sflux_sum/pflux_sum
			# norm = s_factor / flux_factor
			# comb_spec = [comb_spec[t] + norm * sec_spec[t] for t in range(len(sec_spec))]

			mindiff = np.inf

			for k in range(1, len(pri_wl)):
				if pri_wl[k] - pri_wl[k-1] < mindiff:
					mindiff = pri_wl[k] - pri_wl[k-1]

			wl = np.arange(max(pri_wl[0], sec_wl[0]), min(pri_wl[-1],sec_wl[-1]), mindiff)

			i1, i2 = interp1d(pri_wl, pri_spec), interp1d(sec_wl, sec_spec)
			new_pri, new_sec = i1(wl), i2(wl)

			pri_wl, comb_spec = wl, [new_pri[n] + new_sec[n] for n in range(len(wl))]

		if av > 0:
			factor = [1, 0.751, 0.479, 0.282, 0.190, 0.114] #cardelli et al. 1989

			for n in range(len(mags)):
				pri_pars[n_sys][n - 6] = (av * factor[n]) + dm + mags[n] 

				if len(sec_pars.shape) == 1:
					sec_pars[n-6] = (av * factor[n]) + dm + sec_pars[n-6]

				else:
					sec_pars[n_sys][n-6] = (av * factor[n]) + dm + sec_pars[n_sys][n-6]


			pri_wl, comb_spec = np.array(pri_wl), np.array(comb_spec)
			comb_spec = mft.extinct(pri_wl, comb_spec, av)

		comb_spec = [np.abs(c) for c in comb_spec]

		for k in range(len(comb_spec)):
			spec_file.write(str(pri_wl[k]) + ' ' + str(comb_spec[k]) + '\n')
		
		n += 1

	else:
		mass[n] = make_mass(1)

	return pri_pars[1], sec_pars

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
	ax.set_xlabel(r'Mass ($M_{\odot}$)', fontsize = 13)
	ax.set_ylabel('Number of stars', fontsize = 13)
	ax.set_xlim(0.08, 2.1)
	#ax.set_ylim(1e2, 1e3)
	#ax.set_yscale('log')
	ax.set_xscale('log')
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig('masshist_init.pdf')
	plt.show()
	plt.close(fig)

	return

def find_extinct(model_wl, model, data):
	init_guess = 2
	step = 0.5
	niter = 0

	while step > 0.2 and niter < 10:
		print('extinction: ', init_guess, step)
		#convert magnitudes of extinction to flux
		extinct_minus = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess - step))
		extinct_init = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess))
		extinct_plus = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess + step)) 

		model_minus = [model[n] - extinct_minus[n] for n in range(len(model))]
		model_init = [model[n] - extinct_init[n] for n in range(len(model))]
		model_plus = [model[n] - extinct_plus[n] for n in range(len(model))]

		minus_var = np.mean([m * 0.01 for m in model_minus])
		init_var = np.mean([m * 0.01 for m in model_init])
		plus_var = np.mean([m * 0.01 for m in model_plus])

		xs_minus = mft.chisq(data, model_minus, minus_var)
		xs_init = mft.chisq(data, model_init, init_var)
		xs_plus = mft.chisq(data, model_plus, plus_var)

		niter += 1
		if xs_init < xs_minus and xs_init < xs_plus:
			step *= 0.5
			niter = 0
		elif xs_init > xs_minus and xs_init < xs_plus:
			init_guess = init_guess - (init_guess * step)
		elif xs_init < xs_minus and xs_init > xs_plus:
			init_guess = init_guess + (init_guess * step)
		else:
			if xs_minus < xs_plus:
				init_guess = init_guess - (init_guess * step)
			else:
				init_guess = init_guess + (init_guess * step)

	extinct = 10 ** (-0.4 * extinction.fm07(model_wl, init_guess))
	model = [model[n] - extinct[n] for n in range(len(model))]

	return model, init_guess

def find_norm(model_wl, model, data):
	init_guess = max(data)/max(model)

	model_wl = np.array(model_wl)
	step = 0.1e24

	while step > 0.001e24:
		print('norm: ', init_guess, step)
		model_minus = [model[n] * (init_guess - (init_guess * step)) for n in range(len(model))]
		model_init = [model[n] * init_guess for n in range(len(model))]
		model_plus = [model[n] * (init_guess + (init_guess * step)) for n in range(len(model))]

		model_minus, extinct_minus = find_extinct(model_wl, model_minus, data)
		model_init, extinct_init = find_extinct(model_wl, model_init, data)
		model_plus, extinct_plus = find_extinct(model_wl, model_plus, data)

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

	return init_guess, extinct_init

def even_simpler(filename, t_guess, lg_guess, t_range, lg_range):
	wl, spec = np.genfromtxt(filename, unpack = True)
	wle = np.linspace(wl[0], wl[-1], len(spec))
	intep = interp1d(wl, spec) 
	#erg/s/cm^2/A
	spece = intep(wle)

	wle, spece = mft.broaden(wle, spece, 3000, 0, 0)
	#wle, spece = redres(wle, spece, 6)

	xs = []
	temp = []
	logg = []
	norm = []
	st = []
	extinct = []

	for l in lg_range:
		t_init = t_guess

		step = 50
		niter = 0

		while step >= 10:
			print(step, t_init)
			t_minus = t_init - step
			t_plus = t_init + step

			ww_minus, ss_minus = mft.get_spec(t_minus, l, [0.45, 2.5], normalize = False)
			ww_init, ss_init = mft.get_spec(t_init, l, [0.45, 2.5], normalize = False)
			ww_plus, ss_plus = mft.get_spec(t_plus, l, [0.45, 2.5], normalize = False)																													

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
			#wwe_minus, sse_minus = redres(wwe_minus, sse_minus, 6)

			wwe_init, sse_init = mft.broaden(wwe_init, sse_init, 3000, 0, 0)
			#wwe_init, sse_init = redres(wwe_init, sse_init, 6)

			wwe_plus, sse_plus = mft.broaden(wwe_plus, sse_plus, 3000, 0, 0)
			#wwe_plus, sse_plus = redres(wwe_plus, sse_plus, 6)


			second_wl = np.linspace(max(wle[0], wwe_init[0], wwe_minus[0], wwe_plus[0]), min(wle[-1], wwe_init[-1], wwe_minus[-1], wwe_plus[-1]), len(wwe_init))

			di2 = interp1d(wle, spece)
			mi2_minus = interp1d(wwe_minus, sse_minus)
			mi2_init = interp1d(wwe_init, sse_init)
			mi2_plus = interp1d(wwe_plus, sse_plus)

			spece2 = di2(second_wl)
			sse_minus = mi2_minus(second_wl)
			sse_init = mi2_init(second_wl)
			sse_plus = mi2_plus(second_wl)

			n_minus, extinct_minus = find_norm(wwe_minus, sse_minus, spece2)
			n_init, extinct_init = find_norm(wwe_init, sse_init, spece2)
			n_plus, extinct_plus = find_norm(wwe_plus, sse_plus, spece2)

			sse_minus = [s * n_minus for s in sse_minus]
			sse_init = [s * n_init for s in sse_init]
			sse_plus = [s * n_plus for s in sse_plus]

			var = [sp * 0.01 for sp in spece2]
			
			cs_minus = mft.chisq(sse_minus, spece2, var)
			cs_init = mft.chisq(sse_init, spece2, var)
			cs_plus = mft.chisq(sse_plus, spece2, var)

			niter += 1

			# print(step, cs_minus, cs_init, cs_plus, t_init)
			st.append(step)
			xs.append(cs_init)
			temp.append(t_init)
			logg.append(l)
			norm.append(n_init)
			extinct.append(extinct_init)

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

	print('saving')
	np.savetxt(filename.split('/')[0] + '/results/params_' + filename.split('/')[-1], np.column_stack((st, xs, temp, logg, norm, extinct)), \
		header = '#step size, chi square, temperature, log(g), normalization, extinction')
	#np.savetxt('results/params_' + filename.split('/')[-1], np.column_stack((xs, temp, logg, norm)), header = '# chi square, temperature, log(g), normalization')

	return

def find_best(pars, run, number):
	# fn = open(run + '/results/lm_pars_{}.txt'.format(number), 'a+')
	# fn.write([str(pars[n]) for n in len(pars)] + '\n')
	# fn.close()

	filename = run + '/specs/spec_{}.txt'.format(number)
	wl, spec = np.genfromtxt(filename, unpack = True)
	wle = np.linspace(wl[0], wl[-1], len(spec))
	intep = interp1d(wl, spec) 
	#erg/s/cm^2/A
	spece = intep(wle)
	wle, spece = mft.broaden(wle, spece, 3000, 0, 0)
	teff, logg, normalization, extinct = pars[0], pars[1], pars[2], pars[3]#pars['teff'], pars['logg'], pars['norm'], pars['extinct']
	if teff < 5500 and teff > 2700 and logg > 3.5 and logg < 5.5:
		ww_init, ss_init = mft.get_spec(teff, logg, [0.45, 2.5], normalize = False)

		wwe_init = np.linspace(ww_init[0], ww_init[-1], len(wle))

		intep_init = interp1d(ww_init, ss_init)
		sse_init = intep_init(wwe_init)
		wwe_init, sse_init = mft.broaden(wwe_init, sse_init, 3000, 0, 0)

		second_wl = np.linspace(max(wle[0], wwe_init[0]), min(wle[-1], wwe_init[-1]), len(wwe_init))

		di2 = interp1d(wle, spece)
		mi2_init = interp1d(wwe_init, sse_init)
		spece2 = di2(second_wl)
		sse_init = mi2_init(second_wl)

		extinct_init = 10 ** (-0.4 * extinction.fm07(second_wl, extinct))

		sse_init = [sse_init[n] - extinct_init[n] for n in range(len(sse_init))]
		sse_init = [s * normalization for s in sse_init]

		var = [sp * 0.01 for sp in spece2]
		cs_init = mft.chisq(sse_init, spece2, var)
		f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
		f.write('{} {} {} {} {}\n'.format(np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
		f.close()
	else:
		cs_init = np.full(len(wle), np.inf)
		f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
		f.write('{} {} {} {} {}\n'.format(np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
		f.close()
	try:
		return cs_init
	except:
		return np.full(len(wle), np.inf)

def simplex_fit(teff, logg, normalization, extinct, filename):
	run = filename.split('/')[0]
	number = int(filename.split('_')[1].split('.')[0])
	#simp = np.array(([teff + 100, logg, normalization, extinct],
	#	[teff, logg, normalization + normalization * 0.05, extinct],
	#	[teff, logg, normalization, extinct + 0.5],
	#	[teff, logg - 0.2, normalization, extinct],
	#	[teff - 100, logg + 0.2, normalization, extinct - 0.5]))
	#print(simp)
	#a = minimize(find_best, np.array([teff, logg, normalization, extinct]), args = (run, number), method = 'Nelder-Mead', options = {'adaptive': True, 'initial_simplex': simp}, tol = 10)
	#a = leastsq(find_best, [teff, logg, normalization, extinct], args = (run, number))
	a = root(find_best, [teff, logg, normalization, extinct], args = (run, number), method = 'lm', options = {'ftol':1e-1})
	#par = lmfit.Parameters()
	#par.add_many(('teff', teff, True, 2800, 6500),
	#			('logg', logg, True, 3.5, 5),
	#			('norm', normalization, True, 1e22, 1e26),
	#			('extinct', extinct, True, 0, 5))
	#
	#a = lmfit.minimize(find_best, par, args = (run, number), options = {'ftol': 10})
	print('fitting done')
	np.savetxt(run + '/results/{}_simplex.txt'.format(number), a.x)
	return

def fit_test(teff, logg, norm, extinct, filename, maxiter = 1000):

	run = filename.split('/')[0]
	number = int(filename.split('_')[1].split('.')[0])
	logg, logg_try = 4, 4

	wl, spec = np.genfromtxt(filename, unpack = True)

	teff += np.random.normal(scale = 100)

	extinct += np.random.normal(scale = 0.1)
	while extinct < 0:
		extinct += np.random.normal(scale = 0.1)
	
	ww_init, ss_init = mft.get_spec(teff, logg, [0.45, 0.9], normalize = False)

	min_diff = np.inf
	for n in range(1, len(spec)):
		if wl[n] - wl[n-1] < min_diff:
			min_diff = wl[n] - wl[n-1]

	second_wl = np.arange(max(wl[0], ww_init[0], 6000) + 5, min(wl[-1], ww_init[-1], 9000) - 5, min_diff/2)

	dd2 = interp1d(wl, spec)
	md2_init = interp1d(ww_init, ss_init)
	spec2 = dd2(second_wl)
	ss_init = md2_init(second_wl)

	#plt.plot(second_wl, spec2)
	#plt.plot(second_wl, ss_init)
	#plt.savefig('spec_{}.png'.format(number))
	#plt.close()

	ss_init = mft.extinct(second_wl, ss_init, extinct)
	normalization = np.mean(spec2)/np.mean(ss_init)

	normalization += np.random.normal(scale = np.abs(normalization * 0.02))

	ss_init = [s * normalization for s in ss_init]
	ref_variance = 0.01 * np.median(spec2)

	vary = [ref_variance * np.sqrt(s/np.median(spec2)) for s in spec2]

	cs_init = mft.chisq(ss_init, spec2, vary)

	niter = 0
	total_iter = 0

	# plt.figure()
	# plt.plot(ss_init, label = 'initial test')
	# plt.plot(spec2, label = 'data')
	# plt.legend()
	# plt.savefig('initial_test.png')
	# plt.close()

	f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
	f.write('{} {} {} {} {} {}\n'.format(niter, np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
	f.close()

	#extinction and normalization block
	# while niter < 80:
		#rn = np.random.randint(0, 2)
		#rn2 = np.random.rand()

		# parvar = np.asarray([normalization, extinct])

		# if niter < 40:
		# 	var = np.asarray([np.random.normal(scale=np.abs(normalization*0.1)), np.random.normal(scale = 0.1)])
		# 	while parvar[1] + var[1] < 0:
		# 		var[1] = np.random.normal(scale = 0.1)

		# else:
		# 	var = np.asarray([np.random.normal(scale=np.abs(normalization*0.02)), np.random.normal(scale = 0.05)])
		# 	while parvar[1] + var[1] < 0:
		# 		var[1] = np.random.normal(scale = 0.05)

		# parvar = parvar + var

		# normalization_try, extinct_try = parvar

		# if teff < 7000 and teff > 2000 and extinct_try >= 0:# and normalization > 0:

		# 	ww_i, ss_i = mft.get_spec(teff, logg, [0.45, 0.9], normalize = False)

		# 	mi2_init = interp1d(ww_i, ss_i)
		# 	sse_i = mi2_init(second_wl)

		# 	sse_i = mft.extinct(second_wl, sse_i, extinct_try)
		# 	sse_i = [s * normalization_try for s in sse_i]
		# 	vv = [0.01 * s for s in spec2]

		# 	cs_try = mft.chisq(sse_i, spec2, vv)
 
		# 	# wll, sss = mft.get_spec(teff, logg, [0.6, 0.9], normalize = False)
		# 	# sss = [s * normalization for s in sss]

		# 	# fig, ax2 = plt.subplots(nrows = 1)
		# 	# # # ax1.plot(wl2, spece2, label = 'spectrum')
		# 	# # # ax1.plot(wll, sss, label = 'current best fit, chisq = {}'.format(np.sum(cs_init)))
		# 	# # # ax1.legend(loc = 'best')

		# 	# ax2.plot(wl2, sse_i, label = 'test, chisq = {:.0f}'.format(np.sum(cs_try)))
		# 	# ax2.plot(wl2, spece2, label = 'spectrum')
		# 	# ax2.legend(loc = 'best')
		# 	# plt.savefig(run + '/results/test{}_{}.png'.format(number, total_iter))
		# 	# plt.close()
		# 	# print(np.sum(cs_try)/len(cs_try))

		# 	if np.sum(cs_try)/len(cs_try) < np.sum(cs_init)/len(cs_init):
		# 		normalization, extinct = normalization_try, extinct_try
		# 		cs_init = cs_try
		# 		if niter >= 40:
		# 			niter = 40
		# 		else:
		# 			niter = 0
		# 	else:
		# 		niter += 1

		# 	f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
		# 	f.write('{} {} {} {} {} {}\n'.format(niter, np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
		# 	f.close()

		# 	f = open(run + '/results/testpars_{}.txt'.format(number), 'a')
		# 	f.write('{} {} {} {} {} {}\n'.format(0, np.sum(cs_try)/len(cs_try), teff, logg, normalization_try, extinct_try))
		# 	f.close()
		# 	total_iter += 1

	#everything block
	while niter < 10:
		#rn = np.random.randint(0, 3)
		#rn2 = np.random.rand()

		parvar = np.asarray([teff, normalization, extinct])

		if niter < 6:
			var = np.asarray([np.random.normal(scale = 100), np.random.normal(scale=np.abs(normalization*0.01)), np.random.normal(scale = 0.05)])
			while parvar[2] + var[2] < 0:
				var[2] = np.random.normal(scale = 0.05)

		else:
			var = np.asarray([np.random.normal(scale = 5), np.random.normal(scale=np.abs(normalization*0.005)), np.random.normal(scale = 0.01)])
			while parvar[2] + var[2] < 0:
				var[2] = np.random.normal(scale = 0.01)

		parvar = parvar + var

		teff_try, normalization_try, extinct_try = parvar

		if teff_try < 7000 and teff_try > 2000:

			ww_i, ss_i = mft.get_spec(teff_try, logg, [0.45, 0.9], normalize = False)

			mi2_init = interp1d(ww_i, ss_i)
			sse_i = mi2_init(second_wl)
			
			sse_i = mft.extinct(second_wl, sse_i, extinct_try)
			sse_i = [s * normalization_try for s in sse_i]
			# vv = [0.01 * s for s in spec2]

			# fig, ax2 = plt.subplots(nrows = 1)
			# ax2.plot(wl2, sse_i, label = 'test, chisq = {:.0f}'.format(np.sum(cs_try)))
			# ax2.plot(wl2, spece2, label = 'spectrum')
			# ax2.legend(loc = 'best')
			# plt.savefig(run + '/results/test{}_{}.png'.format(number, total_iter))
			# plt.close()

			cs_try = mft.chisq(sse_i, spec2, vary)

			if np.sum(cs_try)/len(cs_try) < np.sum(cs_init)/len(cs_init):
				teff, logg, normalization, extinct = teff_try, logg_try, normalization_try, extinct_try
				cs_init = cs_try
				if niter >= 6:
					niter = 6
				else:
					niter = 0
			else:
				niter += 1

			if niter == 10 and np.sum(cs_init)/len(cs_init) > 1.5:
				niter = 3

			f = open(run + '/results/parvals_{}.txt'.format(number), 'a')
			f.write('{} {} {} {} {} {}\n'.format(niter, np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
			f.close()

			f = open(run + '/results/testpars_{}.txt'.format(number), 'a')
			f.write('{} {} {} {} {} {}\n'.format(1, np.sum(cs_try)/len(cs_try), teff_try, logg_try, normalization_try, extinct_try))
			f.close()
			total_iter += 1

		else:
			teff_try, normalization_try, extinct_try = teff_try + np.random.normal(scale = 200),\
				normalization_try + np.random.normal(scale = np.abs(normalization_try * 0.1)), extinct_try + np.random.normal(scale = 0.1)
	return np.sum(cs_try)/len(cs_try)

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

	z11 = zval_mat[fn_pair(t_mat, lum_mat, t1, lum1)]
	z22 = zval_mat[fn_pair(t_mat, lum_mat, t2, lum2)]
	z12 = zval_mat[fn_pair(t_mat, lum_mat, t1, lum2)]
	z21 = zval_mat[fn_pair(t_mat, lum_mat, t2, lum1)]

	fxy1 = (((t2 - temp_point)/(t2 - t1))*z11) + (((temp_point - t1)/(t2 - t1))*z21)
	fxy2 = (((t2 - temp_point)/(t2-t1))*z12) + (((temp_point - t1)/(t2 - t1))*z22)
	fxy = (((lum2 - lum_point)/(lum2 - lum1))*fxy1) + (((lum_point - lum1)/(lum2 - lum1)) * fxy2)
	return fxy
	
def fit_lum(mag, ts):
	matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)

	age = matrix[:, 1]
	age = [(10**a)/1e6 for a in age]
	a1 = [age[0]]
	for n in range(len(age)):
		if age[n] != a1[-1]:
			a1.append(age[n])

	lum, teff = matrix[:, 5], matrix[:, 6]
	teff = np.asarray([10**(t) for t in teff])
	mags = matrix[:, -6:] #[V R I J H K]

	lum_cand = []

	# print('mag, mags', mag)
	mag = [10** (-0.4 * (m - 0.3)) for m in mag]
	# print('mag', mag)
	
	for a in a1: #leave off ZAMS
		t = teff[np.where(age == a)]
		m = mags[np.where(age == a), :][0]
		l = 10 ** (lum[np.where(age == a)])

		for n, row in enumerate(m):
			m[n] = [10 ** (-0.4 * r) for r in row]

		ll = []
		for n in range(len(mag)):
			# print('mags', mag[n], min(m[:, n]), max(m[:, n]))
			testi = interp1d(m[:, n], l, fill_value = 'extrapolate')
			testl = testi(mag[n])
			# print(min(m[:, n]), max(m[:, n]), mag[n], min(l), max(l), testl)
			# print('mag, test l', mag[n], testl, l[mft.find_nearest(l, testl)])
			# for k in range(len(m[:, n])):
			# 	print(m[k, n], l[k])
			ll.append(testl)

		# print('total possible luminosities', ll)
		lum_cand.append(np.min(ll))

		# tdiff = np.asarray([te - ts for te in t])
		# if max(tdiff) > 0 and min(tdiff) < 0:
		# 	idx1 = 0
		# 	i = 0
		# 	while tdiff[i] < 0:
		# 		idx1 = i
		# 		i += 1
		# 	idx2 = idx1 + 1

		# 	tp1, tp2 = t[idx1], t[idx2]
		# 	mags1, mags2 = m[idx1, :], m[idx2, :]
		# 	lum1, lum2 = l[idx1], l[idx2]

		# 	# if a == 1e-6:
		# 		# print(mag, mags1, mags2, idx1, idx2, tdiff, ts, tp1, tp2)
		# 	mags1, mags2 = [10**(-0.4 * n) for n in mags1], [10**(-0.4 * n) for n in mags2]

		# 	mm = []
		# 	ll = []

		# 	for n in range(len(mag)):
		# 		#want to use teff to predict mag -> interpolate each set of mags in terms of teff, plug in ts 
		# 		#do the same for luminosity so that each (teff, mag) set has an associated lum.
		# 		#then, preserve each mag set and luminosity, compare mags with "observed" mag
		# 		#this is a best fit set of mags and luminosity *per age*, in terms of the effective temperature

		# 		i = interp1d([tp1,tp2], [mags1[n], mags2[n]])
		# 		newmag = i(ts)

		# 		mm.append(newmag)

		# 	il = interp1d([tp1, tp2], [lum1, lum2])
		# 	newlum = il(ts)
		# 	ll.append(newlum)

		# 	mag_cand = np.vstack((mag_cand, mm))
		# 	lum_cand.append(float(ll[0]))
		# 	age_cand.append(a)

	# mag_cand = mag_cand[1:, :]

	# #Now I have an array of the predicted (mags, luminosity) for a given teff (the input teff) for each age
	# #now i need to interpolate mags as a fn of luminosity, and plug in the given mags to get the right luminosity

	# l_try = []

	# mag = [10 ** (-0.4 * m) for m in mag]

	# for n,m in enumerate(mag):
	# 	i = interp1d(mag_cand[:, n], lum_cand, fill_value="extrapolate")
	# 	print(mag_cand[:, n], m)
	# 	l_try.append(i(m))

	# l_try = np.log10(np.mean([10 ** l for l in l_try]))
	# print(np.mean(lum_cand), np.std(lum_cand))
	# print('outpput', np.log10(np.median(lum_cand)), np.log10(lum_cand))
	return np.log10(np.mean(lum_cand))

def analyze_sys(runfolder, model = 'parsec'):
	'''
	Args: 
		runfolder (string): path to follow
		

	Returns:
		Mass and Luminosity from teff and log(g).

	'''
	csqs = glob(runfolder + '/results/parvals*.txt')#glob(runfolder + '/results/params*.txt')

	pnum, pmass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = np.genfromtxt(runfolder + '/ppar.txt', unpack = True)

	pnum, pmass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = pnum[np.argsort(pnum)], pmass[np.argsort(pnum)],\
		 multiplicity[np.argsort(pnum)], page[np.argsort(pnum)], av[np.argsort(pnum)], ptemp[np.argsort(pnum)], plogg[np.argsort(pnum)], pluminosity[np.argsort(pnum)], \
		 distance[np.argsort(pnum)], pvmag[np.argsort(pnum)], prmag[np.argsort(pnum)], pimag[np.argsort(pnum)], pjmag[np.argsort(pnum)], phmag[np.argsort(pnum)], pkmag[np.argsort(pnum)]

	snum, s_mass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = np.genfromtxt(runfolder + '/spar.txt', unpack = True)

	snum, s_mass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = snum[np.argsort(snum)], s_mass[np.argsort(snum)], \
		sep[np.argsort(snum)], sage[np.argsort(snum)], seccentricity[np.argsort(snum)], period[np.argsort(snum)], stemp[np.argsort(snum)], slogg[np.argsort(snum)],\
		 sluminosity[np.argsort(snum)], svmag[np.argsort(snum)], srmag[np.argsort(snum)], simag[np.argsort(snum)], sjmag[np.argsort(snum)], shmag[np.argsort(snum)], skmag[np.argsort(snum)]

	sluminosity = np.asarray([np.log10(10**(-0.4 * s)) for s in sluminosity])
	pmags = np.column_stack((pvmag, prmag, pimag, pjmag, phmag, pkmag))
	smags = np.column_stack((svmag, srmag, simag, sjmag, shmag, skmag))

	sl = np.zeros(len(pluminosity))
	tdiff = np.zeros(len(pluminosity))
	sys_mag = np.zeros(np.shape(pmags))

	for n in range(len(pluminosity)):
		if n in snum:
			sl[n] = np.log10(10**pluminosity[np.where(pnum == n)] + 10**sluminosity[np.where(snum == pnum[n])])
			# print(sl[n])
			# print(ptemp[n] - stemp[np.where(snum == pnum[n])], ptemp[n], stemp[np.where(snum == pnum[n])], np.where(snum == pnum[n]), pnum[n], snum[np.where(snum == pnum[n])])
			tdiff[n] = ptemp[n] - stemp[np.where(snum == pnum[n])]
			for k in range(len(pmags[0, :])):
				sys_mag[n][k] = -2.5 * np.log10((10 ** (-0.4 * pmags[n][k])) + (10 ** (-0.4 * smags[np.where(snum == pnum[n])[0][0]][k])))
		else:
			sl[n] = pluminosity[n]
			tdiff[n] = -1
			for k in range(len(pmags[0, :])):
				sys_mag[n][k] = pmags[n][k]

	# flux = [(10 ** sl[n])/(4 * np.pi * (distance[n] * 3.086e18)**2) for n in range(len(sl))] #erg/s/cm^2/A

	# d = 150 * 3.086e18

	# sl = [np.log10(flux[n] * 4 * np.pi * d**2) for n in range(len(sl))]

	masses = np.zeros(len(csqs))
	lums = np.zeros(len(csqs))
	ages = np.zeros(len(csqs))
	num = np.zeros(len(csqs))
	inp_age = np.zeros(len(csqs))

	inp_t = np.zeros(len(csqs))
	inp_l = np.zeros(len(csqs))
	inp_e = np.zeros(len(csqs))
	inp_n = np.zeros(len(csqs))

	out_t = np.zeros(len(csqs))
	out_l = np.zeros(len(csqs))
	out_e = np.zeros(len(csqs))
	out_n = np.zeros(len(csqs))	

	td = np.zeros(len(csqs))


	isos = glob('isochrones/baraffe*.txt')
	nums = []
	for i in isos:
		nn = i.split('_')[1]
		nums.append(nn)

	nums = sorted(nums)

	mass, temps, lum, logg, rad, lilo, mj, mh, mk = np.genfromtxt(glob('isochrones/baraffe_{}*.txt'.format(nums[0]))[0], unpack = True, autostrip = True, comments = '!')
	age = np.full(len(mass), int(glob('isochrones/baraffe_{}*.txt'.format(nums[0]))[0].split('_')[1])/10)

	for n in range(1, len(nums)):
		m, tt, lumi, llg, r, llo, mjj, mhh, mkk = np.genfromtxt(glob('isochrones/baraffe_{}*.txt'.format(nums[n]))[0], unpack=True, autostrip = True, comments = '!')
		a = np.full(len(m), int(glob('isochrones/baraffe_{}*.txt'.format(nums[n]))[0].split('_')[1])/10)
		age = np.column_stack((age, a))
		mass = np.column_stack((mass, m))
		temps = np.column_stack((temps, tt))
		lum = np.column_stack((lum, lumi))
		logg = np.column_stack((logg, llg))

	fig, ax = plt.subplots()
	age, mass, temps, lum, logg = age.flatten(), mass.flatten(), temps.flatten(), lum.flatten(), logg.flatten()


	matrix = np.genfromtxt('parsec_isos.dat', autostrip = True)
	aage = matrix[:, 1]
	aage = [(10**a)/1e6 for a in aage]
	#mat columns are:
	#mass (0), logL (1), logTe (2), logg (3)
	ma = matrix[:,4]
	lll = matrix[:, 5]
	teff = matrix[:, 6]#[10 ** (tt) for tt in matrix[:, 6]]
	llog = matrix[:, 7]

	vmag, rmag, imag = matrix[:, 11], matrix[:, 12], matrix[:, 13]

	a1 = [aage[0]]
	for n in range(len(aage)):
		if aage[n] != a1[-1]:
			a1.append(aage[n])

	aa1, ll1, tt1 = [np.full(len(np.where(aage == a1[0])[0]), a1[0])], [lll[np.where(aage == a1[0])]], [teff[np.where(aage == a1[0])]]
	tt1 = [10 ** t for t in tt1]
	ax.plot(tt1[0], ll1[0], label = '0')

	for n in range(1, len(a1)):
		a2 = np.full(len(np.where(aage == a1[n])[0]), a1[n])
		tt2 = teff[np.where(aage == a1[n])]
		tt2 = [10 ** t for t in tt2]

		ax.plot(tt2, lll[np.where(aage == a1[n])], label = '{}'.format(int(np.around(a1[n]))), color = cm.plasma(a1[n]/6), zorder = 0)

	xvals = list(np.linspace(6000,7000, 9))
	xvals.insert(0, 3500)
	labelLines(plt.gca().get_lines(), align = False, xvals = xvals, fontsize = 9, alpha = 0.8, zorder = 2.5)


	# m1 = [ma[0]]
	# for n in ma:
	# 	if n not in m1:
	# 		m1.append(n)

	# mm1, ll1, tt1 = [np.full(len(np.where(ma == m1[0])[0]), m1[0])], [lll[np.where(ma == m1[0])]], [teff[np.where(ma == m1[0])]]
	# tt1 = [10 ** t for t in tt1]
	# ax.plot(tt1[0], ll1[0])

	# for n in arange(1, len(m1),2):
	# 	m2 = np.full(len(np.where(ma == m1[n])[0]), m1[n])
	# 	tt2 = teff[np.where(ma == m1[n])]
	# 	tt2 = [10 ** t for t in tt2]

	# 	ax.plot(tt2, lll[np.where(ma == m1[n])], color = cm.plasma(m1[n]/6))

	a = np.array(page)
	b = np.array(sage)

	# pt1 = ax.scatter([4955, 4735], [0.454, 0.268], color = 'navy', label = r'Small $\Delta$T', s = 25)
	# ax.scatter(4971, 0.892, marker = 'x', color = 'navy')

	# pt2 = ax.scatter([3755, 2858], [-0.356, -1.152], color = 'xkcd:coral', label = r'Large $\Delta$T', s = 25)
	# ax.scatter(3138, -0.172, marker = 'x', color = 'xkcd:coral')
	inp = ax.scatter(ptemp, pluminosity, s = 25, color = 'xkcd:orange', label = 'Input')
	# ax.scatter(stemp, sluminosity, s = 25, color = 'xkcd:orange')

	avv = []
	for k, file in enumerate(csqs):
		number = int(file.split('.')[0].split('_')[1])
		#number = int(file.split('.')[0].split('_')[0].split('/')[-1])
		num[k] = number
		numb, cs, temp, lg, norm, ext = np.genfromtxt(file, unpack = True, autostrip = True)
		if cs[-1] > 1.5:
			print(k, cs[-1])
		#temp, lg, norm, extinct = np.genfromtxt(file, unpack = True, autostrip = True)
		ts =temp[np.where(cs == min(cs))][0]
		l = lg[np.where(cs == min(cs))][0]
		extinct = ext[np.where(cs == min(cs))][0]
		norm = norm[np.where(cs == min(cs))][0]

		inp_t[k] = ptemp[np.where(pnum == number)[0]]
		inp_l[k] = plogg[np.where(pnum == number)[0]]
		inp_e[k] = av[np.where(pnum == number)[0]]
		inp_age[k] = page[np.where(pnum == number)[0]]

		out_t[k] = ts
		out_l[k] = l
		out_e[k] = extinct
		pn = np.where(pnum == number)[0][0]
		try:
			td[k] = ptemp[pn] - stemp[np.where(snum == pn)]
		except:
			td[k] = -1

		# print(number, distance[np.where(pnum == number)[0]])#, inp_t[k] - out_t[k], inp_t[k], out_t[k], td[k], ptemp[pn], stemp[np.where(snum == pn)])
		factor = [1, 0.751, 0.479, 0.282, 0.190, 0.114] #cardelli et al. 1989
		m = sys_mag[number,:]
		mags = [m[n] - (extinct * factor[n]) + 5 - (5 * np.log10(distance[np.where(pnum == number)[0]]/10)) for n in range(len(factor))]
		
		luminosity = fit_lum(mags, ts) #sl[np.where(pnum == number)]#
		# print('lums', luminosity, sl[np.where(pnum == number)])
		avv.append(extinct)#[np.where(cs == min(cs))[0][0]])
		if model == 'baraffe':	
			print('baraffe')
			a = griddata((temps, lum), age, (ts, luminosity))#interp_2d(ts, luminosity, temps, lum, age)
			m = griddata((temps, lum), mass, (ts, luminosity))#interp_2d(ts, luminosity, temps, lum, mass)

			ax.scatter(ts, luminosity, s = 25)

			masses[k] = m
			lums[k] = luminosity
			ages[k] = a

		if model == 'parsec':
			# print('parsec')

			a = griddata((teff, lll), aage, (np.log10(ts), luminosity))#interp_2d(ts, luminosity, teff, lll, aage)
			m = griddata((teff, lll), ma, (np.log10(ts), luminosity))#interp_2d(ts, luminosity, teff, lll, ma)

			# print('teff, l, a', np.log10(ts), luminosity, a)

			masses[k] = m
			lums[k] = luminosity
			ages[k] = a

			if multiplicity[np.where(pnum == number)] > 0:
				if k == 0:
					out = ax.scatter(ts, luminosity, marker = 'x', color = 'xkcd:blue purple', label = 'Output')
				else:
					ax.scatter(ts, luminosity, marker = 'x', color = 'xkcd:blue purple')
			else:
				if k == 0:
					out = ax.scatter(ts, luminosity, s = 20, color = 'xkcd:blue purple', label = 'Output')
				else:
					ax.scatter(ts, luminosity, s = 20, color = 'xkcd:blue purple')
	
	ax.set_xlim(7200, 2500)
	ax.set_ylim(-2, 2)
	ax.set_xlabel(r'T$_{eff}$', fontsize = 13)
	ax.set_ylabel(r'log(L)', fontsize = 13)
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.legend(fontsize = 13, loc = 'best', handles = [inp, out])
	fig.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/isochrone.pdf')
	plt.close()

	t_pct = [abs((out_t[n] - inp_t[n])/inp_t[n]) * 100 for n in range(len(inp_t))]
	l_pct = [abs((lums[n] - sl[n])/sl[n]) * 100 for n in range(len(sl))]
	total_pct = t_pct + l_pct

	colors = cm.plasma(total_pct)

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
	ax.set_xlabel('Input age (Myr)', fontsize = 13)
	ax.set_ylabel('Output age (Myr)', fontsize = 13)
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_plot.pdf')
	plt.close()

	it, ot = inp_t[np.where(td >= 0)], out_t[np.where(td >= 0)]
	it_, ot_ = inp_t[np.where(td < 0)], out_t[np.where(td < 0)]

	it1, it2, ot1, ot2 = it_[np.where(it_ < 4300)], it_[np.where(it_ >= 4300)], ot_[np.where(it_ < 4300)], ot_[np.where(it_ >= 4300)]

	for n in range(len(it)):
		if it[n] - ot[n] > 1400:
			print(pnum[np.where(ptemp == it[n])])

	fig, ax = plt.subplots()#figsize = (8, 5))
	if len(it) > 2:
		a = ax.scatter(it, it-ot, marker = 'x', s = 25, label = 'Binary stars', c = td[np.where(td >= 0)], cmap = plt.cm.plasma)#, norm=colors.SymLogNorm(linthresh = 500, vmin=min(tdiff), vmax=max(tdiff)))
		cbar = plt.colorbar(a)
		cbar.set_label(r'T$_{eff}$ difference (K)')
	if inp_e[0] > 0.1:
		b = ax.scatter(it_, it_ - ot_, marker = '.', s = 25, label = 'Single stars', c = np.array(inp_e)[np.where(td <= 0)])
		cbar = plt.colorbar(b)
		cbar.set_label(r'Extinction (A$_{V}$)')
	else:
		ax.scatter(it_, it_ - ot_, marker = '.', s = 25, label = 'Single stars', color = 'navy')
	ax.plot([min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t))], [0, 0], linestyle = ':', label = 'Zero error')#[min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t))], linestyle=':', label = '1:1')
	ax.axhline(np.mean(it_ - ot_), color = 'orange', zorder= 0, label = r'Single Star Error:\\{:.0f} $\pm$ {:.0f} K (T $<$ 4300)\\{:.0f} $\pm$ {:.0f} K (T $\geq$ 4300)'\
		.format(np.mean(it1 -ot1), np.std(it1 - ot1), np.mean(it2 -ot2), np.std(it2- ot2)))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input Temp (K)', fontsize = 13)
	ax.set_ylabel('Input Temp - Output Temp (K)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/temp_plot.pdf')
	plt.close()	

	fig, ax = plt.subplots()
	a = ax.scatter(inp_l, inp_l-out_l, marker = '.', s = 20, label = 'Primary stars')#, c = tdiff * 10, cmap = plt.cm.plasma, norm=colors.SymLogNorm(linthresh = 500, vmin=min(tdiff), vmax=max(tdiff)))
	#cbar = plt.colorbar(a)
	ax.plot([min(min(inp_l), min(out_l)), max(max(inp_l), max(out_l))], [0, 0], linestyle = ':', label = 'Zero error')#[min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t))], linestyle=':', label = '1:1')
	#ax.plot([min(min(inp_l), min(out_l)), max(max(inp_l), max(out_l))], [np.mean(inp_l - out_l),np.mean(inp_l - out_l)], label = r'Average Error: {:.0f} $\pm$ {:.0f}'.format(np.mean(out_l -inp_l), np.std(out_l - inp_l)))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input log(g)', fontsize = 13)
	ax.set_ylabel('Input log(g) - Output log(g)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/logg_plot.pdf')
	plt.close()	

	ie1, ie2, oe1, oe2 = inp_e[np.where(multiplicity == 1)], inp_e[np.where(multiplicity == 0)], out_e[np.where(multiplicity == 1)], out_e[np.where(multiplicity == 0)]

	fig, ax = plt.subplots()
	a = ax.scatter(ie2, ie2-oe2, marker = '.', s = 20, label = 'Single stars', color = 'navy')#, c = tdiff * 10, cmap = plt.cm.plasma, norm=colors.SymLogNorm(linthresh = 500, vmin=min(tdiff), vmax=max(tdiff)))
	# cbar = plt.colorbar(a)
	if len(ie1) >2:
		ax.scatter(ie1, ie1 - oe1, marker = 'x', s = 20, label = 'Binary stars', color = 'xkcd:sky blue')
	ax.plot([min(min(inp_e), min(out_e)), max(max(inp_e), max(out_e))], [0, 0], linestyle = ':', label = 'Zero error')#[min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t))], linestyle=':', label = '1:1')
	ax.axhline(np.mean(ie2 - oe2), label = r'Average Error: {:.2f} $\pm$ {:.2f}'.format(np.mean(oe2 -ie2), np.std(oe2 -ie2)))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input Extinction (mags)', fontsize = 13)
	ax.set_ylabel('Input Extinction - Output Extinction (mags)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/av_plot.pdf')
	plt.close()	

	fig, ax = plt.subplots()
	ax.hist(inp_t, color = 'navy', label = 'Input T', bins = np.arange(min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t)), 250))
	ax.hist(out_t, color='xkcd:sky blue', alpha = 0.7, label = 'Output T', bins = np.arange(min(min(inp_t), min(out_t)), max(max(inp_t), max(out_t)), 250))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	#ax.set_yscale('symlog')
	ax.set_xlabel('Temp (K)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/temp_hist.pdf')
	plt.close()	

	#from Herczeg and Hillenbrand 2014 (Table 5)
	divs = [2570, 2670, 2770, 2860, 2980, 3190, 3410, 3560, 3720, 3900, 4020, 4210, 4710, 4870, 5180, 5430, 5690, 5930, 6130, 6600]
	labels = ['M9', 'M8', 'M7', 'M6', 'M5', 'M4', 'M3', 'M2', 'M1', 'M0', 'K7', 'K5', 'K2', 'K0', 'G8', 'G5', 'G2', 'G0', 'F8', 'F5']
	fig, ax = plt.subplots()
	ax.hist(inp_t, bins = divs, label = "Input SpT", color = 'navy')
	ax.hist(out_t, bins = divs, label = "Fitted SpT", color = 'xkcd:sky blue', alpha = 0.7)
	ax.set_xticks(divs[::2])
	ax.set_xticklabels(labels[::2], fontsize=14)
	ax.set_yscale('log')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Stellar Spectral Type', fontsize = 13)
	ax.set_ylabel('Number', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/spt_hist.pdf')
	plt.close()

	for n, a in enumerate(ages): 
		if np.isnan(a):
			ages[n] = -99
	age_write = ages

	iab, oab = inp_age[np.where(td >= 0)], ages[np.where(td >= 0)]
	ias, oas = inp_age[np.where(td < 0)], ages[np.where(td < 0)]

	iab, oab, ias, oas = iab[np.where(oab > -99)], oab[np.where(oab > -99)], ias[np.where(oas > -99)], oas[np.where(oas > -99)]

	fig, ax = plt.subplots()
	ax.scatter(ias, ias - oas, marker = '.', s = 20, color = 'navy', label = 'Primary stars')
	if len(iab) > 2:
		a = ax.scatter(iab, iab-oab, marker = 'x', s = 25, label = 'Binary stars')#, c = td[np.where(td >= 0)], cmap = plt.cm.plasma)#, norm=colors.SymLogNorm(linthresh = 500, vmin=min(tdiff), vmax=max(tdiff)))
		ax.axhline(np.mean(iab-oab), color = 'xkcd:red orange', label = r'Average binary star error: {:.2f} $\pm$ {:.2f} Myr'.format(np.mean(iab-oab), np.std(iab-oab)))
	# cbar = plt.colorbar(a)
	ax.plot([min(min(inp_age), min(ages)), max(max(inp_age), max(ages))], [0, 0], linestyle=':', label = 'Zero error')
	ax.axhline(np.mean(ias - oas), label = r'Average single star error: {:.2f} $\pm$ {:.2f} Myr'.format(np.mean(ias - oas), np.std(ias - oas)), color = 'orange')
	plt.minorticks_on()
	ax.set_xlim(min(inp_age) - 0.25, max(inp_age) + 0.25)
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input age (Myr)', fontsize = 13)
	ax.set_ylabel('Input age - Output age (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_diff_plot.pdf')
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
	ax.set_xlabel('Input age (Myr)', fontsize = 13)
	ax.set_ylabel('Output age (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_plot2.pdf')
	plt.close()

	agesn = ages[~np.isnan(ages)]
	fig, ax = plt.subplots()
	try:
		ax.hist(inp_age, color = 'navy', label = 'Input ages', bins = np.arange(min(inp_age), max(agesn), 0.25))
		ax.hist(agesn, color = 'xkcd:sky blue', alpha = 0.6, label = 'Output ages', bins = np.arange(min(inp_age), max(agesn), 0.25))
	except:
		ax.hist(inp_age, color = 'navy', label = 'Input ages', bins = np.arange(inp_age, max(agesn), 0.25))
		ax.hist(agesn, color = 'xkcd:sky blue', alpha = 0.6, label = 'Output ages', bins = np.arange(inp_age, max(agesn), 0.25))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.set_xlim(min(inp_age)-0.25, max(agesn)+0.25)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Input and Output Age (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + runfolder + '/age_hist.pdf')
	plt.close()

	np.savetxt(runfolder + '/results/mass_fit_results.txt', np.column_stack((num, age_write, masses, lums, out_t, av)), header= '#number, age (myr), Mass, Luminosity, fitted temperature, extinction')
	return masses, lums

def plot_specs(num, run):
	cwd = os.getcwd()
	wl, spec = np.genfromtxt(run + '/specs/spec_{}.txt'.format(num), unpack = True)

	ewl = np.linspace(wl[0], wl[-1], len(wl))
	inte = interp1d(wl, spec)
	espec = inte(ewl)

	p_num, p_mass, mul, p_age, p_av, p_temp, p_logg, p_luminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = np.genfromtxt(run + '/ppar.txt', unpack = True)
	pav, pnum, multiplicity, p_temp, p_logg = p_av[np.where(p_num == num)[0]], \
		int(p_num[np.where(p_num == num)[0]]), mul[np.where(p_num == num)[0]], p_temp[np.where(p_num == num)[0]], p_logg[np.where(p_num == num)[0]]

	numb, xs, temp, lg, norm, ext = np.genfromtxt(cwd + '/' + run+'/results/parvals_{}.txt'.format(num), unpack = True)
	# if numb[-1] == 119:
	idx = np.where(xs == min(xs))[0][0]

	t, l, n, extinct = temp[idx], lg[idx], norm[idx], ext[idx]

	#t, l, n, extinct = np.genfromtxt(cwd + '/' + run+'/results/{}_simplex.txt'.format(num), unpack = True)

	w_, s_ = mft.get_spec(t, l, [0.45, 1], normalize = False)

	# resid = [s_[n] - espec[n] for n in range(len(s_))]
	# ww, resid = redres(ewl, resid, 6)
	# s1 = [s * n for s in s_]

	s_ = mft.extinct(np.asarray(w_), np.asarray(s_), extinct)

	wl, w_, spec, s_ = np.array(wl), np.array(w_), np.array(spec), np.array(s_)

	# norm = np.mean(s_[np.where((w_ >= 6000) & (w_ <= 9000))])/np.mean(spec[np.where((wl >= 6000) & (wl <= 9000))])
	# print(n)
	s_ = np.array([ss * n for ss in s_])

	fig1, ax1 = plt.subplots()#(nrows = 2, sharex = True)

	if int(multiplicity) == 0:
		if not type(p_temp) is float:
			p_temp = p_temp[0]
			p_logg = p_logg[0]

		ax1.plot(wl[np.where((wl <= 9000) & (wl >= 6000))], spec[np.where((wl <= 9000) & (wl >= 6000))], color = 'navy', \
			label = 'Input: T = {:.0f}, log(g) = {:.1f}, extinction = {:.1f}'.format(p_temp, p_logg, float(pav)))
	else:
		s_num, s_mass, s_sep, s_age, eccentricity, period, s_temp, s_logg, s_luminosity, svmag, srmag, simag, sjmag, shmag, skmag = np.genfromtxt(run + '/spar.txt', unpack = True)
		if np.size(s_num) > 1:
			sn = np.where(s_num == pnum)[0]
			s_temp, s_logg = s_temp[sn], s_logg[sn]

		if not type(s_temp) is float:
			s_temp = s_temp[0]
			s_logg = s_logg[0]

		ax1.plot(wl[np.where((wl <= 9000) & (wl >= 6000))], spec[np.where((wl <= 9000) & (wl >= 6000))], color = 'navy', \
			label = 'Input: T1 = {:.0f}, T2 = {:.0f}, \nlog(g)1 = {:.1f}, log(g)2 = {:.1f} \nextinction = {:.1f}'\
			.format(float(p_temp), float(s_temp), float(p_logg), float(s_logg), float(pav)))
		# pri_wl, pri_spec = mft.get_spec(p_temp, p_logg, [0.45, 2.5], normalize = False)
		# sec_wl, sec_spec = mft.get_spec(s_temp, s_logg, [0.45, 2.5], normalize = False)

		# pri_wl, pri_spec = redres(pri_wl, pri_spec, 6)
		# sec_wl, sec_spec = redres(sec_wl, sec_spec, 6)

		# ax1.plot(pri_wl, pri_spec, label = 'primary')
		# ax1.plot(sec_wl, sec_spec, label = 'secondary')



	ax1.plot(w_[np.where((w_ >= 6000) & (w_ <= 9000))], s_[np.where((w_ >= 6000) & (w_ <= 9000))], color='xkcd:sky blue', \
		label = 'Best fit model: \nT = {:.0f}, log(g) = {:.1f}, \nChi sq = {:.2f}, extinction = {:.1f}'.format(float(t), float(l), xs[idx], float(extinct)), linestyle= '-')
	# ax1.plot(w_, s1, color = 'xkcd:red orange', label = 'Unextincted input spectrum')
	# ax2.plot(w_[15:-15], resid[15:-15], label = 'residual')
	plt.minorticks_on()
	ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax1.tick_params(bottom=True, top =True, left=True, right=True)
	ax1.tick_params(which='both', labelsize = "large", direction='in')
	ax1.tick_params('both', length=8, width=1.5, which='major')
	ax1.tick_params('both', length=4, width=1, which='minor')
	ax1.set_xlabel(r'$\lambda$ (\AA)', fontsize = 13)
	ax1.set_ylabel(r'$L_{\lambda}$', fontsize = 13)
	ax1.set_xlim(6000, 9000)
	# ax1.set_ylim(min(s_[15:-15]) - (0.001 * min(s_[15:-15])), max(s_[15:-15]) + (0.005 * max(s_[15:-15])))
	ax1.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(run + '/plot_spec_{}.pdf'.format(num))
	plt.close(fig1)

	# fig2, ax2 = plt.subplots()
	# if multiplicity == 0:
	# 	ax2.plot(wl, spec, color = 'navy', label = 'Input: T = {:.0f}, log(g) = {:.1f}'.format(p_temp, p_logg))
	# else:
	# 	ax2.plot(wl, spec, color = 'navy', label = 'Input: T1 = {:.0f}, T2 = {:.0f}, \nlog(g)1 = {:.1f}, log(g)2 = {:.1f}'\
	# 		.format(float(p_temp), float(s_temp), float(p_logg), float(s_logg)))
	# ax2.plot(w_, s_, color='xkcd:sky blue', label = 'Best fit model: \nT = {:.0f}, log(g) = {:.1f}, \nChi sq = {:.2f}, extinction = {:.1f}'\
	# 	.format(float(t), float(l), xs[idx], float(extinct)), linestyle= '-')
	# plt.minorticks_on()
	# ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	# ax2.tick_params(bottom=True, top =True, left=True, right=True)
	# ax2.tick_params(which='both', labelsize = "large", direction='in')
	# ax2.tick_params('both', length=8, width=1.5, which='major')
	# ax2.tick_params('both', length=4, width=1, which='minor')
	# ax2.set_xlabel(r'$\lambda$ (\AA)', fontsize = 13)
	# ax2.set_ylabel(r'$L_{\lambda}$', fontsize = 13)
	# ax2.set_xlim(6850, 7250)
	# ax2.legend(fontsize = 13, loc = 'best')
	# plt.tight_layout()
	# plt.savefig(run + '/plot_spec_TiO_{}.pdf'.format(num))
	# plt.close(fig2)

	# fig3, ax3 = plt.subplots()
	# if multiplicity == 0:
	# 	ax3.plot(wl, spec, color = 'navy', label = 'Input: T = {:.0f}, log(g) = {:.1f}'.format(p_temp, p_logg))
	# else:
	# 	ax2.plot(wl, spec, color = 'navy', label = 'Input')#: T1 = {:.0f}, T2 = {:.0f}, \nlog(g)1 = {:.1f}, log(g)2 = {:.1f}'.format(float(p_temp), float(s_temp), float(p_logg), float(s_logg)))
	# ax3.plot(w_, s_, color='xkcd:sky blue', label = 'Best fit model')#: \nT = {:.0f}, log(g) = {:.1f}, \nChi sq = {}'\
	#	.format(float(t[0]), float(l[0]), float((str(xs[idx]).split('.')[0].split('[')[-1])[0])), linestyle= ':')
	# plt.minorticks_on()
	# ax3.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	# ax3.tick_params(bottom=True, top =True, left=True, right=True)
	# ax3.tick_params(which='both', labelsize = "large", direction='in')
	# ax3.tick_params('both', length=8, width=1.5, which='major')
	# ax3.tick_params('both', length=4, width=1, which='minor')
	# ax3.set_xlabel(r'$\lambda$ (\AA)', fontsize = 13)
	# ax3.set_ylabel(r'$L_{\lambda}$', fontsize = 13)
	# ax3.set_xlim(8400, 8950)
	# ax3.legend(fontsize = 13, loc = 'best')
	# plt.tight_layout()
	# plt.savefig(run + '/plot_spec_CaIR_{}.pdf'.format(num))
	# plt.close(fig3)

	return 

def run_pop(nsys, run, new_pop = False):

	cwd = os.getcwd()

	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	if new_pop == True:

		bf = [1, 1, 1,1]#[0.26, 0.4, 0.45, 0.5]
		md = [0.5, 0.7, 1.5]

		if rank == 0:
			if not os.path.exists(cwd + '/' + run):
				os.mkdir(cwd + '/' + run)
			else:
				pass
			if not os.path.exists(cwd + '/' + run + '/specs'):
				os.mkdir(cwd + '/' + run + '/specs')
			else:
				pass
			if not os.path.exists(cwd + '/' + run + '/results'):
				os.mkdir(cwd + '/' + run + '/results')
			else:
				pass

		else:
			pass

		print('Making systems')

		age_range = [1.01, 3.01]

		ns = np.arange(0, nsys)

		s = comm.scatter(ns, root = 0)
		aa = 0.1#5 * np.random.rand(nsys)

		pp, sp = make_binary_sys(s, 1, bf, md, age_range, aa, run)

		out = comm.gather((pp, sp), root = 0)

		# pp, sp = np.asarray(oc).T
		# print('pp and sp', pp, sp)

		ps = ''
		for n in range(len(pp)):
			ps = ps + str(pp[n]) + ' '
		ss = ''
		for n in range(len(sp)):
			ss = ss + str(sp[n]) + ' '

		ppar = open(cwd + '/' + run + '/ppar.txt', 'a')
		spar = open(cwd + '/' + run +'/spar.txt', 'a')

		ppar.write(ps + '\n')

		try:
			spar.write(ss + '\n')
		except:
			pass
		ppar.close()
		spar.close()

		# pool = mp.Pool()
		# results = [pool.apply_async(make_binary_sys, args = (ns, 1, bf, md, age_range, 0, run)) for ns in range(nsys)]
		# print('retrieving results')
		# out = [p.get() for p in results]

		# out1, out2 = np.array(out).T

		# ppar = [out1[n][:][1] for n in range(nsys)]
		# np.savetxt(cwd + '/' + run + '/ppar.txt', ppar)

		# nbin = 0
		# wh = []
		# for k, n in enumerate(ppar):
		# 	nbin += int(n[2])
		# 	if int(n[2]) == 1:
		# 		wh.append(k)
		# if nbin > 0:
		# 	spar = [out2[w][:][1] for w in wh]
		# 	np.savetxt(cwd + '/' + run + '/spar.txt', spar)
		# else:
		# 	pass;

	print('Fitting now')
	n, ex_g, t_g, lg_g = np.genfromtxt(cwd +'/' + run + '/ppar.txt', usecols = (0,4,5,6), unpack = True, autostrip = True)

	# if type(lg_g) is not np.ndarray:
	# 	ls = np.round(lg_g/0.01) * 0.01
	# 	trange = np.arange(((np.round(t_g / 100) * 100) - 300), ((np.round(t_g / 100) * 100) + 700), 100)
	
	# else:
	# 	ls = [np.round(l / 0.01) * 0.01 for l in lg_g]
	# 	trange = [np.arange(((np.round(t / 100) * 100) - 300), ((np.round(t / 100) * 100) + 700), 100) for t in t_g]
	# lgrange = np.arange(3.5, 4.5, 0.5)
	
	files = np.array([run + '/specs/spec_{}.txt'.format(n) for n in range(nsys)])
	# # for n in range(nsys):
	# # 	#simplex_fit(t_g[n], ls[n], 1e18, 0, files[n])
	# # 	fit_test(t_g[n], lg_g[n], 5e23, 0, files[n])

	# if rank == 0:
	# 	ns = np.arange(0, nsys)
	# else:
	# 	ns = None

	# s = comm.scatter(ns, root = 0)

	if type(lg_g) is not np.ndarray:
		tg, lg, norm, ex, fi = t_g, lg_g, 5e23, ex_g, files[rank]
	else:
		tg, lg, norm, ex, fi = t_g[np.where(n == rank)[0][0]], lg_g[np.where(n == rank)[0][0]], 5e23, ex_g[np.where(n == rank)[0][0]], files[rank]

	outcs = fit_test(tg, lg, norm, ex, fi)

	ga = comm.gather(outcs, root = 0)

	
	print('done!')

	return

def plot_init_pars(run, pri_num, sec_num, pri_mass, sec_mass, sep, av, distance):
	mr = []
	sys_mass = []
	for n in pri_num:
		if n in sec_num:
			mr.append(float(sec_mass[np.where(sec_num == n)]/pri_mass[np.where(pri_num == n)][0]))
			sys_mass.append((pri_mass[np.where(pri_num == n)][0] + sec_mass[np.where(sec_num == n)[0]])[0])
		else:
			sys_mass.append(pri_mass[np.where(pri_num == n)][0])

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

	nmr = np.linspace(min(sys_mass), max(sys_mass), len(sys_mass) * 100)
	cm = make_chabrier_imf(nmr)
	primary_numbers = [int(np.rint(len(pri_num) * p)) * 2 for p in cm]

	fig, ax = plt.subplots()
	ax.hist(sys_mass, bins = np.logspace(np.log10(min(sys_mass)), np.log10(max(sys_mass)), 15), label = 'Single + binary stars', color = 'b')
	ax.hist(pri_mass, bins = np.logspace(np.log10(min(pri_mass)), np.log10(max(pri_mass)), 15), label = 'Single stars', facecolor = 'cyan', alpha = 0.7)
	#ax.hist(smass, bins = np.logspace(np.log10(min(smass)), np.log10(max(smass)), 20), label = 'Secondary', alpha = 0.5, color = 'teal')
	ax.plot(nmr, primary_numbers, label = 'Theoretical IMF', linestyle = '--', color = 'tab:orange')
	plt.axvline(x = np.median(pri_mass), label = 'Median stellar mass', color='red')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel(r'Mass ($M_{\odot}$)', fontsize = 13)
	ax.set_ylabel('Number of stars', fontsize = 13)
	ax.set_xlim(0.08, 2.1)
	#ax.set_ylim(1e2, 1e3)
	#ax.set_yscale('log')
	ax.set_xscale('log')
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/masshist_init.pdf')
	plt.close(fig)
	return

def final_analysis(nsys, run, plots = False):
	print('Plotting!')

	if plots == True:
		[plot_specs(n, run) for n in np.random.randint(low = 0, high = nsys, size = 15)]

	print('Running mass analysis')

	analyze_sys(run)

	num, age, fit_mass, logg, fit_t, extinct = np.genfromtxt(run + '/results/mass_fit_results.txt', unpack = True)

	n, mass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = np.genfromtxt(run + '/ppar.txt', unpack = True)

	n, mass, multiplicity, page, av, ptemp, plogg, pluminosity, distance, pvmag, prmag, pimag, pjmag, phmag, pkmag = n[np.argsort(n)], mass[np.argsort(n)], \
		multiplicity[np.argsort(n)], page[np.argsort(n)], av[np.argsort(n)], ptemp[np.argsort(n)], plogg[np.argsort(n)], pluminosity[np.argsort(n)], \
		distance[np.argsort(n)], pvmag[np.argsort(n)], prmag[np.argsort(n)], pimag[np.argsort(n)], pjmag[np.argsort(n)], phmag[np.argsort(n)], pkmag[np.argsort(n)]

	sn, smass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = np.genfromtxt(run + '/spar.txt', unpack = True)

	sn, smass, sep, sage, seccentricity, period, stemp, slogg, sluminosity, svmag, srmag, simag, sjmag, shmag, skmag = sn[np.argsort(sn)], smass[np.argsort(sn)], \
		sep[np.argsort(sn)], sage[np.argsort(sn)], seccentricity[np.argsort(sn)], period[np.argsort(sn)], stemp[np.argsort(sn)], slogg[np.argsort(sn)], \
		sluminosity[np.argsort(sn)], svmag[np.argsort(sn)], srmag[np.argsort(sn)], simag[np.argsort(sn)], sjmag[np.argsort(sn)], shmag[np.argsort(sn)], skmag[np.argsort(sn)]

	sluminosity = np.asarray([np.log10(10**(-0.4 * s)) for s in sluminosity])

	fig, ax = plt.subplots()
	ax.plot(ptemp, pluminosity, 'v', color = 'navy', label = 'Primary stars')
	ax.plot(stemp, sluminosity, 'o', color='xkcd:sky blue', label = 'Secondary stars')
	ax.set_xlabel('Temperature (K)', fontsize = 13)
	ax.set_ylabel('log10(luminosity) (solar lum.)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	ax.invert_xaxis()
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/' + run + '/hrd_input.pdf')
	plt.close()

	teff, ext = [], []
	for k in n:
		try:
			idx = np.where(num == int(k))[0][0]
			teff.append(fit_t[idx])
			ext.append(float(extinct[idx]))
		except:
			pass

	# plot_init_pars(run, n, sn, mass, smass, sep, extinct, distance)

	sys_lum = []

	mass_resid = [] 
	fm = []
	mm = []
	for k in n: 
		try:
			idx = np.where(num == int(k))
			m = mass[int(k)]
			l = pluminosity[int(k)]
			# if int(k) in sn:
			# 	m += smass[np.where(sn == int(k))][0]
			# 	l += sluminosity[np.where(sn == int(k))][0]
			test = m - fit_mass[idx] 
			fm.append(fit_mass[idx][0])
			mm.append(m)
			sys_lum.append(l)
			mass_resid.append(test[0]) 
		except:
			pass

	sys_flux = [((10**sys_lum[n]) * 3.9e33) /(4 * np.pi * (distance[n] * 3.086e18)**2) for n in range(len(num))]
	sys_mag_app = [-2.5 * np.log10(sf/17180) for sf in sys_flux]

	mr = [mass_resid[n]/mm[n] for n in range(len(mass_resid))]


	plt.figure(1)
	for k in n:
		k = int(k)
		try:
			# plt.plot([ptemp[k], teff[k]], [av[k], ext[k]], color = 'k', alpha = 0.5)
			if k == 1:
				# plt.scatter(ptemp[k], av[k], color = 'navy', label = 'input')
				plt.scatter(teff[k], ext[k], color = 'xkcd:sky blue', label = 'output')
			else:
				# plt.scatter(ptemp[k], av[k], color = 'navy')
				plt.scatter(teff[k], ext[k], color = 'xkcd:sky blue')
		except:
			pass
	plt.minorticks_on()
	plt.legend(fontsize = 13, loc = 'best')
	plt.xlabel(r'T$_{eff}$ (K)', fontsize = 13)
	plt.ylabel(r'A$_{V}$ (mag)', fontsize = 13)
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/temp_av.pdf')
	plt.close()


	chab = make_chabrier_imf(np.linspace(min(mm), max(mm), 300))
	plt.figure(1)
	plt.hist(np.log10(mm), label = 'input', color = 'navy', bins = np.linspace(min(np.log10(mm)), max(np.log10(mm)), 7), log=True)
	plt.hist(np.log10(fit_mass), color = 'xkcd:sky blue', label = 'output', alpha = 0.6, bins = np.linspace(min(np.log10(mm)), max(np.log10(mm)), 7), log=True) 
	# plt.plot(np.linspace(min(mm), max(mm), 300), chab * 1200, linestyle = '-', color = 'xkcd:light red')
	plt.legend(fontsize = 13)
	# plt.gca().invert_xaxis()
	plt.minorticks_on()
	plt.xlabel(r'Mass (M$_{\odot}$)', fontsize = 13)
	plt.ylabel('Number', fontsize = 13)
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/masshist.pdf')
	plt.close()

	plt.figure(2)
	plt.hist(mass_resid, color='navy')	
	plt.xlabel(r'Mass residual (M$_{\odot}$)', fontsize = 13)
	plt.ylabel('Number', fontsize = 13)
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/error_fit.pdf')
	plt.close()

	plt.figure(3)
	plt.hist(mr, color = 'navy', label = 'Avg. frac. error = {:.2f}\n 1 stdev = {:.2f}'.format(float(np.mean(mr)), float(np.std(mr))))
	plt.legend(fontsize = 13) 
	plt.xlabel('Fractional mass residual', fontsize = 13)
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/error_fit_fractional.pdf')
	plt.close()

	one_one = np.arange(0, 4)
	plt.figure(4)
	plt.scatter(mm, fm)
	plt.plot(one_one, one_one, ':', label = 'One-to-one correspondence line')
	plt.xlabel(r'Expected mass (M$_{\odot}$)', fontsize =14)
	plt.ylabel(r'Fitted mass (M$_{\odot}$)', fontsize =14)
	plt.legend(fontsize = 13, loc = 'best')
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/mass_scatter.pdf')
	plt.close()

	plt.figure(5)
	plt.scatter(fit_t, sys_lum)
	plt.xlabel('Fitted Temperature (K)', fontsize =14)
	plt.gca().invert_xaxis()
	plt.ylabel(r'log(System Luminosities) (L$_{\odot}$)', fontsize = 13)
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/HRD_output.pdf')
	plt.close()

	plt.figure(6)
	plt.scatter(fit_t, sys_mag_app)
	plt.xlabel('Fitted Temperature', fontsize = 13)
	plt.ylabel(r'System apparent bolometric magnitude', fontsize = 13)
	plt.gca().invert_yaxis()
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/HRD_output_flux.pdf')
	plt.close()

	md = [mm[n] - fm[n] for n in range(len(mm))]
	mm, md = np.array(mm), np.array(md)
	md1 = md[np.where(multiplicity == 0)]
	mm1 = mm[np.where(multiplicity == 0)]
	md2 = md[np.where(multiplicity == 1)]
	mm2 = mm[np.where(multiplicity == 1)]
	mdiff = []
	mass, smass = np.array(mass), np.array(smass)
	for number in sn:
		pri = mass[np.where(n == number)[0][0]]
		mdiff.append(float(smass[np.where(sn == number)[0][0]])/pri)

	d1, d2 = md1[np.where(mm1 < 0.95)], md1[np.where(mm1 >= 0.95)]

	plt.figure(7)
	plt.scatter(mm1, md1, color='navy', s = 20, marker = '.', label = r'Single Star Error:\\{:.2f} $\pm$ {:.2f} (T $<$ 4300 K)\\{:.2f} $\pm$ {:.2f} (T $\geq$ 4300 K)'\
		.format(np.nanmean(d1), np.nanstd(d1), np.nanmean(d2), np.nanstd(d2)))
	if len(mm2) > 2:
		a = plt.scatter(mm2, md2, c = np.array(mdiff), cmap = plt.cm.plasma, marker = 'x', label = 'Binary stars')
		cbar = plt.colorbar(a)
		cbar.set_label(r'Mass ratio')
	plt.plot((min(mm), max(mm)), (0,0), ':', label = 'Zero error line')
	plt.xlabel(r'Input mass (M$_{\odot}$)', fontsize = 13)
	plt.ylabel('Input mass - fitted mass', fontsize = 13)
	plt.legend(fontsize = 13, loc = 'best')
	plt.minorticks_on()
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.tight_layout()
	plt.savefig(run + '/mass_diff.pdf')
	plt.close()

	return

def example_sys(teff, logg, norm, extinct, filename):

	logg, logg_try = 4, 4

	wl, spec = np.genfromtxt(filename, unpack = True)

	teff += np.random.normal(scale = 100)
	
	ww_init, ss_init = mft.get_spec(teff, logg, [0.45, 0.9], normalize = False)

	second_wl = np.linspace(max(wl[0], ww_init[0]), min(wl[-1], ww_init[-1]), len(spec))

	dd2 = interp1d(wl, spec)
	md2_init = interp1d(ww_init, ss_init)
	spec2 = dd2(second_wl)
	ss_init = md2_init(second_wl)

	extinct += np.random.normal(scale = 0.1)
	while extinct < 0:
		extinct += np.random.rand()/2

	ss_init = mft.extinct(second_wl, ss_init, extinct)
	normalization = np.mean(spec2)/np.mean(ss_init)

	normalization += np.random.normal(scale = np.abs(normalization * 0.02))

	ss_init = [s * normalization for s in ss_init]
	vary = [sp * 0.01 for sp in spec2]
	cs_init = mft.chisq(ss_init, spec2, vary)

	niter = 0
	total_iter = 0

	plt.figure()
	plt.plot(second_wl, ss_init, label = 'data')
	plt.plot(second_wl, spec2, label = 'test')
	plt.savefig('test/fit_test.png')
	plt.close()

	f = open('test/parvals.txt', 'a')
	f.write('{} {} {} {} {} {}\n'.format(niter, np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
	f.close()

	#everything block
	while niter < 50:
		rn = np.random.randint(0, 3)
		rn2 = np.random.rand()

		parvar = np.asarray([teff, normalization, extinct])

		if niter < 10:
			var = np.asarray([np.random.normal(scale = 100), np.random.normal(scale=np.abs(normalization*0.05)), np.random.normal(scale = 0.1)])[rn]
			while rn == 2 and parvar[rn] + var < 0:
				var = np.random.normal(scale = 0.5)

		else:
			var = np.asarray([np.random.normal(scale = 10), np.random.normal(scale=np.abs(normalization*0.02)), np.random.normal(scale = 0.05)])[rn]
			while rn == 2 and parvar[rn] + var < 0:
				var = np.random.normal(scale = 0.5)

		parvar[rn] = parvar[rn] + var

		teff_try, normalization_try, extinct_try = parvar

		if teff_try < 7000 and teff_try > 2000:

			ww_i, ss_i = mft.get_spec(teff_try, logg, [0.45, 0.9], normalize = False)

			mi2_init = interp1d(ww_i, ss_i)
			sse_i = mi2_init(second_wl)

			sse_i = mft.extinct(second_wl, sse_i, extinct_try)

			sse_i = [s * normalization_try for s in sse_i]
			vv = [0.01 * s for s in spec2]

			plt.figure()
			plt.plot(second_wl, sse_i, label = 'test, {}'.format(teff_try))
			plt.plot(second_wl, spec2, label = 'data')
			plt.plot(second_wl, vv, label = 'error')
			plt.legend()
			plt.savefig('test/fit_test_{}.png'.format(total_iter))
			plt.close()

			cs_try = [((sse_i[n] - spec2[n])**2)/vv[n]**2 for n in range(len(sse_i))]#returns a chi^2 vector

			print('try, init', np.sum(cs_try)/len(cs_try), np.sum(cs_init)/len(cs_init))

			if np.sum(cs_try)/len(cs_try) < np.sum(cs_init)/len(cs_init): #uses reduced chi^2 as my statistic

				teff = teff_try
				cs_init = cs_try
				if niter >= 20:
					niter = 20
				else:
					niter = 0
			else:
				niter += 1

			f = open('test/parvals.txt', 'a')
			f.write('{} {} {} {} {} {}\n'.format(niter, np.sum(cs_init)/len(cs_init), teff, logg, normalization, extinct))
			f.close()

			f = open('test/testpars.txt', 'a')
			f.write('{} {} {} {} {} {}\n'.format(rn, np.sum(cs_try)/len(cs_try), teff_try, logg_try, normalization_try, extinct_try))
			f.close()
			total_iter += 1

		else:
			teff_try, normalization_try, extinct_try = teff_try + np.random.normal(scale = 200),\
				normalization_try + np.random.normal(scale = np.abs(normalization_try * 0.1)), extinct_try + np.random.normal(scale = 0.1)
	return np.sum(cs_try)/len(cs_try)

def compare_two_runs(run1, run2):
	number1, age1, mass1, lum1, temp1, extinct1 = np.genfromtxt(run1 + '/results/mass_fit_results.txt', unpack = True) 
	number2, age2, mass2, lum2, temp2, extinct2 = np.genfromtxt(run2 + '/results/mass_fit_results.txt', unpack = True)
	print('Median mass, run 1: ', np.median(mass1), '+/-', np.std(mass1))
	print('Median mass, run 2: ', np.median(mass2), '+/-', np.std(mass2))
	fig, ax = plt.subplots()
	ax.hist(mass1, color = 'navy', label = 'Single mass fit', bins = np.logspace(np.log10(min(min(mass1), min(mass2))), np.log10(max(max(mass1), max(mass2))), 15), log = True)
	ax.hist(mass2, color = 'xkcd:sky blue', alpha = 0.6, label = 'Binary mass fit', bins = np.logspace(np.log10(min(min(mass1), min(mass2))), np.log10(max(max(mass1), max(mass2))), 15), log = True)
	ax.axvline(np.median(mass1), label = 'Single median mass', color='navy')
	ax.axvline(np.median(mass2), label = 'Binary median mass', color = 'xkcd:sky blue')
	ax.set_xscale('log')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel(r'Single and Binary Masses (M$_{\odot}$)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/comp_mass_{}_{}.pdf'.format(run1, run2))
	plt.close()

	print('average age, run 1: ', np.mean(age1), '+/-', np.std(age1))
	print('average age, run 2: ', np.mean(age2), '+/-', np.std(age2))
	fig, ax = plt.subplots()
	ax.hist(age1, color = 'navy', label = 'Single age fit', bins = np.linspace(min(min(age1), min(age2)), max(max(age1), max(age2)), 10))
	ax.hist(age2, color = 'xkcd:sky blue', alpha = 0.6, label = 'Binary age fit', bins = np.linspace(min(min(age1), min(age2)), max(max(age1), max(age2)), 10))
	ax.axvline(np.mean(age1), label = 'Single mean age', color='navy')
	ax.axvline(np.mean(age2), label = 'Binary mean age', color = 'xkcd:sky blue')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Single and Binary Ages (Myr)', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/comp_age_{}_{}.pdf'.format(run1, run2))
	plt.close()

	#divisions from
	divs = [2570, 2670, 2770, 2860, 2980, 3190, 3410, 3560, 3720, 3900, 4020, 4210, 4710, 4870, 5180, 5430, 5690, 5930, 6130, 6600]
	labels = ['M9', 'M8', 'M7', 'M6', 'M5', 'M4', 'M3', 'M2', 'M1', 'M0', 'K7', 'K5', 'K2', 'K0', 'G8', 'G5', 'G2', 'G0', 'F8', 'F5']
	fig, ax = plt.subplots()
	ax.hist(temp1, bins = divs, label = "Single SpTs", color = 'navy')
	ax.hist(temp2, bins = divs, label = "Binary SpTs", color = 'xkcd:sky blue', alpha = 0.7)
	ax.set_xticks(divs[::2])
	ax.set_xticklabels(labels[::2], fontsize=14)
	#ax.set_yscale('log')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Single and Binary spectral type dist.', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(os.getcwd() + '/comp_spt_{}_{}.pdf'.format(run1, run2))
	plt.close()

# plot_specs(0, 'run20')
# final_analysis(48, 'run21', plots = True)
# final_analysis(144, 'run11', plots = False)
# final_analysis(240, 'run5', plots = False)
# final_analysis(144, 'run7', plots = True)
# final_analysis(144, 'run24', plots = True)
# final_analysis(480, 'run1', plots = False)
# final_analysis(480, 'run2', plots = False)
# final_analysis(480, 'run4', plots = False)
final_analysis(480, 'run5', plots = False)


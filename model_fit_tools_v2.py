#Kendall Sullivan

#EMCEE VERSION OF MCMC CODE

#TO DO: Write add_disk function, disk/dust to possible fit params

#20190522: Added extinction with a fixed value to model fitting (prior to fit), updated models to theoretical PHOENIX BT-SETTL models with the 
#CFIST line list downloaded from the Spanish Virtual Observatory "Theoretical Tools" resource. 
#20190901 (ish) Updated model spectra to be the PHOENIX BT-SETTL models with the CFIST 2011 line list downloaded from France Allard's website
#and modified from the FORTRAN format into more modern standard text file format before using
#20200514 Commented everything that's out of beta/active development (or at the very least mostly done) - that goes through the end of get_spec
#the remaining content probably needs to be pared down and likely should be troubleshot fairly carefully

"""
.. module:: model_fit_tools_v2
   :platform: Unix, Mac
   :synopsis: Large package with various spectral synthesis and utility tools.

.. moduleauthor:: Kendall Sullivan <kendallsullivan@utexas.edu>

Dependencies: numpy, pysynphot, matplotlib, astropy, scipy, PyAstronomy, emcee, corner, extinction.
"""

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
import extinction
import time

def update_progress(progress):
	"""Displays or updates a console progress bar

	Args:
		Progress (float): Accepts a float between 0 and 1. Any int will be converted to a float.

	Note:
		A value under 0 represents a 'halt'.
		A value at 1 or bigger represents 100%

	"""
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
	lam_corr = []
	for w in wl:
		lam_corr.append(w * (1. + (bcvel - radvel)/3e5))
	return lam_corr

def extinct(wl, spec, av, rv = 3.1, unit = 'aa'):
	"""Uses the package "extinction" to calculate an extinction curve for the given A_v and R_v, 
	then converts the extinction curve to a transmission curve
	and uses that to correct the spectrum appropriately.
	Accepted units are angstroms ('aa', default) or microns^-1 ('invum').

	Args:
		wl (list): wavelength array
		spec (list): flux array
		av (float): extinction in magnitudes
		rv (float): Preferred R_V, defaults to 3.1
		unit (string): Unit to use. Accepts angstroms "aa" or inverse microns "invum". Defaults to angstroms.

	Returns:
		spec (list): a corrected spectrum vwith no wavelength vector. 

	"""
	ext_mag = extinction.fm07(wl, av, unit)
	spec = extinction.apply(ext_mag, spec)
	return np.array(spec)
	
def plots(wave, flux, l, lw=1, labels=True, xscale='log', yscale='log', save=False):
	"""makes a basic plot - input a list of wave and flux arrays, and a label array for the legend.
	If you want to label your axes, set labels=True and enter them interactively.
	You can also set xscale and yscale to what you want, and set it to save if you'd like.
	Natively creates a log-log plot with labels but doesn't save it.
	
	Args:
		wave (list): wavelength array
		flux (list): flux array
		l (list): array of string names for legend labels.
		lw (float): linewidths for plot. Default is 1.
		labels (boolean): Toggle axis labels. Initiates interactive labeling. Defaults to True.
		xscale (string): Set x axis scale. Any matplotlib scale argument is allowed. Default is "log".
		yscale (string): Set y axis scale. Any matplotlib scale argument is allowed. Default is "log".
		save (boolean): Saves figure in local directory with an interactively requested title. Defaults to False.
	
	Returns:
		None

	"""
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
	"""finds index in array such that the array component at the returned index is closest to the desired value.
	
	Args: 
		array (list): Array to search.
		value (float or int): Value to find closest value to.

	Returns: 
		idx (int): index at which array is closest to value

	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def chisq(model, data, var):
	"""Calculates chi square values of a model and data with a given variance.

	Args:
		model (list): model array.
		data (list): data array. Must have same len() as model array.
		variance (float or list): Data variance. Defaults to 10.

	Returns: 
		cs (float): Reduced chi square value.

	"""
	if var == 0:
		var = 10
	#make sure that the two arrays are comparable
	if len(data) == len(model):
		#if there's a variance array, iterate through it as i iterate through the model and data
		if np.size(var) > 1:
			#calculate the chi square vector using xs = (model - data)^2/variance^2 per pixel
			xs = [((model[n] - data[n])**2)/var[n]**2 for n in range(len(model))]
		#otherwise do the same thing but using the same variance value everywhere
		else:
			xs = [((model[n] - data[n])**2)/var**2 for n in range(len(model))]
		#return the chi square vector
		return np.asarray(xs)#np.sum(xs)/len(xs)
	#if the two vectors aren't the same length, yell at me
	else:
		return('data must be equal in length to model')

def shift(wl, spec, rv, bcarr, **kwargs):
	"""for bccorr, use bcarr as well, which should be EITHER:
	1) the pure barycentric velocity calculated elsewhere OR
	2) a dictionary with the following entries (all as floats, except the observatory name code, if using): 
	{'ra': RA (deg), 'dec': dec (deg), 'obs': observatory name or location of observatory, 'date': JD of midpoint of observation}
	The observatory can either be an observatory code as recognized in the PyAstronomy.pyasl.Observatory list of observatories,
	or an array containing longitude, latitude (both in deg) and altitude (in meters), in that order.

	To see a list of observatory codes use "PyAstronomy.pyasl.listobservatories()".
	
	Args:
		wl (list): wavelength array
		spec (list): flux array
		rv (float): Rotational velocity value
		bcarr (list): if len = 1, contains a precomputed barycentric velocity. Otherwise, should 
			be a dictionary with the following properties: either an "obs" keyword and code from pyasl
			or a long, lat, alt set of floats identifying the observatory coordinates.  

	Returns:
		barycentric velocity corrected wavelength vector using bccorr().

	"""
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

def broaden(even_wl, modelspec_interp, res, vsini = 0, limb = 0, plot = False):
	"""Adds resolution, vsin(i) broadening, taking into account limb darkening.

	Args: 
		even_wl (list): evenly spaced model wavelength vector
		modelspec_interp (list): model spectrum vector
		res (float): desired spectral resolution
		vsini (float): star vsin(i)
		limb (float): the limb darkening coeffecient
		plot (boolean): if True, plots the full input spectrum and the broadened output. Defaults to False.

	Returns:
		a tuple containing an evenly spaced wavelength vector spanning the width of the original wavelength range, and a corresponding flux vector

	"""

	#regrid by finding the smallest wavelength step 
	mindiff = np.inf

	for n in range(1, len(even_wl)):
		if even_wl[n] - even_wl[n-1] < mindiff:
			mindiff = even_wl[n] - even_wl[n-1]

	#interpolate the input values
	it = interp1d(even_wl, modelspec_interp)

	#make a new wavelength array that's evenly spaced with the smallest wavelength spacing in the input wl array
	w = np.arange(min(even_wl), max(even_wl), mindiff)

	sp = it(w)

	#do the instrumental broadening and truncate the ends because they get messy
	broad = pyasl.instrBroadGaussFast(w, sp, res, maxsig=5)
	broad[0:5] = broad[5] 
	broad[len(broad)-10:len(broad)] = broad[len(broad) - 11]

	#if I want to impose stellar parameters of v sin(i) and limb darkening, do that here
	if vsini != 0 and limb != 0:
		rot = pyasl.rotBroad(w, broad, limb, vsini)#, edgeHandling='firstlast')
	#otherwise just move on
	else:
		rot = broad

	#Make a plotting option just in case I want to double check that this is doing what it's supposed to
	if plot == True:

		plt.figure()
		plt.plot(w, sp, label = 'model')
		plt.plot(w, broad, label = 'broadened')
		plt.plot(w, rot, label = 'rotation')
		plt.legend(loc = 'best')
		plt.xlabel('wavelength (angstroms)')
		plt.ylabel('normalized flux')
		plt.savefig('rotation.pdf')

	#return the wavelength array and the broadened flux array
	return w, rot

def redres(wl, spec, factor):
	"""Imposes instrumental resolution limits on a spectrum and wavelength array

	"""
	wlnew = []
	specnew = []

	#step through the input wl array
	for i in range(len(wl)):
		#if the index is a multiple of the factor, append the new values to the lists to be returned
		if i%factor == 0:
			wlnew.append(wl[i])
			specnew.append(spec[i])
		#otherwise, add the spectrum value to the pre-existing entry for that wavelength chunk
		else:
			idx = int((i - i%factor)/factor)
			specnew[idx] += spec[i]

	#return the reduced spectrum and wl array
	return wlnew, specnew

def rmlines(wl, spec, **kwargs):
	"""Edits an input spectrum to remove emission lines

	Args: 
		wl (list): wavelength
		spec (list): spectrum.
		add_line (boolean): to add more lines to the linelist (interactive)
		buff (float): to change the buffer size, input a float here. otherwise the buffer size defaults to 15 angstroms
		uni (boolean): specifies unit for input spectrum wavelengths (default is microns) [T/F]
		conv (boolean): if unit is true, also specify conversion factor (wl = wl * conv) to microns

	Returns: 
		spectrum with the lines in the linelist file removed if they are in emission.

	"""
	#reads in a linelist, which contains the names, transition, and wavelength of each emission line
	names, transition, wav = np.genfromtxt('linelist.txt', unpack = True, autostrip = True)
	#define the gap to mark out
	space = 1.5e-3 #15 angstroms -> microns

	#check the kwargs
	for key, value in kwargs.items():
		#if i want to add a line, use add_lines, which then lets me append a new line
		if key == add_line:
			wl.append(input('What wavelengths (in microns) do you want to add? '))
		#if i want to change the size of region that's removed, use a new buffer
		if key == buff:
			space = value
		#if i want to change the unit, use uni to do so
		if key == uni:
			wl = wl * value

	diff = wl[10] - wl[9]

	#for each line, walk trhough and remove the line, replacing it with the mean value of the end points of the removed region
	for line in wav:
		end1 = find_nearest(wl, line-space)
		end2 = find_nearest(wl, line+space)
		if wl[end1] > min(wl) and wl[end2] < max(wl) and (end1, end2)> (0, 0) and (end1, end2) < (len(wl), len(wl)):
			for n in range(len(wl)):
				if wl[n] > wl[end1] and wl[n] < wl[end2] and spec[n] > (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2:
					spec[n] = (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2
	#return the spectrum
	return spec

def make_reg(wl, flux, waverange):
	"""given some wavelength range as an array, output flux and wavelength vectors within that range.

	Args:
		wl (list): wavelength array
		flux (list): flux array
		waverange (list): wavelength range array

	Returns: 
		wavelength and flux vectors within the given range
	
	Note:
		TO DO: interpolate instead of just pulling the closest indices
	"""
	#find the closest min and max index for the range given
	min_wl = find_nearest(wl, min(waverange))
	max_wl = find_nearest(wl, max(waverange))
	#truncate the wavelength and flux arrays
	wlslice = wl[min_wl:max_wl]
	fluxslice = flux[min_wl:max_wl]
	#and return them
	return wlslice, fluxslice

def interp_2_spec(spec1, spec2, ep1, ep2, val):
	"""Args: 
		spec1 (list): first spectrum array (fluxes only)
		spec2 (list): second spectrum array (fluxes only)
		ep1 (float): First gridpoint of the value we want to interpolate to.
		ep2 (float): Second gridpoint of the value we want to interpolate to.
		val (float): a value between ep1 and ep2 that we wish to interpolate to.

	Returns: 
		a spectrum without a wavelength parameter

	"""	
	ret_arr = []
	#make sure the two spectra are the same length
	if len(spec1) == len(spec2):
		#go through the spectra
		for n in range(len(spec1)):
			#the new value is the first gridpoint plus the difference between them weighted by the spacing between the two gridpoints and the desired value.
			#this is a simple linear interpolation at each wavelength point
			v = ((spec2[n] - spec1[n])/(ep2 - ep1)) * (val - ep1) + spec1[n]
			ret_arr.append(v)
		#return the new interpolated flux array
		return ret_arr

	#otherwise yell at me because i'm trying to interpolate things that don't have the same length
	else:
		return('the spectra must have the same length')

def make_varied_param(init, sig):
	"""randomly varies a parameter within a gaussian range based on given std deviation

	Args:
		init (float): initial value
		sig (float): std deviation of gaussian to draw from

	Returns: 
		the varied parameter.

	"""
	var = np.random.normal(init, sig)
	
	return var

def find_model(temp, logg, metal):
	"""Finds a filename for a phoenix model with values that fall on a grid point.
	Assumes that model files are in a subdirectory of the working directory, with that subdirectory called "SPECTRA"
	and that the file names take the form "lte{temp}-{log g}-{metallicity}.BT-Settl.spec.7new.txt"
	The file should contain a wavelength column and a flux column, where wavelength is in Angstroms
	and the flux is in units of log(erg/s/cm^2/Hz/surface area)

	Args: 
		temperature (float): temperature value
		log(g) (float): log(g) value
		metallicity (float): Metallicity value

	Note:
		Values must fall on the grid points of the model grid.

	Returns: 
		file name of the phoenix model with the specified parameters.

	"""
	temp = str(int(temp*1e-2)).zfill(3)
	metal = str(float(metal)).zfill(3)
	logg = str(float(logg)).zfill(3)
	file = glob('SPECTRA/lte{}-{}-0.0a+{}.BT-Settl.spec.7new.txt'.format(temp, logg, metal))[0]
	return file

def get_spec(temp, log_g, reg, metallicity = 0, normalize = True, wlunit = 'aa', pys = False, plot = False, model_dir = 'phoenix', resolution = 3000, reduce_res = True):
	"""Creates a spectrum from given parameters, either using the pysynphot utility from STScI or using a homemade interpolation scheme.
	Pysynphot may be slightly more reliable, but the homemade interpolation is more efficient (by a factor of ~2).
	
	TO DO: add a path variable so that this is more flexible, add contingency in the homemade interpolation for if metallicity is not zero

	Args: 
		temp (float): temperature value
		log_g (float): log(g) value
		reg (list): region array ([start, end])
		metallicity (float): Optional, defaults to 0
		normalize (boolean): Optional, defaults to True
		wlunit: Optional, wavelength unit. Defaults to angstroms ('aa'), also supports microns ('um').
		pys (boolean): Optional, set to True use pysynphot. Defaults to False.
		plot (boolean): Produces a plot of the output spectrum when it is a value in between the grid points and pys = False (defaults to False).

	Returns: 
		a wavelength array and a flux array, in the specified units, as a tuple. Flux is in units of F_lambda (I think)

	Note:
		Uses the Phoenix models as the base for calculations. 

	"""
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

		#initialize a time variable if i want to check how long this takes to run
		t1 = time.clock()
		
		#list all the spectrum files
		files = glob('SPECTRA/lte*.7new.txt')
		#initialize a tempeature array
		t = []
		#sort through and pick out the temperature value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[0].split('e')[1]
			if len(nu) < 4:
				nu = int(nu) * 1e2
				t.append(nu)
		#sort the temperature array so it's in order
		t = sorted(t)

		#initialize a non-redundant array
		temps = [min(t)]

		#go through the sorted array and if the temperature isn't already in the non-redundant array, put it in
		for n, tt in enumerate(t):
			if tt > temps[-1]:
				temps.append(tt)

		#find the closest temperature to the input value
		t1_idx = find_nearest(temps, temp)

		#if that input value is on a grid point, make the second spectrum the same temperature
		if temps[t1_idx] == temp:
			t2_idx = t1_idx
		#if the nearest temp value is above the input value, the other temperature should fall below
		elif temps[t1_idx] > temp:
			t2_idx = t1_idx - 1
		#otherwise the second temperature should fall above
		else:
			t2_idx = t1_idx + 1

		#temp1 and temp2 have been selected to enclose the temperature, or to be the temperature exactly if the temp requested is on a grid point
		temp1 = temps[t1_idx]
		temp2 = temps[t2_idx]

		#now do the same thing for log(g)
		l = sorted([float(files[n].split('-')[1]) for n in range(len(files))])

		lgs = [min(l)]

		for n, tt in enumerate(l):
			if tt > lgs[-1]:
				lgs.append(tt)

		lg1_idx = find_nearest(lgs, log_g)
		 
		if lgs[lg1_idx] == log_g:
			lg2_idx = lg1_idx
		elif lgs[lg1_idx] > log_g:
			lg2_idx = lg1_idx - 1
		else:
			lg2_idx = lg1_idx + 1

		lg1 = lgs[lg1_idx]
		lg2 = lgs[lg2_idx]


		#so now I have four grid points: t1 lg1, t1 lg2, t2 lg1, t2 lg2. now i have to sort out whether some of those grid points are the same value

		#define a first spectrum using t1 lg1 and unpack the spectrum 
		file1 = find_model(temp1, lg1, 0)
		w, s = np.genfromtxt(file1, usecols = (0,1), unpack = True)
		
		#if I'm looking for a log(g) and a temperature that fall on a grid point, things are easy
		#just open the file, make sure it's formatted correctly, and write it to new wavelength and spectrum arrays
		if lg1 == lg2 and temp1 == temp2:
			f = open(file1, 'r')
			spwave, spflux = [], []
			for line in f:
				l = line.strip().split(' ')
				spwave.append(l[0].strip())
				spflux.append(l[1].strip())
			spflux = [float(spflux[n]) for n in range(len(spflux))]

		#If the points don't all fall on the grid points, we need to get the second spectrum at point t2 lg2, as well as the cross products
		#(t1 lg2, t2 lg1)
		else:
			#find the second file as well (i already found t1 lg1 before this if/else loop)
			file2 = find_model(temp2, lg2, 0)
			#open the first file (again, from above) and make nicely formatted wl and flux arrays, doing all sorts of things to make sure the files read correctly
			f = open(file1, 'r')
			wl1, spec1 = [], []
			for line in f:
				l = line.strip().split(' ')
				wl1.append(l[0].strip())
				if l[1] != '':
					spec1.append(l[1].strip())
				else:
					spec1.append(l[2].strip())
			wl1 = [float(w) for w in wl1]
			spec1 = [float(spec1[n]) for n in range(len(spec1))]

			#and do it again for the second pair of points (t2 lg2)
			f = open(file2, 'r')
			wl2, spec2 = [], []
			for line in f:
				l = line.strip().split(' ')
				wl2.append(l[0].strip())
				if l[1] != '':
					spec2.append(l[1].strip())
				else:
					spec2.append(l[2].strip())
			wl2 = [float(w) for w in wl2]
			try:
				spec2 = [float(s) for s in spec2]
			except:
				print(file2)

			#now do it for the two cross products 
			#this is t1 lg2
			f = open(find_model(temp1, lg2, 0), 'r')
			t1wave, t1_inter = [], []
			for line in f:
				l = line.strip().split(' ')
				t1wave.append(l[0].strip())
				if l[1] != '':
					t1_inter.append(l[1].strip())
				else:
					t1_inter.append(l[2].strip())

			t1wave = [float(w) for w in t1wave]
			try:
				t1_inter = [float(t1_inter[n]) for n in range(len(t1_inter))]
			except:
				print(find_model(temp1, lg2, 0))

			#and again for t2 lg1
			f = open(find_model(temp2, lg1, 0), 'r')
			t2wave, t2_inter = [], []
			for line in f:
				l = line.strip().split(' ')
				t2wave.append(l[0].strip())
				if l[1] != '':
					t2_inter.append(l[1].strip())
				else:
					t2_inter.append(l[2].strip())
			
			t2wave = [float(w) for w in t2wave]
			try:
				t2_inter = [float(t2_inter[n]) for n in range(len(t2_inter))]
			except:
				print(find_model(temp2, lg1, 0))

			#so now I have four spectra, and I need to interpolate them correctly to get to some point between the grid points in both log(g) and teff space

			#first make sure everything is on the same wavelength grid
			#find the smallest wavelength step used in the wavelength array
			mindiff = np.inf
			for w in range(1, len(wl1)):
				diff = wl1[w] - wl1[w - 1]
				if diff < mindiff:
					mindiff = diff

			#make a new wl vector using the requested spectral region (which is given in microns, but we're working in angstroms) and that smallest wavelength step
			wls = np.arange(min(reg)*1e4, max(reg)*1e4, mindiff)

			#Convert all the spectrafrom the weird PHOENIX units of log(s) + 8 to just s, where s is in units of erg/s/cm^2/A/surface area 
			spec1, spec2, t1_inter, t2_inter = [10 ** (s - 8) for s in spec1], [10 ** (s - 8) for s in spec2], \
				[10 ** (s - 8) for s in t1_inter], [10 ** (s - 8) for s in t2_inter]

			#interpolate everything onto the same grid using the newly defined wavelength array
			iw1 = interp1d(wl1, spec1)
			spec1 = iw1(wls)
			iw2 = interp1d(wl2, spec2)
			spec2 = iw2(wls)

			it1 = interp1d(t1wave, t1_inter)
			t1_inter = it1(wls)
			it2 = interp1d(t2wave, t2_inter)
			t2_inter = it2(wls)

			#if t1 and t2 AND lg1 and lg2 are different, we need to interpolate first between the two log(g) points, then the two teff points
			if lg1 != lg2 and temp1 != temp2:
				t1_lg = interp_2_spec(spec1, t1_inter, lg1, lg2, log_g)
				t2_lg = interp_2_spec(t2_inter, spec2, lg1, lg2, log_g)

				tlg = interp_2_spec(t1_lg, t2_lg, temp1, temp2, temp)

			#or if we're looking at the same log(g), we only need to interpolate in temperature
			elif lg1 == lg2:
				tlg = interp_2_spec(spec1, spec2, temp1, temp2, temp)

			#similarly, if we're using the same temperature but different log(g), we only interpolate in log(g)
			elif temp1 == temp2:
				tlg = interp_2_spec(spec1, spec2, lg1, lg2, log_g)

			#if you want, make a plot of all the different spectra to compare them
			#this only plots the final interpolated spectrum and the two teff points that are interpolated, after the log(g) interpolation has occurred
			if plot == True:
				wl1a, tla = make_reg(wls, tlg, [1e4, 1e5])
				wl1a, t1l1a = make_reg(wls, t1_lg, [1e4, 1e5])
				wl1a, t1l2a = make_reg(wls, t2_lg, [1e4, 1e5])
				plt.loglog(wl1a, tla, label = 'tl')
				plt.loglog(wl1a, t1l1a, label = 't1l1')
				plt.loglog(wl1a, t1l2a, label = 't1l2')
				plt.legend()
				plt.show()
			
			#reassign some variables used above to match with the environment outside the if/else statement
			spwave = wls
			spf = tlg

	#convert the requested region into angstroms to match the wavelength vector
	reg = [reg[n] * 1e4 for n in range(len(reg))]

	#make sure the flux array is a float not a string
	spflux = []
	for s in spf:
		spflux.append(float(s))

	#and truncate the wavelength and flux vectors to contain only the requested region
	spwave, spflux = make_reg(spwave, spflux, reg)

	#you can choose to normalize
	if normalize == True:
		if len(spflux) > 0:
			if max(spflux) > 0:
				spflux = [spflux[n]/max(spflux) for n in range(len(spflux))]
		else:
			spflux = np.ones(len(spflux))

	#this is the second time object in case you want to check runtime
	t2 = time.clock()
	#print('runtime for spectral retrieval (s): ', t2 - t1)

	#broaden the spectrum to mimic the dispersion of a spectrograph using the input resolution
	spwave, spflux = broaden(spwave, spflux, resolution)

	#and reduce the resolution again to mimic pixellation from a CCD
	#i should make this nicer - it currently assumes you want three resolution elements per angstrom, which isn't necessarily true
	if reduce_res == True:
		factor = int(len(spwave)/((max(spwave) - min(spwave)) * 3)) #for R = 3000
		spwave, spflux = redres(spwave, spflux, factor)

	#depending on the requested return wavelength unit, do that, then return wavelength and flux as a tuple
	if wlunit == 'aa': #return in angstroms
		return spwave, spflux
	elif wlunit == 'um':
		spwave = spwave * 1e-4
		return spwave, spflux
	else:
		factor = float(input('That unit is not recognized for the return unit. \
			Please enter a multiplicative conversion factor to angstroms from your unit. For example, to convert to microns you would enter 1e-4.'))
		spwave = [s * factor for s in spwave]

		return spwave, spflux

# a, b = get_spec(3410, 4, [0.65, 0.66], normalize = False, resolution = 40000, reduce_res = False)
# c, d = get_spec(3410, 4, [0.65, 0.66], normalize = False, resolution = 3000, reduce_res = False)
# e,f = get_spec(3410, 4, [0.65, 0.66], normalize = False, resolution = 3000, reduce_res = True)

# g, h = get_spec(3510, 4, [0.65, 0.66], normalize = False, resolution = 40000, reduce_res = False)
# i, j = get_spec(3510, 4, [0.65, 0.66], normalize = False, resolution = 3000, reduce_res = False)
# k,l = get_spec(3510, 4, [0.65, 0.66], normalize = False, resolution = 3000, reduce_res = True)

# plt.figure()
# plt.plot(a,b, label = '3400, raw data')
# plt.plot(c,d, label = '3400, broadened')
# plt.plot(e,f, label = '3400, reduced')

# plt.plot(g,h, label = '3500, raw data')
# plt.plot(i,j, label = '3500, broadened')
# plt.plot(k,l, label = '3500, reduced')


# plt.legend(loc = 'best')
# plt.show()
# plt.savefig('test_spec_retrieval.png')

def add_spec(wl, spec, flux_ratio, normalize = True):#, waverange):
	"""add spectra together given an array of spectra and flux ratios
	TO DO: handle multiple flux ratios in different spectral ranges

	Args: 
		wl (2-d array): wavelength array (of vectors)
		spec (2-d array): spectrum array (of vectors), 
		flux_ratio (array): flux ratio array with len = len(spectrum_array) - 1, where the final entry is the wavelength to normalize at, sand whether or not to normalize (default is True)
		normalize (boolean): Normalize the spectra before adding them (default is True)

	Returns: 
		spec1 (list): spectra added together with the given flux ratio

	"""
	wl_norm = find_nearest(wl[0][:], flux_ratio[-1])
	spec1 = spec[0][:]
	for n in range(0, len(spec)-1):
		spec2 = spec[n+1][:]
		#ratio = spec2[wl_norm]/spec1[wl_norm]
		num = flux_ratio[n]#/ratio
		spec2 = [spec2[k] * num for k in range(len(spec1))]
		spec1 = [spec1[k] + spec2[k] for k in range(len(spec1))]
	if normalize == True:
	#normalize and return
		spec1 = spec1/max(spec1)
	return spec

def make_bb_continuum(wl, spec, dust_arr, wl_unit = 'um'):
	"""Adds a dust continuum to an input spectrum.

	Args:
		wl (list): wavelength array
		spec (list): spectrum array
		dust_arr (list): an array of dust temperatures
		wl_unit (string): wavelength unit - supports 'aa' or 'um'. Default is 'um'.

	Returns:
		a spectrum array with dust continuum values added to the flux.

	"""
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

def fit_spec(n_walkers, wl, flux, reg, fr, guess_init, sig_init = {'t':[200, 200], 'lg':[0.2, 0.2], 'dust': [100], 'extinct': [0.1]}, wu='um', burn = 100, cs = 10, steps = 200, pysyn=False, dust = False):
	"""Does an MCMC to fit a combined model spectrum to an observed single spectrum.
	guess_init and sig_init should be dictionaries of component names and values for the input guess and the 
	prior standard deviation, respectively. 
	Assumes they have the same metallicity.
	The code will expect an dictionary with values for temperature ('t'), log g ('lg'), and dust ('dust') right now.
	TO DO: add line broadening, disk/dust to possible fit params.

	Args:
		n_walkers (int): number of walkers
		wl (list): wavelength array
		flux (list): spectrum array
		reg (list): Two value array with start and end points for fitting.
		fr (list): flux ratio array. Value1 is flux ratio, value2 is location in the spectrum of value1, etc.
		guess_init (dictionary): dictionary of component names and values for the input guess. The code will expect an dictionary with values for temperature ('t'), log g ('lg'), extinction ('extinct'), and dust ('dust').
		sig_init (dictionary): A dictionary with corresponding standard deviations for each input guess. Default is 200 for temperature, 0.2 for log(g)
		wu (string): wavelength unit. currently supports 'aa' or 'um'. Default: "um".
		burn (int): how many initial steps to discard to make sure walkers are spread out. Default: 100.
		cs (int): cutoff chi square to decide convergence. Default: 10.
		steps (int): maximum steps to take after the burn-in steps. Default: 200.
		pysyn (Bool): Boolean command of whether or not to use pysynphot for spectral synthesis. Default: False
		conv (Bool): Use chi-square for convergence (True) or the number of steps (False). Default: True.
		dust (Bool): Add a dust spectrum. Default: False.

	"""
	if 'm' in guess_init:
		metal = guess_init['m']
	else:
		metal = 0

	#make some initial guess' primary and secondary spectra, then add them
	wave1, spec1 = get_spec(guess_init['t'][0], guess_init['lg'][0], reg, metallicity = metal, wlunit = wu, pys = pysyn)
	wave2, spec2 = get_spec(guess_init['t'][1], guess_init['lg'][1], reg, metallicity = metal, wlunit = wu, pys = pysyn)

	init_cspec = add_spec(wave1, wave2, spec1, spec2, fr)

	if 'extinct' in guess_init:
		extinction = guess_init['extinct']
	else: 
		extinction = 0

	init_cspec = extinct(wave1, init_cspec, extinction)

	if dust == True:
		init_cspec = add_dust(init_cspec, guess_init['dust'][0])

	#calculate the chi square value of that fit
	init_cs= chisq(flux, init_cspec, 0)
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

			if 'extinct' in guess_init:
				test_cspec = extinct(test_wave1, test_cspec, var_par[5])


			#calc chi square between data and proposed change
			test_cs = chisq(test_cspec, flux, 0)

			lh = np.exp(-1 * (init_cs)/2 + (test_cs)/2)

			u = np.random.uniform(0, 1)

			if chi > test_cs and lh > u:
				gi[vp] = var_par[vp]
				chi = test_cs 

			if n > burn:
				sp = np.vstack((sp, gi))
				savechi.append(chi)
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
	"""The natural logarithm of the joint likelihood. 
	Set to the chisquare value. (we want uniform acceptance weighted by the significance)
	
	Possible kwargs are reg (region), wlunit ('um' or 'aa'), dust (defaults to False), \
		normalize (defaults to True), pysyn (defaults to False), 

	Args:
		p0 (list): a sample containing individual parameter values. Then p0[0: n] = temp, p0[n : 2n] = lg, p0[2n : -1] = dust temps
		nspec (int): number of spectra/stars
		ndust (int): number of dust continuum components
		data (list): the set of data/observations
		flux_ratio (array): set of flux ratios with corresponding wavelength value for location of ratio
		broadening (int): The instrumental resolution of the spectra
		r (list): region to use when calculating liklihood
		w (string): Wavelength unit, options are "aa" and "um". Default is "aa".
		pysyn (bool): Use pysynphot to calculate spectra. Default is False.
		dust (bool): Add dust continuum? Default is False.
		norm (bool): Normalize spectra when fitting. Default is True.

	Returns: 
		cs (float): a reduced chi square value corresponding to the quality of the fit.

	Note:
		current options: arbitrary stars, dust (multi-valued). 
		To do: fit for broadening or vsini.

	"""
	le = len(data[:][1])

	wl = np.zeros(le)
	spec = np.zeros(le)

	for n in range(nspec):
		if len(p0) == nspec:
			lg = 4.5
		else:
			lg = p0[nspec + n]

		ww, spex = get_spec(p0[n], lg, normalize = norm, reg = r, wlunit = w, pys = pysyn)

		wl1 = np.linspace(min(ww), max(ww), le)

		if len(spex) == 0:
			spex = np.ones(len(ww))

		#print(le, np.shape(ww), np.shape(spex))
		intep = scipy.interpolate.interp1d(ww, spex)
		spec1 = intep(wl1)

		wl = np.vstack((wl, wl1))
		spec = np.vstack((spec, spec1))

	test_spec = add_spec(wl, spec, flux_ratio)

	if dust == True:
		test_spec = make_bb_continuum([wl[:][1], test_spec], p0[2 * nspec : -1], wl_unit = w)

	test_wl, test_spec = broaden(wl[:][1], test_spec, broadening, 0, 0, plot=False)
	init_cs = chisq(test_spec, data[:][-1], 0)

	if np.isnan(init_cs):
		init_cs = -np.inf

	return init_cs

# WE ASSUME A UNIFORM PRIOR -- add something more sophisticated eventually

def logprior(p0, nspec, ndust):
	temps = p0[0:nspec]
	lgs = [p0[nspec]]

	if ndust > 0:
		dust = p0[2 * nspec : 2 * nspec + ndust]
	for p in range(nspec):
		if 400 <= temps[p] <= 6500 and 3.5 <= lgs[p] <= 5:
			return 0.0
		else:
			return -np.inf

def logposterior(p0, nspec, ndust, data, flux_ratio, broadening, r, wu = 'aa', pysyn = False, dust = False, norm = True):
	"""The natural logarithm of the joint posterior.

	Args:
		p0 (list): a sample containing individual parameter values. Then p0[0: n] = temp, p0[n : 2n] = lg, p0[2n : -1] = dust temps
		nspec (int): number of spectra/stars
		ndust (int): number of dust continuum components
		data (list): the set of data/observations
		flux_ratio (array): set of flux ratios with corresponding wavelength value for location of ratio
		broadening (int): The instrumental resolution of the spectra
		r (list): region to use when calculating liklihood
		w (string): Wavelength unit, options are "aa" and "um". Default is "aa".
		pysyn (bool): Use pysynphot to calculate spectra. Default is False.
		dust (bool): Add dust continuum? Default is False.
		norm (bool): Normalize spectra when fitting. Default is True.

	Returns: 
		lh (float): The log of the liklihood of the fit being pulled from the model distribution.

	Note:
		Assuming a uniform prior for now

	"""
	lp = logprior(p0, nspec, ndust)

	# if the prior is not finite return a probability of zero (log probability of -inf)
	if not np.isfinite(lp):
		return -np.inf
	lh = loglikelihood(p0, nspec, ndust, data, flux_ratio, broadening, r, w = wu, pysyn = False, dust = False, norm = True)
	# return the likeihood times the prior (log likelihood plus the log prior)
	return lp + lh


def run_emcee(fname, nwalkers, nsteps, ndim, nburn, pos, nspec, ndust, data, flux_ratio, broadening, r, nthin=10, w = 'aa', pys = False, du = False, no = True, which='em'):
	"""Run the emcee code to fit a spectrum 

	Args:
		fname (string): input file name to use
		nwalkers (int): number of walkers to use
		nsteps (int): number of steps for each walker to take
		ndim (int): number of dimensions to fit to. For a single spectrum to fit temperature and log(g) for, ndim would be 2, for example. 
		nburn (int): number of steps to discard before starting the sampling. Should be large enough that the walkers are well distributed before sampling starts.
		pos (list): array containing the initial guesses for temperature and log g.
		nspec (int): number of spectra to fit to. For a single spectrum fit this would be 1, for a two component fit this should be 2.
		ndust (int): number of dust continuum components to fit to. (untested)
		data (list): the spectrum to fit to
		flux_ratio (list): an array with a series of flux ratios, followed by the wavelength at which they were measured.
		broadening (float): the instrumental resolution of the input data, or the desired resolution to use to fit.
		r (list): a two valued array containing the region to fit within, in microns.
		nthin (int): the sampling rate of walker steps to save. Default is 10.
		w (string): the wavelength unit to use. Accepts 'um' and 'aa'. Default is 'aa'.
		pys (boolean): Whether to use pysynphot for spectral synthesis (if true). Default is False.
		du (boolean): Whether to fit to dust components. Default is False.
		no (boolean): Whether to normalize the spectra while fitting. Default is True.
		which (string): Use an ensemble sampler ('em') or parallel tempered sampling ('pt'). Default is 'em'. More documentation can be found in the emcee docs.
	
	Note:
		This is still in active development and doesn't always work.

	"""
	if which == 'em':
		sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, threads=nwalkers, args=[nspec, ndust, data, flux_ratio, broadening, r], \
		kwargs={'pysyn': pys, 'dust': du, 'norm':no})

		for p, lnprob, lnlike in sampler.sample(pos, iterations=nburn):
			pass
		sampler.reset()

	if which == 'pt':
		ntemps = int(input('How many temperatures would you like to try? '))
		sampler = emcee.PTSampler(ntemps, nwalkers, ndim, loglikelihood, logprior, threads=nwalkers, loglargs=[\
		nspec, ndust, data, flux_ratio, broadening, r], logpargs=[nspec, ndust], loglkwargs={'w':w, 'pysyn': pys, 'dust': du, 'norm':no})

		for p, lnprob, lnlike in sampler.sample(pos, iterations=nburn):
			pass
		sampler.reset()

		#for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike, iterations=nsteps, thin=nthin):
		#	pass

		assert sampler.chain.shape == (ntemps, nwalkers, nsteps/nthin, ndim)

		# Chain has shape (ntemps, nwalkers, nsteps, ndim)
		# Zero temperature mean:
		mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

		try:
			# Longest autocorrelation length (over any temperature)
			max_acl = np.max(sampler.acor)
			print('max acl: ', max_acl)
			np.savetxt('results/acor.txt', sampler.acor)
		except:
			pass

	f = open("results/{}_chain.txt".format(fname), "w")
	f.close()
	
	for result in sampler.sample(pos, iterations=nsteps, thin = nthin):
		f = open("results/{}_chain.txt".format(fname), "w")
		f.write("{}\n".format(result))
		f.close()
	for i in range(ndim):
		plt.figure(i)
		plt.hist(sampler.flatchain[:,i], nsteps, histtype="step")
		plt.title("Dimension {0:d}".format(i))
		plt.savefig('results/plots/{}_{}.pdf'.format(fname, i))
		plt.close()

		plt.figure(i)

		try:
			for n in range(nwalkers):
				plt.plot(np.arange(nsteps),sampler.chain[n, :, i])
			plt.savefig('results/plots/{}_chain_{}.pdf'.format(fname, i))
			plt.close()
		except:
			pass
	try:
		chain = sampler.chain[:, :, 0].T
		N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
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
		plt.savefig('results/plots/{}_autocorr.pdf'.format(fname))
		plt.close()
	except:
		pass;

	samples = sampler.chain[:, :, :].reshape((-1, ndim))
	fig = corner.corner(samples)
	fig.savefig("results/plots/{}_triangle.pdf".format(fname))
	plt.close()

	print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
	return

import numpy as np
#import pysynphot as ps
import matplotlib.pyplot as plt
from astropy.io import fits
import os 
from glob import glob
from astropy import units as u
from matplotlib import rc
import matplotlib
rc('text', usetex=True)
import model_fit_tools_v2 as mft2
import IMF as imf
from scipy.stats import norm
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['teal', 'turquoise', 'black', 'blue', 'green', 'lightblue', 'cyan', 'lime', 'purple', 'lavender', 'darkgreen']) 

pmass, pmult, page, pav, ptemp, plogg, plum, pdist = np.genfromtxt('run6/ppar.txt', autostrip=True, unpack=True, skip_header=1)
smass, ssep, sage, secc, sper, stemp, slogg, slum = np.genfromtxt('run6/spar.txt', autostrip=True, unpack=True, skip_header=1)

pm = np.where(pmult == 1)
sm = np.where(pmult != 1)
bi = pmass[pm]
sing = pmass[sm]

print(len(bi), len(smass))
mr = np.zeros(len(smass))
for n in range(len(smass)):
	mr[n] = bi[n]/smass[n]

comb_mass = [bi[n] + smass[n] for n in range(len(smass))]
[comb_mass.append(s) for s in sing]

massrange, chab = imf.calc_pct('c')
chab = [chab[n] * 1e2 for n in range(len(chab))]

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.hist(mr, bins = 30)
ax1.set_xlabel('Mass ratio')

ax2.hist(comb_mass, bins = 100, label = 'All system masses')
#ax2.hist(pmass, bins = 100, label = 'Primary stars', alpha = 0.8)
ax2.hist(sing, bins = 100, label = 'Single stars', alpha = 0.8)
ax2.plot(massrange, chab, linestyle = '--')
ax2.legend(loc = 'best')
ax2.set_xlim(0, 5)
ax2.set_yscale('log')
ax2.set_xlabel(r'Mass ($M_{\odot}$)')

ax1.minorticks_on()
ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax1.tick_params(bottom=True, top =True, left=True, right=True)
ax1.tick_params(which='both', labelsize = "large", direction='in')
ax1.tick_params('both', length=6, width=1.25, which='major')
ax1.tick_params('both', length=4, width=1, which='minor')

ax2.minorticks_on()
ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax2.tick_params(bottom=True, top =True, left=True, right=True)
ax2.tick_params(which='both', labelsize = "large", direction='in')
ax2.tick_params('both', length=6, width=1.25, which='major')
ax2.tick_params('both', length=4, width=1, which='minor')

fig1.tight_layout()
fig1.savefig('check_pars.pdf')

fig2, (ax3, ax4) = plt.subplots(ncols=2)
ax3.hist(secc, label ='eccentricity', color ='xkcd:sky blue')
ax3.axvline(np.mean(secc), label = 'mean value')
#ax3.hist(ssep, label ='separation', alpha = 0.8, color ='xkcd:sky blue')
ax3.legend(loc='lower left')

mu, sig = norm.fit(sper)
a = np.arange(mu - sig*4, mu + sig*4, 0.02)
p = norm.pdf(a, mu, sig) * (7/3)*180

ax4.plot(a, p, linestyle = '-.', label = r'$\mu$ = {}, $\sigma$ = {}'.format(round(mu, 2), round(sig, 2)), color = 'k')
ax4.hist(sper, color ='teal', label = 'log(P) (days)')
ax4.legend(loc = 'lower left')

ax3.minorticks_on()
ax3.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax3.tick_params(bottom=True, top =True, left=True, right=True)
ax3.tick_params(which='both', labelsize = "large", direction='in')
ax3.tick_params('both', length=6, width=1.25, which='major')
ax3.tick_params('both', length=4, width=1, which='minor')

ax4.minorticks_on()
ax4.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax4.tick_params(bottom=True, top =True, left=True, right=True)
ax4.tick_params(which='both', labelsize = "large", direction='in')
ax4.tick_params('both', length=6, width=1.25, which='major')
ax4.tick_params('both', length=4, width=1, which='minor')

fig2.tight_layout()
fig2.savefig('sec_pars.pdf')
import numpy as np
import matplotlib.pyplot as plt 
import model_fit_tools_v2 as mft
import emcee 

data_wl, data_spec = np.genfromtxt('Data/Spectra_for_Kendall/fftau_wifes_spec.csv', delimiter=',', unpack = True)

data_wl = data_wl[1:-1]/1e4
data_spec = data_spec[1:-1]

fig, ax = plt.subplots(2, sharex = True)
ax[0].plot(data_wl, data_spec, label = "old", color='g')

newspec = mft.rmlines(data_wl, data_spec)

ax[1].plot(data_wl, newspec, label="trimmed", color='k')
fig.legend()
plt.savefig('modified_spec.pdf')

data = [data_wl, data_spec]

cen_wl = data_wl[int(len(data_wl)/2)]

nspec, ndust = 2, 0

nwalkers, nsteps, ndim, nburn, flux_ratio, broadening = 4, 20, 2, 10, [0.25, cen_wl], 3000

pos = [4100, 3800]

p0 = emcee.utils.sample_ball(pos, [150, 150], size=nwalkers)

a = mft.run_emcee(nwalkers, nsteps, ndim, nburn, p0, nspec, ndust, data, flux_ratio, broadening, [min(data_wl), max(data_wl)], \
	nthin=2, w = 'aa', pys = False, du = False, no = False)
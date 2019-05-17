import numpy as np
import matplotlib.pyplot as plt 
import model_fit_tools as mft
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

ndim, nsteps, nwalkers = 4, 40, 20


p0  = emcee.utils.sample_ball([4000, 4000, 3, 3], [200, 200, 0.2, 0.2], size=(nwalkers))

a = mft.run_emcee(nwalkers, nsteps, ndim, p0, 2, 0, [data_wl, data_spec], [0.5], 3000, [min(data_wl), max(data_wl)], w = 'um')
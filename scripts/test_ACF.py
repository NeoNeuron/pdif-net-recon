#%%
import causal4.utils as c4u
import numpy as np
import matplotlib.pyplot as plt

# %%
spk_data = c4u.load_spike_data('/data/kchen/causal4-dev/benchmark/N10/HHII/HHp=0.25s=0.080f=0.080u=0.150_spike_train.dat',
                    xrange=(0,2000), verbose=True)
# %%
spkbin = c4u.spk2bin(spk_data[spk_data[:,1]==0,], dt=0.5)
plt.plot(spkbin)
# plt.plot(spk_data[:,0], spk_data[:,1], '|')
#%%
acf = c4u.ACF(spkbin, nlags=100, plot=True)
# %%
spk_data = c4u.load_spike_data('/data/kchen/causal4-dev/benchmark/N10/Logistic/Logp=0.25s=0.005_spike_train.dat',
                    xrange=(0,1000), verbose=True)
# %%
spkbin = c4u.spk2bin(spk_data[spk_data[:,1]==0,], dt=2)
plt.plot(spkbin)
#%%
plt.plot(spk_data[:,0], spk_data[:,1], '|')
plt.xlim(0,100)
#%%
acf = c4u.ACF(spkbin, nlags=20, plot=True)
# %%
acf
# %%

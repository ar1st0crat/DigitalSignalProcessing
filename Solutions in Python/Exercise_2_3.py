# -*- coding: utf-8 -*-
import numpy as np
from pylab import specgram
import matplotlib.pyplot as plt
  

t = 0.6             # duration: 600 milliseconds
fs = 16000          # sampling frequency: 16000 Hz
a1 = 2              # amplitudes:
a2 = 3
a3 = 1

f1 = 900           # frequency components:
f2 = 1400
f3 = 6100

plot_samples = 500

# initial signal (s1)
n = np.arange(fs*t)
s1 = a1 * np.sin( 2*np.pi*n*f1/fs ) + \
     a2 * np.sin( 2*np.pi*n*f2/fs ) + \
     a3 * np.sin( 2*np.pi*n*f3/fs )

plt.figure(num="Initial signal")
plt.subplot(211)
plt.plot(n[:plot_samples], s1[:plot_samples])

# spectrum of s1 (fs samples taken)
spec = np.abs(np.fft.fft(s1,fs))
nspec = np.arange(np.size(spec))

plt.subplot(212)
plt.plot(nspec, spec)
plt.show()


# resampled signal (s4) at 11025 Hz
n = np.arange(fs*t)
s4 = a1 * np.sin( 2*np.pi*n*f1/11025 ) + \
     a2 * np.sin( 2*np.pi*n*f2/11025 ) + \
     a3 * np.sin( 2*np.pi*n*f3/11025 )

plt.figure(num="Initial signal sampled at 11025 Hz")
plt.subplot(211)
plt.plot(n[:plot_samples], s4[:plot_samples])

# spectrum of s1 (fs samples taken)
spec = np.abs(np.fft.fft(s4,fs))
nspec = np.arange(np.size(spec))

plt.subplot(212)
plt.plot(nspec, spec)
plt.show()


# noise (sn)
mu, sigma = 0, np.sqrt(max([a1,a2,a3]))
sn = np.random.normal(mu,sigma, t*fs)

plt.figure(num="Noisy signal")
plt.subplot(211)
plt.plot(n[:plot_samples], sn[:plot_samples])

# spectrum of sn (fs samples taken)
spec = np.abs(np.fft.fft(sn,fs))
nspec = np.arange(np.size(spec))

plt.subplot(212)
plt.plot(nspec, spec)
plt.show()

# plot spectrograms of s1 and sn
plt.figure("Spectrograms")
plt.subplot(211)
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(212)
Pxx2, freqs2, bins2, im2 = specgram(sn, 256, fs)
plt.show()

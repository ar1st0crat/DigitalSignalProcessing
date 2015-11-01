# -*- coding: utf-8 -*-

"""
Write code to do the following:
1. Resample a signal at new rate fsNew=20 kHz using MATLAB functions interp() and decimate().
2. Resample a signal at new rate fsNew=20 kHz using MATLAB function resample().
3. Plot spectrogram of each signal after resampling.

NOTE:
sciPy has resample() and decimate() functions.
We also  define our own function interp() that works like its MATLAB analog:
- insert zeros
- do LP-filtering
"""

import numpy as np
import scipy.signal as sig
from pylab import specgram
import matplotlib.pyplot as plt


def interp(x,n):
    if n == 1:
        return x
        
    # insert zeros after each n-th sample in x
    y = np.zeros(len(x) * n)
    y[::n] = x
    
    # design LP-filter and do the iterpolation filtering
    cutoff = 1.0/n
    b = sig.firwin(90, cutoff)    
    b *= n
    y = sig.filtfilt(b,[1], y)
    
    return y


def gcd(a,b):
    if b == 0:
        return a
    else:
        return gcd(b, a%b)
    
    

t = 0.6             # duration: 600 milliseconds
fs = 16000          # sampling frequency: 16000 Hz
fsNew = 20000
a1 = 2              # amplitudes:
a2 = 3
a3 = 1

f1 = 900           # frequency components:
f2 = 1400
f3 = 6100

plot_samples = 300


# initial signal (s1)
n = np.arange(fs*t)
s1 = a1 * np.sin( 2*np.pi*n*f1/fs ) + \
     a2 * np.sin( 2*np.pi*n*f2/fs ) + \
     a3 * np.sin( 2*np.pi*n*f3/fs )


divisor = gcd(fs, fsNew)
y = interp(s1, fsNew/divisor)
y = sig.decimate(y, fs/divisor)

# plot spectrograms "before and after"
plt.figure("Before and after interp() and decimate()")
plt.subplot(221)
specgram(s1, 256, fs)
plt.subplot(222)
plt.plot(s1[:plot_samples])
plt.subplot(223)
specgram(y, 256, fsNew)
plt.subplot(224)
plt.plot(y[:plot_samples*float(fsNew)/fs])      # multiply the number of plot_samples by fsNew / fs


y = sig.resample(s1, fsNew*t)

# plot spectrograms "before and after"
plt.figure("Before and after resample()")
plt.subplot(221)
specgram(s1, 256, fs)
plt.subplot(222)
plt.plot(s1[:plot_samples])
plt.subplot(223)
specgram(y, 256, fsNew)
plt.subplot(224)
plt.plot(y[:plot_samples*float(fsNew)/fs])
plt.show()
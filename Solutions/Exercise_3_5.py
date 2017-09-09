# -*- coding: utf-8 -*-

"""
Design low-pass IIR filter with cutoff frequency fc=f1.
Write code that directly implements in time domain filter you’ve designed.
Save filtered signals to WAVE files.
Plot the spectrograms of each signal before and after filtering.
"""

import numpy as np
import scipy.signal as sig
from pylab import specgram
import matplotlib.pyplot as plt

"""
User-defined function
Filter signal using conventional difference equation:
y[n] = sum(b[k]*x[n-k]) - sum(a[m]*y[n-m])
"""
def FilterSignal(x,b,a):
    N = len(x)
    y = np.zeros(N)
    
    startPos = max(len(a), len(b))
    for n in range(startPos,N):
        for k in range(len(b)):
            y[n] += b[k]*x[n-k]
        for m in range(1,len(a)):
            y[n] -= a[m]*y[n-m]
    
    return y
    

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

cutoff = (f1 * 2.0) / fs

# IIR filter design

# 1) Custom IIR filter. In this example the filter is elliptic
# b,a = sig.iirdesign(wp = [0.0, cutoff-0.01], ws=[cutoff+0.01, 0.5], gstop= 60, gpass=1, ftype='ellip')
# 2) Butterworth IIR filter
b,a = sig.butter(4, cutoff) 
w,h = sig.freqz(b,a)

plt.figure(num="Designed filter")
plt.subplot(211)
plt.title("Magnitude response")
plt.plot(np.abs(h))
plt.subplot(212)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h)))

# filter
filtered = FilterSignal(s1, b, a)

plt.figure(num="Filtered signal")
plt.subplot(211)
plt.plot(s1[:plot_samples])
plt.subplot(212)
plt.plot(filtered[:plot_samples])
plt.show()

# plot spectrograms "before and after"
plt.figure("Applying filter designed with firwin()")
plt.subplot(211)
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(212)
Pxx2, freqs2, bins2, im2 = specgram(filtered, 256, fs)
plt.show()
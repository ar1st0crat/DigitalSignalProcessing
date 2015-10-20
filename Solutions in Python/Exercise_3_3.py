# -*- coding: utf-8 -*-

"""
Write code that filters an input signal using MATLAB functions filter(), fir1(), fir2(), remez(). 
Your filter should be low-pass with cutoff frequency fc=f1.
Save filtered signal to WAVE file. Plot the spectrograms of signal before and after filtering.

NOTE: 
SciPy analogs of fir1(), fir2(), remez(), filter()
are firwin(), firwin2(), remez(), lfilter() respectively
"""

import numpy as np
import scipy.signal as sig
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

kernelsize = 213
cutoff = (f1 * 2.0) / fs

# firwin()
kernel1 = sig.firwin(kernelsize, cutoff )
w1,h1 = sig.freqz(kernel1)
filtered1 = sig.lfilter(kernel1, [1], s1)

# firwin2()
kernel2 = sig.firwin2(kernelsize, [0.0, cutoff-0.01, cutoff+0.01, 1.0], [1.0, 1.0, 0.0, 0.0])
w2,h2 = sig.freqz(kernel2)
filtered2 = sig.lfilter(kernel2, [1], s1)

# remez()
kernel3 = sig.remez(kernelsize+1, [0.0, cutoff/2-0.01, cutoff/2+0.01, 0.5], [1.0, 0.0])
w3,h3 = sig.freqz(kernel3)
filtered3 = sig.lfilter(kernel3, [1], s1)


plt.figure(num="firwin()")
plt.subplot(311)
plt.plot(s1[:plot_samples])
plt.subplot(312)
plt.plot(filtered1[:plot_samples])
plt.subplot(313)
plt.plot(np.abs(h1))
plt.show()

plt.figure(num="firwin2()")
plt.subplot(311)
plt.plot(s1[:plot_samples])
plt.subplot(312)
plt.plot(filtered2[:plot_samples])
plt.subplot(313)
plt.plot(np.abs(h2))
plt.show()

plt.figure(num="remez()")
plt.subplot(311)
plt.plot(s1[:plot_samples])
plt.subplot(312)
plt.plot(filtered3[:plot_samples])
plt.subplot(313)
plt.plot(np.abs(h3))
plt.show()


# plot spectrograms "before and after"
plt.figure("Applying filter designed with firwin()")
plt.subplot(211)
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(212)
Pxx2, freqs2, bins2, im2 = specgram(filtered1, 256, fs)
plt.show()

plt.figure("Applying filter designed with firwin2()")
plt.subplot(211)
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(212)
Pxx2, freqs2, bins2, im2 = specgram(filtered2, 256, fs)
plt.show()

plt.figure("Applying filter designed with remez()")
plt.subplot(211)
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(212)
Pxx2, freqs2, bins2, im2 = specgram(filtered3, 256, fs)
plt.show()
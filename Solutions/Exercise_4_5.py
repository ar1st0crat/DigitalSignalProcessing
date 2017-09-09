# -*- coding: utf-8 -*-

"""
Use filter kernel computed in exercise 3.5.
Write code that computes new filter kernel to make filter high-pass with the same cutoff frequency.
Implement and your high-pass filter. Plot the spectrograms of a signal before and after filtering
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


# initial signal (s1)
n = np.arange(fs*t)
s1 = a1 * np.sin( 2*np.pi*n*f1/fs ) + \
     a2 * np.sin( 2*np.pi*n*f2/fs ) + \
     a3 * np.sin( 2*np.pi*n*f3/fs )

kernelsize = 213
cutoff = (f3 * 2.0) / fs

# firwin() LP
kernelLP = sig.firwin(kernelsize, cutoff )

# compute HP kernel from LP kernel
kernelHP = -kernelLP
kernelHP[kernelsize/2] += 1

wLP, hLP = sig.freqz(kernelLP)
wHP, hHP = sig.freqz(kernelHP)

plt.figure(num="LP and HP filter kernels")
plt.subplot(221)
plt.plot(np.abs(hLP))
plt.subplot(222)
plt.plot(kernelLP)
plt.subplot(223)
plt.plot(np.abs(hHP))
plt.subplot(224)
plt.plot(kernelHP)
plt.show()

# filter signal with both LP and HP filters
filteredLP = sig.lfilter(kernelLP, [1], s1)
filteredHP = sig.lfilter(kernelHP, [1], s1)

# plot spectrograms "before and after"
plt.figure("Applying filter designed with firwin()")
plt.subplot(311)
specgram(s1, 256, fs)
plt.subplot(312)
specgram(filteredLP, 256, fs)
plt.subplot(313)
specgram(filteredHP, 256, fs)
plt.show()
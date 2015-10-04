# -*- coding: utf-8 -*-
"""
In this task you will investigate various window functions. Write code to do
the following:
    1. Compute and plot the rectangular, Hamming, Hanning, Blackman, and
       Kaiser window functions of length N = 93 on a single figure.
    2. Compute and plot magnitude spectrum on a dB scale of each window.
       Use FFT size NF=512. What can you say about rectangular window?
"""

import numpy as np
import scipy.special as bessel
import matplotlib.pyplot as plt

"""
User-defined function
Return rectangular window of size 'size'
"""
def rectangular(size):
    return np.ones(size)
    
"""
User-defined function
Return hamming window of size 'size'. Complete analog of numPy.hamming(size)
"""
def hamming(size):
    n = np.arange(size)
    return 0.54 - 0.46 * np.cos(2*np.pi*n/size)

"""
User-defined function
Return hanning window of size 'size'. Complete analog of numPy.hanning(size)
"""
def hanning(size):
    n = np.arange(size)
    return 0.5 - 0.5 * np.cos(2*np.pi*n/size)

"""
User-defined function
Return blackman window of size 'size'. Complete analog of numPy.blackman(size)
"""
def blackman(size):
    n = np.arange(size)
    return 0.42 - 0.5 * np.cos(2*np.pi*n/size) + 0.08 * np.cos(4*np.pi*n/size)

"""
User-defined function
Return kaiser window of size 'size' and beta parameter.
Complete analog of numPy.kaiser(size,beta)
"""
def kaiser(size,beta):
    window = np.zeros(size)
    for i in range(size):
        window[i] = bessel.i0(beta*np.sqrt(1 - np.power(2.0*i/(size-1)-1, 2))) / bessel.i0(beta)
    return window

FFT_size = 512
window_size = 93

# =========================================== Part 1 (plot window functions)
plt.figure(num="Window functions")
plt.ylim(ymax=1.1)
plt.plot(rectangular(window_size))
plt.plot(hamming(window_size))
plt.plot(hanning(window_size))
plt.plot(blackman(window_size))
plt.plot(kaiser(window_size,14))
plt.legend( ('rectangular','hamming','hanning','blackman','kaiser') )
plt.grid()
plt.show()


# =========================================== Part 2 (plot spectra on decibel scale (20*log10(x)))
plt.figure(num="Window functions spectra")

rect_spec = np.abs(np.fft.fft(rectangular(window_size),FFT_size))
plt.plot(20*np.log10(rect_spec[:FFT_size/2]))

hamming_spec = np.abs(np.fft.fft(hamming(window_size),FFT_size))
plt.plot(20*np.log10(hamming_spec[:FFT_size/2]))

hanning_spec = np.abs(np.fft.fft(hanning(window_size),FFT_size))
plt.plot(20*np.log10(hanning_spec[:FFT_size/2]))

blackman_spec = np.abs(np.fft.fft(blackman(window_size),FFT_size))
plt.plot(20*np.log10(blackman_spec[:FFT_size/2]))

kaiser_spec = np.abs(np.fft.fft(kaiser(window_size,14),FFT_size))
plt.plot(20*np.log10(kaiser_spec[:FFT_size/2]))

plt.legend( ('rectangular','hamming','hanning','blackman','kaiser') )
plt.grid()
plt.show()
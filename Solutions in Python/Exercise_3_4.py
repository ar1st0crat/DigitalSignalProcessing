# -*- coding: utf-8 -*-

"""
Open MATLAB Filter Design & Analysis Tool. Design low-pass FIR filter with cutoff frequency fc=f1. 
Write code that applies the designed filter using the Overlap-Add method. 
Write code that applies the designed filter using the Overlap-Save method. 
Save filtered signals to WAVE files. Plot the spectrograms of each signal before and after filtering.

NOTE: 
Instead of FDAtool we're simply using firwin() function
"""

import numpy as np
import scipy.signal as sig
from pylab import specgram
import matplotlib.pyplot as plt
  

"""
User-defined function
FFT Convolution
"""
def fft_convolve(a,b):
    N = len(a)
    M = len(b)
    
    A = np.fft.fft(a, n = N+M-1)
    B = np.fft.fft(b, n = N+M-1)
    C = A * B
    
    c = np.real(np.fft.ifft(C))
    
    return c[:N+M-1]
    
"""
User-defined function
Overlap-Add method of filtering
In:  x      - signal to filter
     kernel - FIR filter kernel of size M
     N      - overlap size: N+M-1 should be a power of 2
Out: y      - filtered signal
"""
def OverlapAdd(x,kernel,N):
    M = len(kernel)
    L = len(x)
    y = np.zeros(L)

    startPos = 0
    while startPos < L-N-M:
        conv = fft_convolve(x[startPos:startPos+N], kernel)
        y[startPos:startPos+N+M-1] += conv
        startPos += N
    
    return y
    
"""
User-defined function
Overlap-Save method of filtering
In:  x      - signal to filter
     kernel - FIR filter kernel of size M
     N      - overlap size: N+M-1 should be a power of 2
Out: y      - filtered signal
"""    
def OverlapSave(x,kernel,N):
    M = len(kernel)
    N += M-1
    L = len(x)
    y = np.zeros(L)
    
    startPos = 0
    conv = fft_convolve(x[startPos:startPos+N], kernel)
    y[startPos:startPos+N+M-1] = conv
    startPos += N-M+1
        
    while startPos < L-N:
        conv = fft_convolve(x[startPos:startPos+N], kernel)
        y[startPos:startPos + N] = conv[M-1:]
        startPos += N-M+1
    
    return y
    

t = 0.6             # duration: 600 milliseconds
fs = 16000          # sampling frequency: 16000 Hz
a1 = 2              # amplitudes:
a2 = 3
a3 = 1

f1 = 900           # frequency components:
f2 = 1400
f3 = 6100

plot_samples = 1500


# initial signal (s1)
n = np.arange(fs*t)
s1 = a1 * np.sin( 2*np.pi*n*f1/fs ) + \
     a2 * np.sin( 2*np.pi*n*f2/fs ) + \
     a3 * np.sin( 2*np.pi*n*f3/fs )

kernelsize = 213
overlapsize = 300               # overlapsize + kernelsize - 1 = 512
cutoff = (f1 * 2.0) / fs

# lowpass filter design via firwin()
kernel = sig.firwin(kernelsize, cutoff)
w,h = sig.freqz(kernel)
filtered1 = OverlapAdd(s1, kernel, overlapsize)
filtered2 = OverlapSave(s1, kernel, overlapsize)

plt.figure(num="Filtering by OverlapAdd and OverlapSave")
plt.subplot(411)
plt.plot(s1[:plot_samples])
plt.subplot(412)
plt.plot(filtered1[:plot_samples])
plt.subplot(413)
plt.plot(filtered2[:plot_samples])
plt.subplot(414)
plt.plot(np.abs(h))
plt.show()


# plot spectrograms "before and after"
plt.figure("Spectrograms")
plt.subplot(311)
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(312)
Pxx2, freqs2, bins2, im2 = specgram(filtered1, 256, fs)
plt.subplot(313)
Pxx3, freqs3, bins3, im3 = specgram(filtered2, 256, fs)
plt.show()

plt.figure(num="Difference")
plt.plot(filtered1[:1500] - filtered2[:1500])
plt.show()
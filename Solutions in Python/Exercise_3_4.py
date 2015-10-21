# -*- coding: utf-8 -*-

"""
Design low-pass FIR filter with cutoff frequency fc=f1. 
Write code that applies the designed filter using the Overlap-Add method. 
Write code that applies the designed filter using the Overlap-Save method. 
Save filtered signals to WAVE files. Plot the spectrograms of each signal before and after filtering.
"""

import numpy as np
import scipy.signal as sig
from pylab import specgram
import matplotlib
import matplotlib.pyplot as plt
  

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
    
    FilterFR = np.fft.fft(kernel, n = N+M-1)

    y = np.zeros(L+M-1)
    startPos = 0
    while startPos < L-N:
        A = np.fft.fft(x[startPos:startPos+N], n = N+M-1)
        C = A * FilterFR
        conv = np.real(np.fft.ifft(C))
        y[startPos:startPos+N+M-1] += conv[:N+M-1]
        startPos += N

    # OA for last frame (this part can be omitted)
    A = np.fft.fft(x[startPos:], n = N+M-1)
    C = A * FilterFR
    conv = np.real(np.fft.ifft(C))
    y[startPos:] += conv[:L+M-1-startPos]

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
    
    FilterFR = np.fft.fft(kernel, n = N)
    
    y = np.zeros(L)
    startPos = 0
    while startPos < L-N:
        A = np.fft.fft(x[startPos:startPos+N], n = N)
        C = A * FilterFR
        conv = np.real(np.fft.ifft(C))
        y[startPos:startPos + N-M+1] = conv[M-1:]
        startPos += N-M+1
    
    # OS for last frame (this part can be omitted)
    A = np.fft.fft(x[startPos:], n = N)
    C = A * FilterFR
    conv = np.real(np.fft.ifft(C))
    y[startPos:] = conv[M-1:M-1+L-startPos]

    return y
    

t = 0.6             # duration: 600 milliseconds
fs = 16000          # sampling frequency: 16000 Hz
a1 = 2              # amplitudes:
a2 = 3
a3 = 1

f1 = 900           # frequency components:
f2 = 1400
f3 = 6100

plot_samples = 1000


# initial signal (s1)
n = np.arange(fs*t)
s1 = a1 * np.sin( 2*np.pi*n*f1/fs ) + \
     a2 * np.sin( 2*np.pi*n*f2/fs ) + \
     a3 * np.sin( 2*np.pi*n*f3/fs )

kernelsize = 113
overlapsize = 400               # overlapsize + kernelsize - 1 = 512
cutoff = (f1 * 2.0) / fs

# lowpass filter design via firwin()
kernel = sig.firwin(kernelsize, cutoff)
w,h = sig.freqz(kernel)
filtered1 = OverlapAdd(s1, kernel, overlapsize)
filtered2 = OverlapSave(s1, kernel, overlapsize)


matplotlib.rcParams.update({'font.size': 8})

plt.figure(num="Filtering by OverlapAdd and OverlapSave")
plt.subplot(411)
plt.title("Initial signal")
plt.plot(s1[:plot_samples])
plt.subplot(412)
plt.title(("Signal filtered by Overlap-Add method"))
plt.plot(filtered1[:plot_samples])
plt.subplot(413)
plt.title(("Signal filtered by Overlap-Save method"))
plt.plot(filtered2[:plot_samples])
plt.subplot(414)
plt.title(("Frequency response of designed LP filter"))
plt.plot(np.abs(h))
plt.show()


# plot spectrograms "before and after"
plt.figure("Spectrograms")
plt.subplot(311)
plt.title("Initial signal")
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(312)
plt.title("Signal filtered by Overlap-Add method")
Pxx2, freqs2, bins2, im2 = specgram(filtered1, 256, fs)
plt.subplot(313)
plt.title("Signal filtered by Overlap-Save method")
Pxx3, freqs3, bins3, im3 = specgram(filtered2, 256, fs)
plt.show()
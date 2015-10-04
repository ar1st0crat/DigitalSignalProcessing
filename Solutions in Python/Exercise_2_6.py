# -*- coding: utf-8 -*-

"""
Write code that carries out FFT convolution of two signals. Plot the result.
Compare the result with your solution of exercise 1.4.
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

"""
User-defined function
Convolution
"""
def convolve(a,b):
    N = len(a)
    M = len(b)
    res = np.zeros(N + M - 1)
    
    # solution 1: check conditions each time
    # see correlate() function for solution 2
    for n in xrange(N+M-1):    
        for k in xrange(M):
            if n >= k and n - k < N:
                res[n] += a[n-k] * b[k]
    return res

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


# demo
x = np.array([1, 5, 3, 2, 6])
h = np.array([2, 3, 1])

conv1 = sig.convolve(x,h)
conv2 = convolve(x,h)
conv_fft = fft_convolve(x,h)

plt.figure("Convolution demo")
plt.subplot(311)
plt.stem(conv1)
plt.subplot(312)
plt.stem(conv2)
plt.subplot(313)
plt.stem(conv_fft)

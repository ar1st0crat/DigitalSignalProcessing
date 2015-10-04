# -*- coding: utf-8 -*-

"""
Write code that computes the convolution and cross-correlation of signals
and plots the results. Use signals from Example 2.4 and signal pairs {s1, s2}
and {s1, s3} as the input into your program. Explain the results. Compare
your results with the results of MATLAB functions conv() and xcorr().
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
Cross-correlation
"""
def correlate(a,b):
    N = len(a)
    M = len(b)
    res = np.zeros(N + M - 1)

    # solution 2: reserve new array with zeros at the beginning and at the end
    x = np.concatenate( (np.zeros(M-1), a, np.zeros(M-1)) )
    
    for n in xrange(N+M-1):    
        for k in xrange(M):
            res[n] += x[M-1 + n-k] * b[M-1 -k]
    return res


# ===================================== demo (for signals x and h only)
x = np.array([1, 5, 3, 2, 6])
h = np.array([2, 3, 1])

conv1 = sig.convolve(x,h)
conv2 = convolve(x,h)

plt.figure("Convolution demo")
plt.subplot(211)
plt.stem(conv1)
plt.subplot(212)
plt.stem(conv2)


xcorr1 = sig.correlate(x,h)
xcorr2 = correlate(x,h)

plt.figure("Cross-correlation demo")
plt.subplot(211)
plt.stem(xcorr1)
plt.subplot(212)
plt.stem(xcorr2)

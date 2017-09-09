# -*- coding: utf-8 -*-

"""
Write code that processes a segment of an input signal and evaluates the
following characteristics: 1) energy, 2) zero-cross rate, 3) mean, and 4)
variance. The program should allow setting the positions of the first and the
last samples of a signal segment to process.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

"""
User-defined function:
Sample mean
"""
def mean(a):
    return np.sum(a) / np.size(a)

"""
User-defined function:
(unbiased) sample variance
"""
def variance(a):
    mu = mean(a)
    return np.sum( (a - mu)**2 ) / np.size(a)

"""
User-defined function:
(unbiased) sample variance
"""
def variance_unbiased(a):
    mu = mean(a)
    return np.sum( (a - mu)**2 ) / (np.size(a) - 1)


"""
User-defined function:
Energy
"""
def energy(a):
    return np.sum( a**2 ) / np.size(a)

"""
User-defined function:
Zero-cross-rate
"""
def zero_cross_rate(a):
    signs = np.sign(a)          # normalize onto [-1,0,1]
    signs[signs == 0] = -1      # normalize onto [-1,1]

    return len(np.where(np.diff(signs))[0]) / float(np.size(a))
  

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

plt.figure(num="Initial signal, noise and their superposition")
plt.subplot(211)
plt.plot(s1[:plot_samples])


# noise (sn)
mu, sigma = 0, np.sqrt(max([a1,a2,a3]))
sn = np.random.normal(mu,sigma, t*fs)

plt.subplot(212)
plt.plot(sn[:plot_samples])


left = 200
right = 300

print "MEASUREMENTS FOR NOISY SIGNAL SN:\n"
print str.format("Mean (SciPy) = {0}\nMean (User-defined) = {1}\n", \
                    sp.mean( sn[left:right] ), mean(sn[left:right]))
print str.format("Variance (SciPy) = {0}\nVariance (User-defined) = {1}\n", \
                    sp.var( sn[left:right] ), variance(sn[left:right]))  
print str.format("Energy = {0}\n", energy(sn[left:right]) )
print str.format("ZeroCrossRate = {0}\n", zero_cross_rate(sn[left:right]) )

print "MEASUREMENTS FOR SIGNAL S1:\n"
print str.format("Mean (SciPy) = {0}\nMean (User-defined) = {1}\n", \
                    sp.mean( s1[left:right] ), mean(s1[left:right]))
print str.format("Variance (SciPy) = {0}\nVariance (User-defined) = {1}\n", \
                    sp.var( s1[left:right] ), variance(s1[left:right]))  
print str.format("Energy = {0}\n", energy(s1[left:right]) )
print str.format("ZeroCrossRate = {0}\n", zero_cross_rate(s1[left:right]) )
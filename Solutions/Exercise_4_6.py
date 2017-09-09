# -*- coding: utf-8 -*-

"""
Create low-pass, band-pass and band-reject filters with the coefficients defined in Eq.4.13-Eq.4.29 
and with the following parameters: fc=1200 Hz; BW=500 Hz. 
Write code to plot the filter frequency response and its zero-pole diagram.
Plot the spectrograms of a speech signal (loaded from file) before and after filtering
"""

import numpy as np
import scipy.io.wavfile as wave
import scipy.signal as sig
from pylab import specgram
import matplotlib.pyplot as plt
 
 
"""
User-defined function:
Plot zero-pole diagram and frequency response of the filter (b,a)
"""
def PlotFilterCharacteristics(b,a,figname):
    w, h = sig.freqz(b,a)
    z, p, k = sig.tf2zpk(b, a)

    plt.figure(num=figname)
    plt.subplot(211)
    plt.plot(np.real(z), np.imag(z), 'xb')
    plt.plot(np.real(p), np.imag(p), 'or')
    # plot unit circle
    phis=np.arange(0, 2*np.pi, 0.01)
    plt.plot( np.cos(phis), np.sin(phis), c='b')
    plt.grid()
    plt.xlim([-3,3])
    plt.ylim([-1.2, 1.2])
    plt.legend(['Zeros', 'Poles'], loc=2)

    plt.subplot(212)
    plt.plot(np.abs(h))
    

# initial signal
try:
    fs, s1 = wave.read("speech_sample.wav")
except IOError:
    print "Could not open speech_sample.wav file"
else:
    cutoff = 1200
    omega_cutoff = 2*np.pi*cutoff / fs
    Bw = 500
    
    
    # Four-pole LP filter
    b = [ (1 - np.exp(-14.445*cutoff/fs)) ** 4 ]
    a = [  1, \
          -4*np.exp(-14.445*cutoff/fs), \
           6*np.exp(-14.445*2*cutoff/fs), \
          -4*np.exp(-14.445*3*cutoff/fs), \
             np.exp(-14.445*4*cutoff/fs)]
    
    # get and plot LP filter characteristics
    PlotFilterCharacteristics(b,a,"LP Filter")
    
    # filter signal with LP
    filtered1 = sig.lfilter(b, a, s1)
    
    R = 1 - 3 * float(Bw) / fs
    K = (1 - 2*R*np.cos(omega_cutoff) + R**2) / (2 - 2*np.cos(omega_cutoff))
    
    b = [1-K, 2*(K-R)*np.cos(omega_cutoff), R**2 - K]
    a = [1, -2*R*np.cos(omega_cutoff), R**2]
    
    # get Band-Reject filter characteristics
    PlotFilterCharacteristics(b,a,"BP Filter")
    
    # filter signal with BP
    filtered2 = sig.lfilter(b, a, s1)
    
    b = [K, -2*K*np.cos(omega_cutoff), K]
    a = [1, -2*R*np.cos(omega_cutoff), R**2]
    
    # get Band-Reject filter characteristics
    PlotFilterCharacteristics(b,a,"BR Filter")
    
    # filter signal with BR
    filtered3 = sig.lfilter(b, a, s1)
    
    # plot spectrograms "before and after"
    plt.figure("Spectrograms")
    plt.subplot(411)
    specgram(s1, 256, fs)
    plt.subplot(412)
    specgram(filtered1, 256, fs)
    plt.subplot(413)
    specgram(filtered2, 256, fs)
    plt.subplot(414)
    specgram(filtered3, 256, fs)
    plt.show()
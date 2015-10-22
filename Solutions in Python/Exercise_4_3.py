# -*- coding: utf-8 -*-

"""
Write your own code to design low-pass FIR-filter with custom order and cutoff frequency using Window method.
Plot impulse response and step response of the designed filter.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


f1 = 900
fs = 16000

cutoff = float(f1)

FRsize = 512
kernelsize = 93

# generate ideal frequency response for LP filter with cutoff
FR = np.zeros(FRsize)
FR[:FRsize*cutoff/fs] = 1.0
FR[-FRsize*cutoff/fs:] = 1.0

# compute kernel: inverse FFT of ideal filter response
IR = np.real(np.fft.ifft(FR))

# shift and truncate the kernel
kernel = np.append( IR[-kernelsize/2:], IR[:kernelsize/2] )

# apply window
kernelWindowed = kernel * np.hamming(kernelsize)

# compute real frequency response from the windowed kernel
realFR = np.abs(np.fft.fft(kernelWindowed, n = FRsize))

# step response
step = np.cumsum(kernelWindowed)


matplotlib.rcParams.update({'font.size': 8})

plt.figure("Ideal filter")
plt.subplot(321)
plt.title("Ideal frequency response")
plt.ylim([-0.05, 1.05])
plt.plot(FR[:FRsize/2])
plt.subplot(322)
plt.title("Hamming window")
plt.plot(np.hamming(kernelsize))
plt.subplot(323)
plt.title("Truncated kernel")
plt.stem(kernel)
plt.subplot(324)
plt.title("Truncated and windowed kernel")
plt.stem(kernelWindowed)
plt.subplot(325)
plt.title("Step response")
plt.stem(step)
plt.subplot(326)
plt.title("Real frequency response")
plt.plot(realFR[:FRsize/2])
plt.show()
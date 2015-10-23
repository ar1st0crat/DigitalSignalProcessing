# -*- coding: utf-8 -*-

"""
In this task you will investigate the effect of windowing. Use 1024 samples
for FFT. Write code to do the following:
    1. Compute the DFT of the signal x(n) = cos(πn/4). Use sampling frequency
       fs = 4096 Hz. Plot the magnitude and phase spectrum of x(n).
    2. Truncate x(n) using Hamming window of size 93. Plot its magnitude and
       phase spectrum. Obtain the 512-point signal z[n] by zero-padding x[n].
       Plot its magnitude and phase spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
  

fs = 4096          # sampling frequency: 4096 Hz
f = fs/8           # frequency component

plot_samples = 100

FFT_size = 1024
window_size = 93

# =========================================== Part 1
# initial signal x[n] = cos(pi*n/4)
n = np.arange(fs)
s = np.cos( 2*np.pi*n*f/fs )
spec = np.fft.fft(s,FFT_size)
mag_spec = np.abs(spec)             # magnitude spectrum
phase_spec = np.angle(spec)         # phase spectrum

plt.figure(num="Initial signal and its spectrum")
plt.subplot(311)
plt.plot(s[:plot_samples])

# spectrum of s (FFT_size samples taken)
# plot only the 1st half of spectrum (since it's symmetric)
plt.subplot(312)
plt.xlabel("Freq (Hz)")
plt.xlim(0,FFT_size/2)
plt.plot(mag_spec[:FFT_size/2])

plt.subplot(313)
plt.xlabel("Freq (Hz)")
plt.xlim(0,FFT_size/2)
plt.plot(phase_spec[:FFT_size/2])

plt.show()

# ============================================ Part 2
weighted = s[:window_size] * np.hamming(window_size)

ext = np.ceil(np.log2(window_size))     # compute closest power of 2 for window_size
FFT_size = int(np.power(2,ext))

spec = np.fft.fft(weighted,FFT_size)
mag_spec = np.abs(spec)             # magnitude spectrum
phase_spec = np.angle(spec)         # phase spectrum

plt.figure(num="Weighted signal and its spectrum")
plt.subplot(311)
plt.plot(weighted)

# spectrum of s (FFT_size samples taken with zero-padding)
# plot only the 1st half of spectrum (since it's symmetric)
plt.subplot(312)
plt.xlabel("Freq (Hz)")
plt.xlim(0,FFT_size/2)
plt.plot(mag_spec[:FFT_size/2])

plt.subplot(313)
plt.xlabel("Freq (Hz)")
plt.xlim(0,FFT_size/2)
plt.plot(np.unwrap(phase_spec[:FFT_size/2]))

plt.show()


weighted = s[:window_size] * np.hamming(window_size)

FFT_size = 512
spec = np.fft.fft(weighted,FFT_size)
mag_spec = np.abs(spec)             # magnitude spectrum
phase_spec = np.angle(spec)         # phase spectrum

plt.figure(num="Weighted signal and its spectrum for 512-point FFT")
plt.subplot(311)
plt.plot(weighted)

# spectrum of s (fs samples taken)
# plot only the 1st half of spectrum (since it's symmetric)
plt.subplot(312)
plt.xlabel("Freq (Hz)")
plt.xlim(0,FFT_size/2)
plt.plot(mag_spec[:FFT_size/2])

plt.subplot(313)
plt.xlabel("Freq (Hz)")
plt.xlim(0,FFT_size/2)
plt.plot(np.unwrap(phase_spec[:FFT_size/2]))

plt.show()
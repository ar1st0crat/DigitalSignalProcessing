# -*- coding: utf-8 -*-
import numpy as np
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

plot_samples = 500

# =========================================== Part 1
# initial signal (s1)
plt.figure(num="Initial signal")
plt.subplot(211)

n = np.arange(fs*t)
s1 = a1 * np.sin( 2*np.pi*n*f1/fs ) + \
     a2 * np.sin( 2*np.pi*n*f2/fs ) + \
     a3 * np.sin( 2*np.pi*n*f3/fs )

plt.plot(s1[:plot_samples])

# spectrum of s1 (fs samples taken)
# plot only the 1st half of spectrum (since it's symmetric)
plt.subplot(212)

spec = np.abs(np.fft.fft(s1,fs))

plt.xlabel("Freq (Hz)")
plt.xlim(0,fs/2)
plt.plot(spec[:fs/2])
plt.show()


# resampled signal (s4) at 11025 Hz
fs_new = 11025
s4 = a1 * np.sin( 2*np.pi*n*f1 / fs_new ) + \
     a2 * np.sin( 2*np.pi*n*f2 / fs_new ) + \
     a3 * np.sin( 2*np.pi*n*f3 / fs_new )

plt.figure(num="Initial signal sampled at 11025 Hz")
plt.subplot(211)

plt.plot(s4[:plot_samples])

# spectrum of s1 (fs samples taken)
# plot only the 1st half of spectrum (since it's symmetric)
spec = np.abs(np.fft.fft(s4,fs_new))

plt.subplot(212)
plt.xlabel("Freq (Hz)")
plt.xlim(0,fs/2)                # set x limits the same as for s1 (to compare frequency peaks)
plt.plot(spec[:fs_new/2])
plt.show()


# noise (sn)
mu, sigma = 0, np.sqrt(max([a1,a2,a3]))
sn = np.random.normal(mu,sigma, t*fs)

plt.figure(num="Noisy signal")
plt.subplot(211)
plt.plot(sn[:plot_samples])

# spectrum of sn (fs samples taken)
spec = np.abs(np.fft.fft(sn,fs))

plt.subplot(212)
plt.plot(spec)
plt.show()


# ============================================= Part 2
# inverse fft; compare s1 with the result of ifft of s1 spectrum
spec = np.fft.fft(s1,fs)
s1_ifft = np.real(np.fft.ifft(spec,fs))

plt.figure(num="Inverse FFT")
plt.subplot(311)
plt.plot( s1_ifft[:plot_samples])
plt.subplot(312)
plt.plot( s1[:plot_samples])
plt.subplot(313)
plt.ylim(-1,1)
plt.plot( s1[:plot_samples] - s1_ifft[:plot_samples])
plt.show()


# =============================================== Part 3
# plot spectrograms of s1 and sn
plt.figure("Spectrograms")
plt.subplot(211)
Pxx1, freqs1, bins1, im1 = specgram(s1, 256, fs)
plt.subplot(212)
Pxx2, freqs2, bins2, im2 = specgram(sn, 256, fs)
plt.show()

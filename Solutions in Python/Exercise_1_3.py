# -*- coding: utf-8 -*-

"""
In this task you’ll programmatically generate digital audio signals. Write
code to do the following:
    1. Generate signal s1 consisting of three sinusoids with frequencies f1, f2
       and f3, respectively. The signal has to be 600 milliseconds long, and it
       should be sampled with sampling frequency fs. The values of the
       frequencies are given in table 1 .1.
    2. Generate noise signal sn 600 milliseconds long.
    3. Superimpose signals s1 and sn (get the “noisy” version of s1). The
       resulting signal will be s2.
    4. Generate signal s3 by shifting s1 by 300 samples.
    5. Generate signal s4. It must be identical to signal s1 except that
       it has to be sampled at frequency fs4 = 11025 Hz.
    6. Plot the first 500 samples of each signal.
    7. Save each signal to WAVE file (set number of channels equal to 1).
    8. Load each signal from corresponding WAVE file. Listen to signal s1 and s4.
       Explain the difference.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


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
plt.subplot(311)
plt.plot(s1[:plot_samples])


# noise (sn)
mu, sigma = 0, np.sqrt(max([a1,a2,a3]))
sn = np.random.normal(mu,sigma, t*fs)

plt.subplot(312)
plt.plot(sn[:plot_samples])


# noisy signal (s2)
s2 = s1 + sn

plt.subplot(313)
plt.plot(s2[:plot_samples])


# shift signal
s3 = np.zeros(fs*t)
s3[300:] = s1[:-300]

plt.figure("Shifted signal (300 samples)")
plt.plot(s3[:plot_samples])


# resampled signal
fs_new = 11025
s4 = a1 * np.sin( 2*np.pi*n*f1/fs_new ) + \
     a2 * np.sin( 2*np.pi*n*f2/fs_new ) + \
     a3 * np.sin( 2*np.pi*n*f3/fs_new )

plt.figure("Resampled signal at 11025 Hz")
plt.plot(s4[:plot_samples])    


# save s1 and s4 signals into wave files
wav_x = np.int16(s1 / max([a1,a2,a3]) * 16384)
wav.write(r'd:\test1.wav', fs, wav_x)

wav_resampled = np.int16(s4 / max([a1,a2,a3]) * 16384)
wav.write(r'd:\test2.wav', fs_new, wav_resampled)

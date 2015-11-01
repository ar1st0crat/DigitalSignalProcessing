# -*- coding: utf-8 -*-

"""
Write code to do the following:
1. Resample a signal at new rate fsNew = 12 kHz using polyphase filters.
2. Plot frequency response of each polyphase filter.
"""

import numpy as np
import scipy.signal as sig
from pylab import specgram
import matplotlib.pyplot as plt
    

t = 0.6             # duration: 600 milliseconds
fs = 16000          # sampling frequency: 16000 Hz
fsNew = 12000
M = 4
L = 3

a1 = 2              # amplitudes:
a2 = 3
a3 = 1

f1 = 900           # frequency components:
f2 = 1400
f3 = 6100

plot_samples = 300


# initial signal (s1)
n = np.arange(fs*t)
x = a1 * np.sin( 2*np.pi*n*f1/fs ) + \
    a2 * np.sin( 2*np.pi*n*f2/fs ) + \
    a3 * np.sin( 2*np.pi*n*f3/fs )


h = sig.firwin(96, 0.23)       # 96 is multiple of 4 and 3

# ----------------------------------------------------- poyphase decimation
h1 = h[::M]
h2 = h[1::M]
h3 = h[2::M]
h4 = h[3::M]

x1 = np.concatenate((x[::M], [0]))
x2 = np.concatenate(([0],x[3::M]))
x3 = np.concatenate(([0],x[2::M]))
x4 = np.concatenate(([0],x[1::M]))

y1 = sig.lfilter(h1, [1], x1)
y2 = sig.lfilter(h2, [1], x2)
y3 = sig.lfilter(h3, [1], x3)
y4 = sig.lfilter(h4, [1], x4)

y = y1 + y2 + y3 + y4

# ------------------------------------------------ plot polyphase filters PR
w1, H1 = sig.freqz(h1)
w2, H2 = sig.freqz(h2)
w3, H3 = sig.freqz(h3)
w4, H4 = sig.freqz(h4)

plt.figure("Filters")
plt.subplot(221)
plt.plot(np.unwrap(np.angle(H1)))
plt.subplot(222)
plt.plot(np.unwrap(np.angle(H2)))
plt.subplot(223)
plt.plot(np.unwrap(np.angle(H3)))
plt.subplot(224)
plt.plot(np.unwrap(np.angle(H4)))

# ------------------------------------------------- polyphase interpolation
h = sig.firwin(96, 0.32)

h1 = h[::L]
h2 = h[1::L]
h3 = h[2::L]

y1 = L * sig.lfilter(h1, [1], y)
y2 = L * sig.lfilter(h2, [1], y)
y3 = L * sig.lfilter(h3, [1], y)

y = np.zeros(len(y1)+len(y2)+len(y3))
y[::L] = y1
y[1::L] = y2
y[2::L] = y3
# --------------------------------------------------------------------------

# plot spectrograms "before and after"
plt.figure("Before and after polyphase filtering")
plt.subplot(221)
specgram(x, 256, fs)
plt.subplot(222)
plt.plot(x[:plot_samples])
plt.subplot(223)
specgram(y, 256, fsNew)
plt.subplot(224)
plt.plot(y[:plot_samples*float(fsNew)/fs])


y = sig.resample(x, fsNew*t)

# plot spectrograms "before and after"
plt.figure("Resample() result")
plt.subplot(221)
specgram(x, 256, fs)
plt.subplot(222)
plt.plot(x[:plot_samples])
plt.subplot(223)
specgram(y[96:], 256, fsNew)
plt.subplot(224)
plt.plot(y[:plot_samples*float(fsNew)/fs])
plt.show()


'''
# Efficient polyphase decomposition (see pdf file)
e0 = h[::2]
e1 = h[1::2]
E0 = np.reshape(e0, 3, len(e0) / 3);
E1 = np.reshape(e0, 3, len(e1) / 3);

# Down-sampling
u1 = x
u0 = np.concatenate([0], x[:-1])

u00d = u0[::3]
u01d = np.concatenate([0], u0[2::3])
u02d = np.concatenate([0], u0[1::3])

u10d = u1[::3]
u11d = np.concatenate([0], u1[2::3])
u12d = np.concatenate([0], u0[1::3])

# Polyphase filtering
x00 = np.lfilter(2*E0[0,:], [1], u00d)
x01 = np.lfilter(2*E0[1,:], [1], u01d)
x02 = np.lfilter(2*E0[2,:], [1], u02d)

x10 = np.lfilter(2*E1[0,:], [1], u10d)
x11 = np.lfilter(2*E1[1,:], [1], u11d)
x12 = np.lfilter(2*E1[2,:], [1], u12d)

yy0 = x00 + x01 + x02
yy1 = x10 + x11 + x12

y0ii = np.zeros(2*len(yy0))
y1ii = np.zeros(2*len(yy1))

y0ii[::2] = yy0                           # Up-sampled signal
y1ii[::2] = yy1                           # Up-sampled signal

y1 = np.concatenate([0], y1ii[:-1])       

y = y0ii + y1                             # Output signal
'''
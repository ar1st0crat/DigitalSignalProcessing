# -*- coding: utf-8 -*-

"""
Write code to plot:
    a) frequency response
    b) poles and zeros of transfer function
of each of the filters given in exercises 3.2 and 4.2:

1) y[n]=0.41x[n] + 0.8y[n-1] - 0.24y[n-2] + 0.032y[n-3] - 0.002y[n-4]
2) y[n]=0.93x[n] – 0.93x[n-1] + 0.86y[n-1]
3) y[n]=0.32x[n] + 0.68y[n-1]

1) y[n] = 0.36x[n] + 0.22x[n-1] – 0.85x[n-2]
2) y[n] = 0.76x[n] + 0.32x[n-1] + 0.15y[n-1]
3) y[n] = x[n] – x[n-5]
4) y[n] = 0.8x[n] – 0.2y[n-1] – 0.3y[n-2] + 0.8y[n-3]
5) y[n] = x[n] – x[n-2] + 0.9y[n-1] – 0.6y[n-2]

Compare figures with your analytic solutions of exercises 3.2 and 4.2.
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

b1 = [0.41]
a1 = [1, -0.8, 0.24, -0.032, 0.002]
w1,h1 = sig.freqz(b1,a1)

b2 = [0.93, -0.93]
a2 = [1, -0.86]
w2,h2 = sig.freqz(b2,a2)

b3 = [0.32]
a3 = [1, -0.68]
w3,h3 = sig.freqz(b3,a3)

plt.figure(num="Filters from ex.3.2")
plt.subplot(321)
plt.title("Magnitude response")
plt.plot(np.abs(h1))
plt.subplot(322)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h1)))
plt.subplot(323)
plt.title("Magnitude response")
plt.plot(np.abs(h2))
plt.subplot(324)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h2)))
plt.subplot(325)
plt.title("Magnitude response")
plt.plot(np.abs(h3))
plt.subplot(326)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h3)))


b = [0.36, 0.22, -0.85]
a = [1]
w,h = sig.freqz(b,a)

plt.figure(num="Filter1 from ex.4.2")
plt.subplot(211)
plt.title("Magnitude response")
plt.plot(np.abs(h))
plt.subplot(212)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h)))

b = [0.76, 0.32]
a = [1, -0.15]
w,h = sig.freqz(b,a)

plt.figure(num="Filter2 from ex.4.2")
plt.subplot(211)
plt.title("Magnitude response")
plt.plot(np.abs(h))
plt.subplot(212)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h)))

b = [1, 0, 0, 0, 0, -1]
a = [1]
w,h = sig.freqz(b,a)

plt.figure(num="Filter3 from ex.4.2")
plt.subplot(211)
plt.title("Magnitude response")
plt.plot(np.abs(h))
plt.subplot(212)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h)))

b = [0.8]
a = [1, 0.2, 0.3, -0.8]
print np.roots(a)
w,h = sig.freqz(b,a)

plt.figure(num="Filter4 from ex.4.2")
plt.subplot(211)
plt.title("Magnitude response")
plt.plot(np.abs(h))
plt.subplot(212)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h)))

b = [1, 0, -1]
a = [1, -0.9, 0.6]
w,h = sig.freqz(b,a)

plt.figure(num="Filter5 from ex.4.2")
plt.subplot(211)
plt.title("Magnitude response")
plt.plot(np.abs(h))
plt.subplot(212)
plt.title("Phase response (unwrapped)")
plt.plot(np.unwrap(np.angle(h)))
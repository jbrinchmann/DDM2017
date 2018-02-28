#!/usr/bin/env python

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.units.imperial as ui
from sklearn.preprocessing import StandardScaler

t = Table().read('height-weight.vot')

h = t['Height_inches_']*ui.inch
w = t['Weight_pounds_']*ui.pound

# First make a scatter plot showing it zoomed in.
plt.scatter(h.to(u.meter), w.to(u.kilogram))
plt.xlabel('Height [m]')
plt.ylabel('Weight [kg]')
plt.savefig("H-W-m-kg-zoom.pdf")
plt.close()


# Then use equal size axes
plt.scatter(h.to(u.meter), w.to(u.kilogram))
plt.xlabel('Height [m]')
plt.ylabel('Weight [kg]')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.savefig("H-W-m-kg-equal.pdf")
plt.close()


# Next, normalise.
X = np.vstack([w.value, h.value]).T
Xstd = StandardScaler().fit_transform(X)
plt.scatter(Xstd[:, 0], Xstd[:, 1])
plt.xlabel('Height [scaled]')
plt.ylabel('Weight [scaled]')
plt.savefig('H-W-scaled.pdf')
plt.close()


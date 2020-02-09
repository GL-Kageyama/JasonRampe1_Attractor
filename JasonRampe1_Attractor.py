#===================================================================================================
#-------------------------------      Jason Rampe 1 Attractor     ----------------------------------
#===================================================================================================

#------------------------     new_X = cos(b * Y) + c * sin(b * X)     ------------------------------
#------------------------     new_Y = cos(a * X) + d * sin(a * Y)     ------------------------------

#===================================================================================================

import numpy as np
import pandas as pd
import panel as pn
import datashader as ds
from numba import jit
from datashader import transfer_functions as tf

#===================================================================================================

@jit(nopython=True)
def JasonRampe1_trajectory(x0, y0, n, a, b, c, d):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    
    for i in np.arange(n-1):

        x[i+1] = np.cos(b*y[i]) + c*np.sin(b*x[i])
        y[i+1] = np.cos(a*x[i]) + d*np.sin(a*y[i])
    
    return x, y

#===================================================================================================

def JasonRampe1_plot(x0=0, y0=0, n=100000, a=2.6, b=-2.6, c=-0.6, d=0.7, cmap=["yellow", "orange"]):
    
    cvs = ds.Canvas(plot_width=700, plot_height=700)
    x, y = JasonRampe1_trajectory(x0, y0, n, a, b, c, d)
    agg = cvs.points(pd.DataFrame({'x':x, 'y':y}), 'x', 'y')
    
    return tf.shade(agg, cmap)

#===================================================================================================

pn.extension()
pn.interact(JasonRampe1_plot, n=(1,1000000))

#===================================================================================================

# The value of this attractor can be changed freely. Try it in the jupyter notebook.


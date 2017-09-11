from JBMath.PDF import PDF
import numpy as np

class GaussianPDF(PDF):
    """A Gaussian PDF
    """

    def __init__(self, mean, sd, xrange=None):

        if xrange is None:
            xrange = np.linspace(mean-7*sd, mean+7*sd, 1000)

        z = (xrange-mean)/sd
        pdf = np.exp(-z*z/2.0)/(np.sqrt(2*np.pi)*sd)
            
        PDF.__init__(self, xrange, pdf)




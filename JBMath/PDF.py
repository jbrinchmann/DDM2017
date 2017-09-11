from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

class PDF(object):
    """A class to encapsulate a probability distribution function

    It only supports 1D PDFs. 

    """

    def __init__(self, x, pdf, name="Generic PDF"):
        """Define the object

        """
        self.x = x
        self.pdf = pdf
        self.name = name
        self.__ecdf = None # Calculated on need
        self.__entropy = None
        self.__bimodality = None
        self.normalised = None
        self.__norm = None
        self.__moment = [None, None, None, None, None]
        self.central_moment = [None, None, None, None, None]

    def reset(self):
        self.ecdf = None # Calculated on need
        self.__entropy = None
        self.__bimodality = None
        self.normalised = None
        self.__norm = None
        self.__moment = [None, None, None, None, None]
        self.central_moment = [None, None, None, None, None]
        
        
    def norm(self, recalculate=False):
        if self.__norm is None or recalculate:
            self.__norm = np.trapz(self.pdf, self.x)
        return self.__norm
    
    def normalise(self):
        """
        Normalise the PDF to integral 1.
        """

        print("Normalising the PDF")
        self.pdf = self.pdf/self.norm()
        self.normalised = True

        # Reset the ECDF
        self.__ecdf = None

    def ensure_normalised(self):
        if not self.normalised:
            self.normalise()

    def ecdf(self):
        """
        Return the empricial distribution function

        If necessary, calculate this. It does assume that the
        x axis is ordered and does not check for this.

        The ECDF is calculated using a cumulative sum, rather than
        integration so can differ a bit from exact integration if 
        the PDF is coarsely sampled. 
        """

        if self.__ecdf is None:
            ecdf = np.cumsum(self.pdf)
            # Correct for the bin size.
            ecdf[1:] = ecdf[1:]*np.diff(self.x)
            self.__ecdf = ecdf

        return self.__ecdf

    def moment(self, n):
        """
        Calculate the nth moment of a PDF. Only 0-4 are defined. These
        are defined in general as 
   
        $$mu_n = \int x^n pdf(x)$$
   
        In general these are not very useful except the first which is
        the mean. 
    
        We cache these values and use a flag array to keep track of
        whether they have been calculated or not.
        """
        if (n>4) | (n<0):
            print("moment: We do not support moments with n>4 or n<0.")
            return None
        
        if self.__moment[n] is None:
            # We need to calculate it.
            moment = np.trapz(self.x**n*self.pdf, self.x)
            self.__moment[n] = moment

        return self.__moment[n]


    def moments(self):
        return [self.moment(n) for n in range(0, 5)]

    def centralised_moment(self, n):
        """
        Calculate the centralised nth moment of a PDF. Only 0-4 are defined. These
        are defined in general as 
   
        $$mu_n = \int (x-mean(x))^n pdf(x)$$
   
        We cache these values as for the normal moments.
        """
    
        if (n>4) | (n<0):
            print("moment: We do not support moments with n>4 or n<0.")
            return None

        # We only use normalised PDFs when running this.
        self.ensure_normalised()

        mean = self.mean()

        mom = np.trapz((self.x-mean)**n*self.pdf, self.x)

        self.central_moment[n] = mom

        return self.central_moment[n]
        
    def centralised_moments(self):
        return [self.centralised_moment(n) for n in range(0, 5)]
    
    def standardised_moment(self, n):
        """
        Return a standardised moment - ie. one normalised by the
        standard deviation to an appropriate power.
        """

        sigma = self.standard_deviation()
        return self.centralised_moment(n)/sigma**n


    def mean(self):
        """Calculate the mean of a PDF.
        """
        return self.moment(1)


    def standard_deviation(self):
        """Calculate the standard deviation of a PDF.
        """

        return np.sqrt(self.centralised_moment(2))
    

    def skewness(self):
        """
        Calculate the skewness of a PDF. This is the 3rd standardised
        moment of the PDF 
        """
    
        return self.standardised_moment(3)


    def kurtosis(self):
        """
        Calculate the kurtosis of a PDF. This is the 4th standardised
        moment of the PDF.
        """
   
        return self.standardised_moment(4)

    
    def plot(self, ax=None):
        """
        A simple plot of the PDF
        """

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.x, self.pdf)

        return ax

    def entropy(self, axis=-1):
        """
        Calculate the entropy of a PDF
        """

        if self.__entropy is None:
            ok, = np.where(self.pdf > 0)

            if len(ok) > 0:
                integrand = -self.pdf[ok]*np.log(self.pdf[ok])/np.log(2.0)
                entropy = np.trapz(integrand, self.x[ok], axis=axis)
            else:
                print("entropy: No valid points in the PDF")
                entropy = 0.0

            self.__entropy = entropy

        return self.__entropy
    

    def bimodality_coefficient(self):
        """This returns Sarle's bimodality coefficient based on the  Wikipedia definition.
   
        $$\beta = (\gamma^2+1)/\kappa,$$
   
        where $\gamma = skewness, \kappa = kurtosis$

        A uniform distribution has beta = 5/9, values larger might indicate bimodality

        See http://en.wikipedia.org/wiki/Bimodal_distribution for details
        """

        if self.__bimodality is None:
            beta = (self.skewness()**2+1.0)/self.kurtosis()
            self.__bimodality = beta

        return self.__bimodality


    def mode(self):
        """Return the mode (peak) of the PDF"""

        i_max = np.argmax(self.pdf)

        return self.x[i_max]


    def info(self):
        """
        Print basic info on the PDF.
        """
        
        print("{0}".format(self.name))
        print("----------------------")
        print("        Mean = {0}".format(self.mean()))
        print("        Stdv = {0}".format(self.standard_deviation()))
        print("        Kurt = {0}".format(self.kurtosis()))
        print("        Skew = {0}".format(self.skewness()))
        print("        Mode = {0}".format(self.mode()))
        print("     Entropy = {0}".format(self.entropy()))
        print("  Bimodality = {0}".format(self.bimodality_coefficient()))
                
        
    
    # The functions below modify the PDF.
    def zero_center(xrange=None, delta_x=None):
        """Shift the PDF to be centred on zero

        This routine takes the PDF and shifts it so that it has zero
        center and cover xrange with the same sampling as existing x
        unless delta_x is given (which is a good idea if the sampling of
        the PDF is irregular)

        """
   
        xc = self.mean()
        # We will preseve the normalisation
        norm = self.norm()

        # Figure out the output axis.
        if xrange is None:
            # Default to an output axis that is the original
            # shifted to have zero mean
            xnew = self.x - xc
        else:
            # The user provided the xrange.
            if delta_x is None:
                delta_x = np.median(np.diff(self.x))

            xnew = np.arange(xrange[0], xrange[1], delta_x)

        xshift = self.x-xc
        pdf_new = np.interp(xnew, xshift, self.pdf)
        outside, = np.where((xnew < min(xshift)) | (xnew > max(xshift)))

        if len(outside) > 0:
            pdf_new[outside] = 0.0

        # Conserve norm
        new_norm = np.trapz(pdf_new, xnew)

        pdf_new = pdf_new*norm/new_norm

        # Redefine the PDF.
        self.pdf = pdf_new
        self.x = xnew
            

    def convolve_gaussian(self, sigma):
        """
        Convolve the PDF with a Gaussian with standard deviation SIGMA. 
        """

        x = self.x
        y = self.pdf

        # Get sigma in pixel units.
        sig_pix = sigma/(x[1]-x[0])
        
        y = gaussian_filter1d(y, sig_pix, mode='wrap')

        self.pdf = y
        self.reset()
        # print("Gaussian convolution is not yet implemented!")


    #------------------------------------------------------
    # Divergences
    #
    # I could probably split this into separate functions.
    #------------------------------------------------------
    def kl_divergence(self, pdf2):
        """
        Calculate the Kullback-Leibler divergence between two PDFs. 
        """

        return self.divergence(pdf2, type='KL')


    def divergence(self, pdf2, div_type, no_interpolate=False,
                       alpha=0.5, beta=0.5):
        """
        Calculate a divergence between two PDFs. 
        
        Supported divergences:

        Kullback-Leibler (KL)
        Hellinger
        Jeffrey
        Chernoff_alpha
        Exponential
        Kagan
        Product
        
        Resistor
        Sum?
        """

        if no_interpolate:
            #
            # The simplest case - the PDFs are on the same x-axis
            #
            p = self.pdf
            q = pdf2.pdf
            x = self.x
        else:
            #
            # Now we need to put the PDFs on a common axis. This is
            # potentially tricky so here we use a conservative approach -
            # we create a final array that has sampling equal to the
            # minimum sampling in the two PDFs (assuming _equal_ sampling)
            # and spanning from min([x_1, x_2]) to max([x_1, x_2]) and then
            # interpolate both PDFs onto this grid.
            #
            x1 = self.x
            p1 = self.pdf
            x2 = pdf2.x
            p2 = pdf2.pdf

            xall = np.hstack((x1, x2))

            diff1 = np.diff(x1)
            diff2 = np.diff(x2)
            min_dx = np.min(np.hstack([diff1, diff2]))

            xr = [np.min(xall), np.max(xall)]

            x = np.linspace(xr[0], xr[1], (xr[1]-xr[0])/min_dx)

            p = np.interp(x, x1, p1, left=0.0, right=0.0)
            q = np.interp(x, x2, p2, left=0.0, right=0.0)


        # We need the smallest value also
        m = np.finfo(np.double)
        eps = m.tiny

        if (div_type == 'KL') | (div_type == 'Kullback-Leibler'):
            #
            # Kullback-Leibler divergence
            #
            arg = p*(np.log(p)-np.log(q.clip(min=eps)))


#            print(np.mean(arg), np.min(arg), np.max(arg))
#            print(x[0], x[-1], arg[0], arg[-1])
            
            # Due to finite precision we might have cases where p(x)
            # or q(x) are zero. Thus we need to decide how to deal
            # with this. For the case of p(x) = 0 I set the integrand
            # to zero which should be correct unless q(x) grows
            # exponentially since lim x->0 of x ln(x) = 0
            #
            # For the case where q(x) goes to zero however it is not
            # clear what the right approach is - as a general rule the
            # distance should grow quickly. Thus in that case q(x) is
            # set to the smallest value (machar(/double).xmin)
            # distinguished by the compiler. It is NOT obvious that
            # this is a good idea!
            #
            zero_p = np.where(np.logical_not(np.isfinite(arg)))
            arg[zero_p] = 0.0

        elif div_type == 'Hellinger':
            #
            # Squared Hellinger distance.
            #
            arg = 2.0*(sqrt(p)-sqrt(q))^2
        elif div_type == 'Jeffrey':
            # The Jeffrey's divergence. This is a symmetric version of
            # the Kullback-Leibler divergence and as such has the same
            # considerations wrt. zero values as that.
            # 
            # Rather than split the integration into various argument
            # regions, however, I am instead thresholding p & q at
            # minimum x in the logarithmic arguments
            #

            arg = (p - q)*(np.log(p.clip(min=eps))-np.log(q.clip(min=eps)))

        elif div_type  == 'Chernoff_alpha':
            #
            # This is an important distance for classification problems but
            # it should really be maximised over alpha - somewhat
            # inconvenient in practice
            #
            
            pre_chernoff = 4.0/(1.0-alpha**2)
            arg = np.pow(p, ((1.-alpha)/2.)*q**((1+alpha)/2.))

        elif div_type == 'Exponential':
            #
            # The exponential divergence - similar to K-L but with a
            # quadratic term.
            #
            arg = p*(alog(p) - alog(q.clip(min=eps)))**2

            zero_p = np.where(np.logical_not(np.isfinite(arg)))
            arg[zero_p] = 0.0
        elif div_type == 'Kagan':
            #
            # Kagan's divergence. This is ill-defined where p == 0
            # so we do not tackle that situation in any proper way
            #
            arg = 0.5*(p-q)**2/p
        elif div_type == 'Product':
            prefactor = 2.0/((1-alpha)*(1-beta))
            arg = prefactor*(1.0-(q/p)**((1-alpha)/2.))*\
              (1-(q/p)**((1-beta)/2.0))*p
      
            zero_p = np.where(np.logical_not(np.isfinite(arg)))
            arg[zero_p] = 0.0
        elif div_type == 'Resistor':
            #
            # The resistor divergence defined in
            # www.ece.rice.edu/~dhj/resistor.pdf 
            #
            # This differs from the others because it is the sum of
            # two distances.
            #
            d1 = self.divergence(pdf2, div_type='KL',
                                     no_interpolate=no_interpolate)
            d2 = pdf2.divergence(self, div_type='KL',
                                     no_interpolate=no_interpolate)
      
            invD = 1.0/d1+1.0/d2
            D = 1.0/invD

        if div_type != 'Resistor':
            D = np.trapz(arg, x)

        if div_type == 'Chernoff_alpha':
            D = pre_chernoff*(1.0-D)

        return D

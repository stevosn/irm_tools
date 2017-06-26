#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# - Author: steve simmert
# - E-mail: steve.simmert@fremiste.de
# - Copyright: 2017
# """
from os.path import join

from scipy import arange, array, float64, median
from scipy import histogram, logical_and, percentile
from scipy import logical_or, meshgrid

from matplotlib import pyplot as plt
from matplotlib import cm

from lmfit import report_fit
from lmfit.models import GaussianModel, ConstantModel

from PIL import Image

from .plotting.plotting import add_plot_to_figure, get_gridded_axis


class MT_SNR(object):
    """
    Measure microtubule signal, noise and their ratio.
    """
    def __init__(self, directory, file,
                 pixel_size=1.0, rsl=slice(None), csl=slice(None)):
        """
        Load microtubule snippet.
        
        The image is expected to be a tiff image, but other formats might also
        work. The MT is expected to be oriented from left to right, without
        curves, i.e. it should stay at a fixed row index.
        
        Arguments
        ---------
        directory : str
        file : str
        pixel_size : float
            Pixel size in length_unit per pixel.
        rsl : slice
            Row slice which selects the rows to be loaded.
        csl : slice
            Column slice which selects the columns to be loaded.
        """
        im = Image.open(join(directory, file))

        ncols, nrows = im.size
        
        self.ima = array(im.getdata()).reshape((nrows, ncols))[rsl, csl]
        
        self.directory = directory
        self.filename = file

        self.nrows, self.ncols = self.ima.shape
        # first axis is along x coordinate (47 values)
        # second axis is along y coordinate (60 values)

        self.pixel_size = pixel_size

        self.x = arange(0, self.ncols * pixel_size, pixel_size)
        self.y = arange(0, self.nrows * pixel_size, pixel_size)

    @property
    def med_profile(self):
        """ Return median profile along the MT axis. """
        return median(self.ima, axis=1)
        
    def fit_median_profile(self, A=None, s=None, x0=None, c=None):
        """
        Fit median profile to a Gaussian.
        
        A Gaussian with offset is fittet to the profile gotten by med_profile.
        
        Arguments
        ---------
        A : float
            Amplitude
        s : float
            Sigma = standard deviation of the Gaussian.
        x0 : float
            Center of the Gaussian.
        c : float
            Constant offset of the Gaussian.
        """
        med_prof = self.med_profile
        x = self.y

        gmod = GaussianModel()
        cmod = ConstantModel()

        pars = gmod.make_params(amplitude=A or -(self.med_profile.max() - self.med_profile.min()),
                                sigma=s or 3 * self.pixel_size,
                                center=x0 or self.med_profile.tolist().index(min(self.med_profile)) * self.pixel_size)
        pars['amplitude'].max = 100.0
        pars['center'].max = 0.9 * self.nrows * self.pixel_size
        pars['center'].min = 0.1 * self.nrows * self.pixel_size
        pars['sigma'].min = self.pixel_size
        pars['sigma'].max = 4 * self.pixel_size

        pars += cmod.make_params(c=c or self.med_profile.max())
        mod = gmod + cmod
        self.median_fit = mod.fit(med_prof, params=pars, x=x)

    def get_median_fit_surf(self, x=None):
        """
        Return a 2D array made form the fitted Gaussian.
        """
        prof = self.median_fit.eval(x=x or self.y)
        return array([prof for i in range(self.ncols)])
    
    def get_fit_residuals(self):
        """
        Return the residual image.
        
        The fitted Gaussian is subtracted from all columns of the loaded image.
        """
        profile = self.ima.astype(float64)

        return array([profile[:, idx] - self.med_profile
                      for idx in range(self.ncols)]).transpose()

    def get_noise(self, width=3):
        """
        Returns a part of the residual image.
        
        Only rows outside the range -width*sd < x0 < width*sd, so possible
        artifacts from the subtraction of the Gaussian fit are excluded.
        
        """
        mzr = self.median_fit
        mean = mzr.params['center'].value
        sigma = mzr.params['sigma'].value

        residuals = self.get_fit_residuals()

        mask = array([logical_or(self.y < mean - width * sigma,
                                 self.y > mean + width * sigma)
                      for i in range(len(self.x))]
                     ).transpose()
        b = residuals[mask]
        
        return b.reshape((int(len(b) / self.ncols), self.ncols))

    def get_filtered_noise(self, width=3,
                           axis=None, bins=100, verbose=False, **pltkws):
        """
        Return an array of filtered noise.
        
        The method uses the interquartile range to calculate an upper or lower
        threshold of the residual noise.
        
        Arguments
        ---------
        width : float
            Factor of how many standard deviations away from the fitted
            MT center to exclude from the residual image.
        axis : matplotlib.Axis or None
            If provided the noise and histogram is plotted into it.
        bins : int
            Set the number of bins of the histogram.
        verbose : bool
            Print additional info.
        pltkws : Keyword arguments given to matplotlib plot function.
        """
        noise = self.get_noise(width=width)
        
        h, b = histogram(noise.flatten(), bins=bins)
        bc = b[:-1] + (b[1:] - b[:-1])/2

        if axis:
            fig = axis.figure
            ax = axis
            ax.hist(noise.flatten(), bins=bins, color='red')

        q1 = percentile(noise, 25, interpolation='lower')
        q3 = percentile(noise, 75, interpolation='higher')
        iqr = q3 - q1
        low = q1 - 1.5 *iqr
        upp =  q3 + 1.5 * iqr
        new_noise = noise[logical_and(noise >= low, noise <= upp)]
        
        if axis:
            ylims = ax.get_ylim()
            add_plot_to_figure(fig, [low, low], ylims, fmt='--k', axis=ax, linewidth=1.0, alpha = 0.7)
            add_plot_to_figure(fig, [upp, upp], ylims, fmt='--k', axis=ax, linewidth=1.0, alpha = 0.7)
            
        h2, _ = histogram(new_noise, bins=b)

        mod = GaussianModel()
        pars = mod.guess(h2, x=bc)
        mzr2 = mod.fit(h2, pars, x=bc)
        
        if verbose:
            report_fit(mzr2)
        if axis:
            ax.hist(new_noise, bins=b, color='lightblue')
            add_plot_to_figure(fig, bc, mzr2.eval(), axis=ax, color='blue', fmt='--' ,
                               label='filtered noise', **pltkws)

        return new_noise

    def show_image(self, axis=None, **kws):
        """
        Plot the MT image.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.gca()
        else:
            fig = axis.figure

        im = axis.imshow(self.ima, **kws)
        
        return fig, im

    def plot(self):
        """
        """
        profile = self.ima.astype(float64)
        noise = self.get_noise()
        
        grid = (2, 4)

        plt.close('all')
        fig = plt.figure()
        ax1 = get_gridded_axis(figure=fig, grid=grid, axslice=(slice(0, 1), slice(0, 1)))
        ax2 = get_gridded_axis(figure=fig, grid=grid, axslice=(slice(1, 2), slice(0, 1)))
        

        fs =14
        ax1.imshow(profile, cmap=cm.Greys_r)
        ax1.set_title('MT profile', fontsize=fs)
        ax1.set_xlabel('X (µm)', fontsize=fs)
        ax1.set_ylabel('Y (µm)', fontsize=fs)

        ax2.imshow(noise, cmap=cm.Greys_r)
        ax2.set_title(r'Residuals $\mathsf{(> \pm \sigma)}$', fontsize=fs)
        ax2.set_xlabel('X (µm)', fontsize=fs)
        ax2.set_ylabel('Y (µm)', fontsize=fs)

        ax3 = get_gridded_axis(figure=fig, grid=grid, axslice=(slice(0, 3), slice(1, 4)))
        self.plot_residuals(axis=ax3)

        fig.tight_layout()

        return fig

    def plot_median_profile(self, plot_fit=True, axis=None,
                            xlabel=None, ylabel=None, **pltkws):
        """
        Plot the median profile and the fitted Gaussian
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.gca()
        else:
            fig = axis.figure
        
        y_ = arange(min(self.y), max(self.y), 0.1 * self.pixel_size)
        add_plot_to_figure(fig, self.y, self.med_profile, label='median profile', axis=axis)
        if plot_fit:
            add_plot_to_figure(fig, y_, self.median_fit.eval(x=y_),
                               label='Gaussian fit', axis=axis,
                               linewidth=1.0, fmt='--',
                               xlabel=xlabel or 'Pixel',
                               ylabel=ylabel or 'IRM signal', **pltkws)
        return fig
    
    def plot_residuals(self, axis=None, cmap=None, **kws):
        """
        Plot the residual image.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        else:
            fig = axis.figure
        axis.imshow(self.get_fit_residuals(), cmap=cmap or 'Greys_r', **kws)
        
        return fig
        
    def plot_median_surface(self, axis=None,
                            xlabel=None, ylabel=None, zlabel=None, **pltkws):
        """
        plot the fitted gaussian as a surface
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111, projection='3d')
        else:
            fig = axis.figure
        
        Xs, Ys = meshgrid(self.x, self.y)

        axis.plot_surface(Xs, Ys, self.ima / 2**16,
                          rstride=1, cstride=1, cmap=cm.Greys_r,
                          linewidth=0.7, alpha=0.5)
        axis.plot_surface(Xs, Ys, self.get_median_fit_surf() / 2**16,
                          rstride=10, cstride=10,
                          color='Green', linewidth=1.0, alpha=0.4)

        axis.set_xlabel(xlabel or "X (µm)")
        axis.set_ylabel(ylabel or "Y (µm)")
        axis.set_zlabel(zlabel or r'Intensity $\mathsf{(2^{-16})}$')
        axis.axesPatch.set_facecolor('None')
        axis.w_xaxis.set_pane_color((0, 0, 0, 0))
        axis.w_yaxis.set_pane_color((0, 0, 0, 0))
        axis.w_zaxis.set_pane_color((0, 0, 0, 0))
        axis.elev = 40
        axis.azim = -40
        
        return fig

    def get_signal(self):
        """ Return the measured signal, i.e. the height of the Gaussian. """
        mzr = self.median_fit
        return -1 * mzr.params['height'].value

    def get_SNR(self):
        """ Return the signal-to-noise ratio. """
        S = self.get_signal()
        N = self.get_noise().std()

        return S / N

    def get_SNR_corrected(self, width=3, **kws):
        """ Return the SNR with filtered noise. """
        S = self.get_signal()
        N = self.get_filtered_noise(width=width, **kws).std()

        return S / N

    def get_SNR_err(self):
        """
        Return the error bars on the SNR measurement.
        
        The error is a result of the errorbar on the signal from the Gaussian
        fit.
        """
        S = self.get_signal()
        
        mzr = self.median_fit
        dS = mzr.params['amplitude'].stderr
        
        SNR = self.get_SNR()
        return dS / S * SNR

    def report(self):
        """
        Print the Gaussian fit results and the SNR.
        """
        report_fit(self.median_fit)

        print('=================================')
        print('SNR = {0:1.3f} +/- {1:1.3f}'.format(self.get_SNR(), self.get_SNR_err()))
        print('=================================')

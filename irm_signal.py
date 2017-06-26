#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# - Author: steve simmert
# - Copyright: 2017
# """
# Model function of IRM intensity profile of a tilted MT and TIRF intensity profile
from scipy import sin, cos, arcsin, pi, sinc


def height_fun(x, x0, A, xt):
    """
    Model the height of a cantilevered microubule.
    
    The MT is fixed at x0 and pulled up. A is the shape giving/ scaling factor,
    equal to F/EI, where F ist the forc acting on the MT at xt, and EI is the
    flexural regidity of the mictotubule.
    """
    def heaviside(x, x0):
        return 1 * (x >= x0)
    
    return heaviside(x, x0) * A * ((xt - x0) / 2 * (x - x0)**2 - ((x - x0)**3) / 6)


def irm_intensity_vs_height(h, bg, d, INA, wl, n1):
    """
    Intensity of an IRM signal, given the background and difference
    signal and the height of the scatting object.
    
    All length units should be in meters!
    
    bg could be, for example, 0.0 for a perfect gray value.
    Rädler & Sackmann, Eq. 4 in Limozion et al. 2009
    """
    theta = arcsin(INA / n1)
    k = 2 * pi * n1 / wl
    # following division by pi is done because of the scipy.sinc
    # definition: sinc(x) = sin(pi*x)/(pi*x)
    y =  2 * k * h * (sin(theta / 2))**2 / pi
    
    i = bg - d * sinc(y) * cos(2 * k * (h * (1 - (sin(theta/2))**2)))
    
    return i

def irm_signal_bent_mt(x, x0, bg, d, A, xt, INA=1.33, wl=0.448, n1=1.33):
    """
    Intensity profile of a bent and cantilevered microtubule.
    
    Here the position and wavelength values should be in µm!
    """
    # conversion to meters
    h = height_fun(x, x0, A, xt) * 1e-6  # conversion to meters
    
    return irm_intensity_vs_height(h, bg, d, INA, wl * 1e-6, n1)

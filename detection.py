import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Rebin the simulation images to match the detector's pixel sizes 
# (typically 13µm for CCDs, 55µm for Minipix)
# -----------------------------------------------------------------------------

def rebin_2d(a, bin_factor):
    nrows = (a.shape[0] // bin_factor) * bin_factor
    ncols = (a.shape[1] // bin_factor) * bin_factor
    a_cropped = a[:nrows, :ncols]
    new_shape = (nrows // bin_factor, bin_factor, ncols // bin_factor, bin_factor)
    return a_cropped.reshape(new_shape).mean(axis=(1, 3))

def CCD_detection_binned(intensity, bin_factor, gain=10, QE=0.57, ADC_bits=18):
    """
    Simulate CCD detection by combining binning and photon statistics.
    
    Parameters:
      intensity : 2D array of continuous simulated intensity (e.g., |E|^2).
      bin_factor: Integer number of simulation pixels to combine into one CCD pixel.
      gain      : Gain factor (default: 1).
      QE        : Quantum efficiency (default: 0.57, i.e. 57%).
      ADC_bits  : ADC resolution in bits (default: 10 bits, so max value = 2^10).
      
    Returns:
      detected  : 2D array of simulated CCD counts (integers) after binning and Poisson noise.
    """
    # First, rebin the high-resolution intensity image to the detector pixel scale.
    binned_intensity = rebin_2d(intensity, bin_factor)
  
    plt.figure()
    plt.imshow(binned_intensity)
    plt.show()
    
    # Now, simulate photon detection using Poisson noise.
    # Here the binned intensity is assumed to represent the mean number of photons per CCD pixel.
    detected = np.random.poisson(binned_intensity) * gain * QE

    # Clip values that exceed the ADC maximum
    max_value = 2 ** ADC_bits
    detected[detected > max_value] = max_value
    
    # Round to nearest integer (simulate ADC digitization)
    detected = np.round(detected).astype(int)
    
    return detected
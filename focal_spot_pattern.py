import numpy as np
from scipy.ndimage import gaussian_filter

def create_slit_mask(X, Y):
    """
    Create a simple slit mask: light is transmitted (1) only when |x| < slit_width/2.
    """
    mask = np.ones_like(X)
    slit_width = 1e-5  # Slit width
    mask[np.abs(X) > slit_width/2] = 0
    return mask

def create_gaussian_mask(X, Y, diameter=20e-6, sigma=None):
    """
    Create a Gaussian amplitude mask with a specified 1/e² intensity diameter.
    The amplitude falls to 1/e at a radius of (diameter/2).
    """
    if sigma is None:
        sigma = diameter / 2
    mask = np.exp(- (X**2 + Y**2) / (2*sigma**2))
    return mask

def create_slit_pattern(X, Y, period=1e-6, duty_cycle=0.5, angle=0):
    """
    Create a binary slit pattern with a given period, duty cycle, and rotation angle.
    """
    xp = X * np.cos(angle) + Y * np.sin(angle)
    mod_val = np.mod(xp, period)
    pattern = np.where(mod_val < duty_cycle * period, 1, 0)
    return pattern

def create_slit_pattern_rand(X, Y, period=0.5e-6, duty_cycle=0.5, angle=0):
    """
    Create a binary slit pattern with a given period, duty cycle, and rotation angle.
    Introduces a random phase shift between 0 and 1 period to simulate sampling different parts of the periodic pattern.
    
    Parameters:
      X, Y      : 2D coordinate grids (meters).
      period    : Period of the slits (meters).
      duty_cycle: Fraction of each period that is open (transparent).
      angle     : Rotation angle (radians) of the slit pattern.
    
    Returns:
      pattern   : 2D binary array with values 1 (open) and 0 (blocked).
    """
    # Compute rotated coordinate system
    xp = X * np.cos(angle) + Y * np.sin(angle)
    
    # Generate a **random phase shift** between 0 and 1 full period
    shift_amount = np.random.uniform(0, 2*period)  
    xp_shifted = xp + shift_amount  # Apply random shift
    
    # Create the slit pattern with the randomly shifted coordinate
    mod_val = np.mod(xp_shifted, period)
    pattern = np.where(mod_val < duty_cycle * period, 1, 0)
    
    return pattern


def create_slit_pattern_rand_smooth(X, Y, period=4e-6, duty_cycle=0.3, angle=0, smoothing_fraction=0.1, dx_source=1e-6):
    """
    Create a slit pattern with soft edges using a Gaussian blur, where the blur width is a fraction of the period.

    Parameters:
      X, Y               : 2D coordinate grids (meters).
      period             : Period of the slits (meters).
      duty_cycle         : Fraction of each period that is open (transparent).
      angle              : Rotation angle (degrees) of the slit pattern.
      smoothing_fraction : Fraction of the period over which smoothing occurs (e.g., 0.1 for 10%).
      dx_source          : Grid spacing (pixel size in meters).

    Returns:
      pattern_smooth     : 2D **smoothed** array (values between 0 and 1).
    """
    
    # Compute rotated coordinate system
    angle = np.deg2rad(angle)
    xp = X * np.cos(angle) + Y * np.sin(angle)
    
    # Generate a **random phase shift** between 0 and 1 full period
    shift_amount = np.random.uniform(0, period)  
    xp_shifted = xp + shift_amount  # Apply random shift
    
    # Create binary slit pattern
    mod_val = np.mod(xp_shifted, period)
    pattern_hard = np.where(mod_val < duty_cycle * period, 1, 0)
    
    # Define blur width as a fraction of the period
    blur_width = smoothing_fraction * period  # Defines blur in meters
    
    # Convert to pixels
    blur_sigma = blur_width / dx_source  # Converts to pixels
    
    # Apply Gaussian blur for smooth edges
    pattern_smooth = gaussian_filter(pattern_hard.astype(float), sigma=blur_sigma)
    pattern_smooth = np.clip(pattern_smooth, 0, 1)  # Clip values to [0, 1] to avoid numerical phase shifts
    
    return pattern_smooth


def create_double_gaussian_mask(X, Y, sigma=30e-6, separation=0.5e-3, angle=0):
    """
    Create a mask consisting of two Gaussian spots.
    
    Each spot is a 2D Gaussian with an RMS width 'sigma'.
    The centers of the spots are located along a line at the specified 'angle'
    (in radians) with respect to the horizontal X-axis. The spots are separated
    by 'separation' (center-to-center).
    
    Parameters:
      X, Y       : 2D coordinate grids (in meters).
      sigma      : RMS width (standard deviation) of each Gaussian spot (in meters).
                   Default is 30 µm.
      separation : Center-to-center separation between the two spots (in meters).
                   Default is 0.5 mm.
      angle      : Rotation angle (in radians) of the line joining the centers.
                   Default is 0 (horizontal alignment).
    
    Returns:
      mask       : 2D array where each pixel value is the sum of the amplitudes
                   of the two Gaussian spots.
    """
    # Convert angle from degrees to radians.
    angle = np.deg2rad(angle)
    
    # Calculate half the separation projected onto x and y based on the angle.
    dx = (separation / 2) * np.cos(angle)
    dy = (separation / 2) * np.sin(angle)
    
    # Define centers of the two spots.
    # When angle=0, centers are at (-separation/2, 0) and (separation/2, 0)
    center1_x, center1_y = -dx, -dy
    center2_x, center2_y =  dx,  dy
    
    # Create the two Gaussian spots.
    spot1 = np.exp(-(((X - center1_x)**2 + (Y - center1_y)**2) / (2 * sigma**2)))
    spot2 = np.exp(-(((X - center2_x)**2 + (Y - center2_y)**2) / (2 * sigma**2)))
    
    # Sum the two spots to form the final mask.
    mask = spot1 + spot2
    return mask
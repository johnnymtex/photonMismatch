import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def combined_propagation(E, wavelength, z, dx, threshold=1.0):
    """
    Propagate the field E over distance z using either Fraunhofer or Fresnel propagation,
    depending on the computed Fresnel number.
    """
    N = E.shape[0]
    aperture_size = dx * N
    Fresnel_number = aperture_size**2 / (wavelength * z)
    
    if Fresnel_number < threshold:
        #print("Using Fraunhofer propagation.")
        E_out, _, _ = fraunhofer_propagation(E, wavelength, z, dx)  # Unpack and discard x_det, y_det
        return E_out
    else:
        #print("Using Fresnel propagation.")
        E_out, _, _ = fresnel_propagation(E, wavelength, z, dx)  # Unpack and discard x_out, y_out
        return E_out
    
    
def fraunhofer_propagation(E, wavelength, z, dx):
    """
    Propagate an input field E using the Fraunhofer approximation.
    """
    N = E.shape[0]

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    FX, FY = np.meshgrid(fx, fy)

    # Fourier transform of the input field
    input_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))

    # Coordinates in the observation plane
    X_det = FX * wavelength * z
    Y_det = FY * wavelength * z

    k = 2*np.pi/wavelength

    # Calculate the output field using the Fraunhofer approximation
    output_field = np.exp(1j * k * z) * np.exp(1j*np.pi/(wavelength*z)*(X_det**2+Y_det**2)) * input_spectrum
    output_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(output_field)))

    return output_field

def fresnel_propagation(E, wavelength, z, dx):
    """
    Propagate an input field E using the Fresnel approximation.
    """

    N = E.shape[0]

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength

    H = np.exp(1j*k*z)*np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    input_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))

    output_spectrum = input_spectrum * H
    output_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(output_spectrum)))

    return output_field
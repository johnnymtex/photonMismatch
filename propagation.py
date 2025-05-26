import numpy as np
import matplotlib.pyplot as plt

def combined_propagation(E, wavelength, z, dx, threshold=1.0, padding_factor=1):
    """
    Propagate the field E over distance z using either Fraunhofer or Fresnel propagation,
    depending on the computed Fresnel number.
    """
    N = E.shape[0]
    aperture_size = dx * N
    Fresnel_number = aperture_size**2 / (wavelength * z)
    
    if Fresnel_number < threshold:
        E_out = fraunhofer_propagation(E, wavelength, z, dx, padding_factor=padding_factor)  # Unpack and discard x_det, y_det
        return E_out
    else:
        E_out = fresnel_propagation(E, wavelength, z, dx, padding_factor=padding_factor)  # Unpack and discard x_out, y_out
        return E_out, x_det, y_det
    
    
def fraunhofer_propagation(E, wavelength, z, dx, padding_factor=1):
    """
    Propagate an input field E using the Fraunhofer approximation with zero-padding.
    """
    N = E.shape[0]

    # Zero-padding the input field
    pad_width = N * (padding_factor-1)//2

    padded_E = np.pad(E, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=(0+0j,0+0j))
    padded_N = padded_E.shape[0]

    fx = np.fft.fftshift(np.fft.fftfreq(padded_N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(padded_N, d=dx))
    FX, FY = np.meshgrid(fx, fy)

    X_det = FX*wavelength*z
    Y_det = FY*wavelength*z

    # Fourier transform of the padded input field
    input_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_E)))

    # Coordinates in the observation plane
    print(len(np.fft.fftfreq(N, d=dx)))
    x_det = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * wavelength * z
    y_det = x_det.copy()

    k = 2 * np.pi / wavelength

    # Calculate the output field using the Fraunhofer approximation
    output_field = np.exp(1j * k * z) * np.exp(-1j * k / (2 * z) * (X_det**2 + Y_det**2)) * input_spectrum
    output_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(output_field)))

    return output_field[pad_width:pad_width+N, pad_width:pad_width+N]

def fresnel_propagation(E, wavelength, z, dx, padding_factor=1):
    """
    Propagate an input field E using the Fresnel approximation.
    """
    N = E.shape[0]

    pad_width = N * (padding_factor-1)//2

    padded_E = np.pad(E, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=(0+0j,0+0j))
    padded_N = padded_E.shape[0]

    fx = np.fft.fftshift(np.fft.fftfreq(padded_N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(padded_N, d=dx))
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength

    H = np.exp(-1j*k*z)*np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    input_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_E)))

    output_spectrum = input_spectrum * H

    x_det = fx * wavelength * z
    y_det = fy * wavelength * z

    output_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(output_spectrum)))

    return output_spectrum, x_det, y_det
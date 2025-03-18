import numpy as np
import matplotlib.pyplot as plt

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

    E_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    x_det = wavelength * z * fx
    y_det = wavelength * z * fy

    k = 2 * np.pi / wavelength
    scaling = np.exp(1j * k * z) / (1j * wavelength * z)

    E_out = scaling * E_ft * (dx ** 2)

    return E_out, x_det, y_det

def fresnel_propagation(E, wavelength, z, dx):
    """
    Propagate an input field E using the Fresnel approximation.
    """
    N = E.shape[0]

    k = 2 * np.pi / wavelength

    x = np.linspace(-N/2*dx, N/2*dx, N)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    Q_in = np.exp(1j * k/(2*z) * (X**2 + Y**2))

    U1 = E * Q_in
    U2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U1)))

    print(np.abs(U2)**2)

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    print(fx)

    x_out = wavelength*z*fx
    y_out = wavelength*z*fy

    print(x_out)
    X_out, Y_out = np.meshgrid(x_out, y_out)  # Create meshgrid for correct shape

    Q_out = np.exp(1j * k/(2*z) * (X_out**2 + Y_out**2))
    U_out = np.exp(1j * k * z) / (1j * wavelength * z) * U2 * Q_out
    
    return U_out, x_out, y_out
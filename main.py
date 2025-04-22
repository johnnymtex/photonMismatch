import sys
import get_input as inp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from focal_spot_pattern import create_gaussian_mask, create_slit_pattern_rand_smooth
from set_simulation import simulate_intensity_images, compute_g2, save_results
from scipy.ndimage import map_coordinates

#defining constants
e_charge = 1.6e-19
planck = 6.626e-34
light = 3e8

# starting plotting configuration
plt.rcParams['figure.figsize'] = (8,6)

# initializing variables
setup = inp.Config('inp_files/setup.inp')

dx_source = 50e-6/setup.num_pixels
dx_det_sim = (setup.wavelength*setup.z_prop)/(setup.num_pixels*dx_source)
bin_factor = int(round(setup.detector_pixel_size/dx_det_sim))

#auxiliary variables
source_size=dx_source*setup.num_pixels
x_source = np.linspace(-source_size/2, source_size/2, setup.num_pixels)
y_source = np.linspace(-source_size/2, source_size/2, setup.num_pixels)
X_source, Y_source = np.meshgrid(x_source, y_source)

gaussian_mask1 = create_gaussian_mask(X_source, Y_source, diameter=setup.gauss_width)
print(x_source)
print(setup.gauss_width)
#grating_mask1 = create_slit_pattern_rand_smooth(X_source, Y_source, period=setup.stripe_period, angle=setup.angle ,smoothing_fraction=.1)
object_mask1 = gaussian_mask1 #* grating_mask1
print('Total photons at source: ', f'{setup.I0*np.sum(object_mask1):.2e}')
print('Duration of Kalpha (tc=0.6 fs): ', setup.num_modes_per_shot*0.6, ' fs')
#4500 represents the photon energy in eV considering the given wavelength; the electron charge is just a conversion factor (1eV = 1.6e-19J)

photon_energy = (planck*light)/setup.wavelength

print('Total energy in Ka ', f'{setup.I0*np.sum(object_mask1)*photon_energy:.2e}', ' (J)')
print('conversion efficiency assuming 4mJ laser ', f'{setup.I0*np.sum(object_mask1)*photon_energy/4e-3:.2e}')

# current_object_mask_func is our function to generate a grating mask.
current_object_mask_func = create_slit_pattern_rand_smooth

print("Starting simulation...")
intensity_images, field_images = simulate_intensity_images(X_source, Y_source, setup.num_shots, setup.num_modes_per_shot, setup.I0, setup.z_prop, 
                                                          setup.gauss_width, setup.stripe_period,
                                                          current_object_mask_func,
                                                          setup.num_pixels, dx_source, setup.angle, setup.wavelength,
                                                          bin_factor, setup.gain, setup.QE, setup.ADC_bits)

# Compute g² and vertical lineout.
avg_intensity, autocorr_avg, vertical_sum, I_per_pix = compute_g2(intensity_images)

print("\nCheckpoint 5a: Ensemble-Averaged Intensity")
print("Min =", np.min(avg_intensity), "Max =", np.max(avg_intensity), "Avg =", np.mean(avg_intensity))
print(f"Photons per pixel: {I_per_pix:.2f}")

# Visualization: Show the autocorrelation image (g² proxy)
plt.figure()
plt.plot(autocorr_avg[int(len(autocorr_avg)/2)])
plt.show()


plt.figure()
extent = [0, setup.detector_pixel_size*(setup.num_pixels/bin_factor)*1e6, 0, setup.detector_pixel_size*(setup.num_pixels/bin_factor)*1e6]
plt.imshow(autocorr_avg, origin='lower', cmap='inferno', norm=mcolors.LogNorm(), extent=extent)
plt.title("Ensemble-Averaged Intensity Autocorrelation (g² proxy) - Log Scale")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.colorbar(label="Autocorrelation")
plt.show()


# Plot the vertical lineout.
x_pixels = np.arange(len(vertical_sum))
x_microns = x_pixels * setup.detector_pixel_size / bin_factor * 1e6  # Convert pixels to microns
plt.figure(figsize=(6, 4))
plt.plot(x_pixels, vertical_sum, 'b-', linewidth=2)
plt.xlabel("pixels")
plt.ylabel("Summed g²")
plt.title("Vertically Summed g² Function")
plt.grid(True)
plt.show()

# Plot the log of the vertical sum.
plt.figure(figsize=(6, 4))
plt.plot(x_microns, np.log(np.sum(np.abs(autocorr_avg), axis=0)), 'b-', linewidth=2)
plt.xlabel("x (µm)")
plt.ylabel("log(Summed g²)")
plt.title("log abs Vertically Summed g² Function")
plt.grid(True)
plt.show()

# Create a configuration dictionary.
config = {
    "wavelength": setup.wavelength,
    "z_prop": setup.z_prop,
    "detector_pixel_size": setup.detector_pixel_size,
    "dx_source": dx_source,
    "num_shots": setup.num_shots,
    "num_modes_per_shot": setup.num_modes_per_shot,
    "I0": setup.I0,
    "I_per_pix": I_per_pix,
    "bin_factor": bin_factor,
    "gauss_width": setup.gauss_width,
    "stripe_period": setup.stripe_period
}

# Save the configuration, autocorr_avg image, and vertical lineout.
save_results(config, autocorr_avg, vertical_sum)
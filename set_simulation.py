import json
import numpy as np
import matplotlib.pyplot as plt
import LightPipes as lp

from focal_spot_pattern import create_gaussian_mask
from propagation import combined_propagation
from detection import CCD_detection_binned

from scipy.signal import fftconvolve, correlate2d
from scipy.ndimage import map_coordinates

# ----------------------------------------------------------------------------
# Function 1: Simulation of intensity images (binned) for a multi-shot experiment
# Run this function to simulate intensity images for num_shots shots.
# ----------------------------------------------------------------------------
def simulate_intensity_images(X_source, Y_source, num_shots, num_modes_per_shot, I0, z_prop, 
                              gauss_width, stripe_period,
                              current_object_mask_func,
                              num_pixels, dx_source, angle, wavelength,
                              bin_factor, gain, QE, ADC_bits, padding_factor, incoherent=False):
    """
    Simulate intensity images from a multi-shot experiment.
    
    Parameters:
      X_source, Y_source: 2D coordinate grids in the source plane (m)
      num_shots: number of shots (different grating masks)
      num_modes_per_shot: number of modes per shot (random phase realizations)
      I0: source intensity (photons per pixel at source)
      z_prop: propagation distance (m)
      gauss_width: diameter (or width) of the Gaussian aperture (m)
      stripe_period: period of the grating (m)
      current_object_mask_func: function to generate a grating mask;
          Called as: current_object_mask_func(X, Y, period, duty_cycle, angle, smoothing_fraction, dx_source)
      num_pixels: simulation grid size (assumed square)
      dx_source: pixel size in the source plane (m)
      angle: angle for the grating mask (radians)
      wavelength: wavelength (m)
      bin_factor: binning factor to convert simulation resolution to detector resolution
      gain, QE, ADC_bits: parameters for CCD_detection_binned.
      
    Returns:
      intensity_images: list of binned intensity images (one per shot)
      field_images: list of the last propagated field from each shot (optional)
    """
    
    intensity_images = []
    field_images = []
    
    # Create the Gaussian mask using the provided gauss_width.
    gaussian_mask = create_gaussian_mask(X_source, Y_source, w=gauss_width)
    
    for shot in range(num_shots):
        # Generate a grating mask (one per shot) using the provided stripe_period.
        grating_mask = current_object_mask_func(X_source, Y_source, stripe_period, angle=angle, dx_source=dx_source)

        # Combine with the Gaussian mask to form the overall object mask.
        current_object_mask = gaussian_mask * grating_mask
        
        # Compute intensity per mode.
        intensity_per_mode = I0 * np.ones((num_pixels, num_pixels)) / num_modes_per_shot
        
        # Initialize accumulator for shot's full-resolution intensity.
        shot_intensity = np.zeros((padding_factor*num_pixels, padding_factor*num_pixels))

        for mode in range(num_modes_per_shot):
            # Generate a new random phase pattern.
            if not incoherent:
              E_source = np.sqrt(intensity_per_mode)
            else:
              random_phase = np.random.uniform(0, 2*np.pi, (num_pixels, num_pixels))
              E_source = np.sqrt(intensity_per_mode) * np.exp(1j * random_phase)
                
            # Apply the object mask.
            E_after_object = E_source * current_object_mask

            if shot == 0 and mode == 0:  # Plot only for the first shot and mode
                # Plot amplitude and phase *after* the random phase is applied
                intensity_to_plot = np.abs(E_after_object**2) * num_modes_per_shot
                extent = [np.min(X_source)*1e6, np.max(X_source)*1e6, np.min(Y_source)*1e6, np.max(Y_source)*1e6]
                plt.figure(figsize=(8, 6))
                plt.imshow(intensity_to_plot, extent=extent, cmap='Greys')
                plt.title(f"Total Source Intensity: {np.sum(intensity_to_plot):.2e} photons per pulse")
                plt.colorbar()
                plt.xlabel("x [$\mu$m]")
                plt.ylabel("x [$\mu$m]")
                plt.show()

            # Propagate the field.
            E_det, x_det, y_det = combined_propagation(E_after_object, wavelength, z_prop, dx_source, padding_factor=padding_factor)
            I_det = np.abs(E_det)**2
            #I_det /= np.max(I_det)

            if shot == 0:
              print("\n######## Checking energy conservation... ########")
              print(f"Before propagation: {np.sum(np.abs(E_after_object)**2)}")
              print(f"After propagation: {np.sum(np.abs(E_det)**2)}")
              print("#################################################\n")

              extent = [np.min(x_det)*1e6, np.max(x_det)*1e6, np.min(y_det)*1e6, np.max(y_det)*1e6]

              plt.figure()
              plt.imshow(I_det, extent=extent, cmap="Greys")
              plt.title("Normalized intensity at detector")
              plt.xlabel("x [$\mu$m]")
              plt.ylabel("y [$\mu$m]")
              plt.colorbar()
              plt.show()

            shot_intensity += I_det
        
        # Optionally store the last propagated field.
        field_images.append(E_det)

        # Apply CCD detection (including binning, Poisson noise, etc.)
        shot_intensity_binned = CCD_detection_binned(shot_intensity, bin_factor=bin_factor, gain=gain, QE=QE, ADC_bits=ADC_bits)

        if shot == 0:
            plt.figure()
            extent = [np.min(x_det)*1e6, np.max(x_det)*1e6, np.min(y_det)*1e6, np.max(y_det)*1e6]
            plt.imshow(shot_intensity_binned, extent=extent, cmap="Greys")
            plt.xlabel("x [$\mu$m]")
            plt.xlabel("y [$\mu$m]")
            plt.title("Binned intensity after detection")
            plt.colorbar()
            plt.show()

        intensity_images.append(shot_intensity_binned)
        print(f"Completed Shot {shot+1}/{num_shots} - Photons per pixel: {np.sum(shot_intensity_binned)/((num_pixels/bin_factor)**2):.2f}")
    
    return intensity_images, field_images, x_det, y_det

# ----------------------------------------------------------------------------
# Function 2: Compute g² from the intensity images
# ----------------------------------------------------------------------------
def compute_g2(intensity_images):
    """
    Compute the ensemble-averaged intensity, autocorrelation (g² proxy),
    and vertical lineout from a list of intensity images.
    
    Returns:
      avg_intensity: ensemble-averaged intensity image
      autocorr_avg: averaged autocorrelation of intensity fluctuations
      vertical_sum: vertical lineout (summed along rows) of autocorr_avg
      I_per_pix: average number of photons per pixel (scalar)
    """
    num_shots = len(intensity_images)
    avg_intensity = np.mean(intensity_images, axis=0)
    deltaI_images = [img - avg_intensity for img in intensity_images]
    N_bin = avg_intensity.shape[0]
    autocorr_sum = np.zeros((N_bin, N_bin))
    for I in intensity_images:
        autocorr = fftconvolve(I, I[::-1, ::-1], mode='same')
        autocorr /= np.mean(I)**2
        autocorr_sum += autocorr
    autocorr_avg = autocorr_sum / num_shots
    I_per_pix = np.mean(avg_intensity)
    vertical_sum = np.sum(autocorr_avg, axis=0)
    return avg_intensity, autocorr_avg, vertical_sum, I_per_pix

def compute_g2_reviewer(intensity_images):
    avg_intensity = np.mean(intensity_images, axis=0)
    N = avg_intensity.shape[0]

    N_image2, N_image3, N_corr_c2 = 200, 100, 200
    displace_x1, displace_y1, displace_x, displace_y = 0, 0, 0, 0

    images_2 = intensity_images[:,
              N//2 - N_image2 // 2 + displace_x1: N//2 + N_image2 // 2 + displace_x1,
              N//2 - N_image2 // 2 + displace_y1: N//2 + N_image2 // 2 + displace_y1,
              ]
    
    print(images_2.shape)
    print(np.sum(images_2[:]**2))

    c2j_2_zeros = np.zeros((N_image2, N_image2))
    denominator = np.zeros((N_image2, N_image2))

    # Compute g2 correlation
    for q1, q11 in enumerate(range(-N_image2 // 2 - 1, N_image2 // 2 - 1)):
        for q2, q22 in enumerate(range(-N_image2 // 2 - 1, N_image2 // 2 - 1)):
            padded_images = np.pad(intensity_images, pad_width=((0,0),(intensity_images.shape[1], intensity_images.shape[1]), (intensity_images.shape[2], intensity_images.shape[2])),
                          mode='constant', constant_values=0)

            # Pad the image with zeros on each side
            disp_padded_images = np.zeros((intensity_images.shape[0], intensity_images.shape[1] * 3, intensity_images.shape[2] * 3))
            desired_position = [desired_position[0] + intensity_images.shape[1], desired_position[1] + intensity_images.shape[2]]
            # Place the original image at the desired position
            disp_padded_images[:, desired_position[0]:desired_position[0] + intensity_images.shape[1], desired_position[1]:desired_position[1] + intensity_images.shape[2]] = intensity_images

            denominator[q2,q1] = np.sum(np.sum(np.mean(disp_padded_images,0 ) * np.mean(padded_images,0 )))

    for image in intensity_images:
        for q1, q11 in enumerate(range(-N_image2 // 2 - 1, N_image2 // 2 - 1)):
        #for q1, q11 in enumerate(range(-1, 1)):
            for q2, q22 in enumerate(range(-N_image2 // 2 - 1, N_image2 // 2 - 1)):
        #    for q2, q22 in enumerate(range(-1, 1)):
                #c2j_2[q2, q1] = g2_correlation_Lucas(image_2, [q11, q22])
                desired_position = [q11, q22]

                padded_image = np.pad(image, pad_width=((image.shape[0], image.shape[0]), (image.shape[1], image.shape[1])),
                          mode='constant', constant_values=0)

                # Pad the image with zeros on each side
                disp_padded_image = np.zeros((image.shape[0] * 3, image.shape[1] * 3))
                desired_position = [desired_position[0] + image.shape[0], desired_position[1] + image.shape[1]]
                # Place the original image at the desired position
                disp_padded_image[desired_position[0]:desired_position[0] + image.shape[0], desired_position[1]:desired_position[1] + image.shape[1]] = image
                
                numerator = np.sum(np.sum(disp_padded_image * padded_image))

                c2j_2_zeros[q2, q1] = numerator/ denominator[q2, q1]

    return c2j_2_zeros

def compute_inclined_lineout(autocorr_avg, angle_deg=20, width=10):
    """
    Compute a lineout along an inclined direction at a given angle.
    
    Parameters:
      autocorr_avg: 2D autocorrelation image.
      angle_deg   : Angle of the stripes (in degrees, default 20º).
      width       : Number of pixels to average over perpendicular to the line.
    
    Returns:
      lineout_values: 1D array of extracted values along the inclined line.
      x_coords_um: Corresponding x-coordinates in microns.
    """

    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Get image dimensions
    Ny, Nx = autocorr_avg.shape

    # Define center of image
    center_x, center_y = Nx // 2, Ny // 2

    # Define range of x-values (line will pass through center)
    x_coords = np.arange(-Nx//2, Nx//2)  # Centered at 0
    y_coords = np.tan(angle_rad) * x_coords  # 20º inclined line

    # Convert to actual image indices
    x_indices = np.round(x_coords + center_x).astype(int)
    y_indices = np.round(y_coords + center_y).astype(int)

    # Ensure indices are within bounds before modifying them
    valid_mask = (x_indices >= 0) & (x_indices < Nx) & (y_indices >= 0) & (y_indices < Ny)

    # Apply valid_mask BEFORE the loop
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    x_coords = x_coords[valid_mask]  # Keep only valid x-coordinates

    # Extract values along the line using interpolation
    lineout_values = map_coordinates(autocorr_avg, [y_indices, x_indices], order=1)

    # Average over a band of width=10 pixels around the inclined line
    summed_values = np.zeros_like(lineout_values)
    for offset in range(-width//2, width//2):
        y_offset = y_indices + offset
        valid_offset_mask = (y_offset >= 0) & (y_offset < Ny)
        summed_values[valid_offset_mask] += map_coordinates(
            autocorr_avg, [y_offset[valid_offset_mask], x_indices[valid_offset_mask]], order=1
        )

    # Convert x-coordinates to microns using detector scale
    detector_pixel_size_um = 13  # Each pixel is 13 µm
    x_coords_um = x_coords * detector_pixel_size_um  # Now this matches the valid indices

    return x_coords_um, summed_values

# ----------------------------------------------------------------------------
# Function 3: Save configuration and results
# ----------------------------------------------------------------------------
def save_results(config, autocorr_avg, vertical_sum):
    """
    Save configuration parameters to a JSON file, and save the autocorrelation image and vertical lineout.
    
    Parameters:
      config: dictionary with configuration parameters.
      autocorr_avg: 2D autocorrelation array.
      vertical_sum: 1D vertical lineout.
    """
    # Save configuration to JSON.
    config_filename = f"out_files/Config_wavelength{config['wavelength']:.2e}_z{config['z_prop']:.2f}_detPix{config['detector_pixel_size']:.2e}_dx{config['dx_source']:.2e}_gaussWidth{config['gauss_width']:.2e}_stripePeriod{config['stripe_period']:.2e}_N{config['num_shots']}_M{config['num_modes_per_shot']}.json"
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_filename}")
    
    # Save autocorrelation image as a TIFF.
    autocorr_filename = f"out_files/Autocorr_wavelength{config['wavelength']:.2e}_z{config['z_prop']:.2f}_detPix{config['detector_pixel_size']:.2e}_dx{config['dx_source']:.2e}_gaussWidth{config['gauss_width']:.2e}_stripePeriod{config['stripe_period']:.2e}_N{config['num_shots']}_M{config['num_modes_per_shot']}.tiff"
    plt.imsave(autocorr_filename, autocorr_avg, cmap='inferno')
    print(f"Autocorrelation saved to {autocorr_filename}")
    
    # Save vertical lineout as a text file.
    lineout_filename = f"out_files/Lineout_wavelength{config['wavelength']:.2e}_z{config['z_prop']:.2f}_detPix{config['detector_pixel_size']:.2e}_dx{config['dx_source']:.2e}_gaussWidth{config['gauss_width']:.2e}_stripePeriod{config['stripe_period']:.2e}_N{config['num_shots']}_M{config['num_modes_per_shot']}.txt"
    np.savetxt(lineout_filename, vertical_sum)
    print(f"Lineout saved to {lineout_filename}")

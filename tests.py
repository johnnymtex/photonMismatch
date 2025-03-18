import matplotlib.pyplot as plt
import LightPipes as lp
import numpy as np

from focal_spot_pattern import create_gaussian_mask
from propagation import fresnel_propagation

grid_size = 50e-6
grid_dimension = 500
lambda_ = 2.75e-10
I0 = 1e6

F_in = lp.Begin(grid_size, lambda_, grid_dimension)
F_in = lp.GaussAperture(F_in, 10e-6)
F_in.field = np.sqrt(I0) * F_in.field

x = np.linspace(-25e-6, 25e-6, 500)
y = np.linspace(-25e-6, 25e-6, 500)
X,Y = np.meshgrid(x,y)

E_in = np.sqrt(I0)*create_gaussian_mask(X, Y, diameter=20e-6)

fig, ax = plt.subplots(ncols=2, figsize=(8,4))

im1=ax[0].imshow(np.abs(F_in.field)**2, cmap='inferno')
im2=ax[1].imshow(np.abs(E_in)**2, cmap='inferno')

fig.colorbar(im1, ax=ax[0], shrink=0.75)
fig.colorbar(im2, ax=ax[1], shrink=0.75)

F_out = lp.Fresnel(F_in, 1)
print(np.abs(F_out.field)**2)

E_out, _, _ = fresnel_propagation(E_in, lambda_, 1, 0.1e-6)

fig, ax = plt.subplots(ncols=2, figsize=(8,4))

im1=ax[0].imshow(np.abs(F_out.field)**2, cmap='inferno')
im2=ax[1].imshow(np.abs(E_out)**2, cmap='inferno')

fig.colorbar(im1, ax=ax[0], shrink=0.75)
fig.colorbar(im2, ax=ax[1], shrink=0.75)
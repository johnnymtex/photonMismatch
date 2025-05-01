# %%
from propagation import fresnel_propagation
from propagation_LP import fresnel_propagation as fresnel_lp
from focal_spot_pattern import create_gaussian_mask

import numpy as np
import matplotlib.pyplot as plt
import LightPipes as lp
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# %%
# initial simulation settings
grid_dimension = 500
grid_size = 700e-6

gauss_width = 75e-6

wavelength = 2.75e-10
z = 1
k = 2 * np.pi / wavelength

#grid_size = np.sqrt(grid_dimension*wavelength*z)
dx = grid_size/grid_dimension

x = dx * np.arange(-int(grid_dimension/2), grid_dimension-int(grid_dimension/2))
y = dx * np.arange(-int(grid_dimension/2), grid_dimension-int(grid_dimension/2))
X, Y = np.meshgrid(x,y)


I0 = 1e6

# %%
# Fresnel and sampling conditions
print("Fresnel check...")
print(f"N_f = {gauss_width**2/(wavelength*z)}\n")

print("\nSampling conditions?")
print(bool(x[1]-x[0] < wavelength*z/grid_size))

print(f"\n{wavelength*z/(grid_dimension*(x[1]-x[0])**2)}")

# %%
# Waist evolution function (just declaration)
def w(w0, lambda_, z):
    print("wz = ", w0 * np.sqrt(1 + (lambda_*z/(np.pi*w0*w0))**2))
    return w0 * np.sqrt(1 + (lambda_*z/(np.pi*w0*w0))**2)

# %%
# Theoretical solutions for intensity
theoretical_start = I0 * np.exp(-2*(X**2+Y**2)/(gauss_width*gauss_width))
theoretical_prop = I0 * (gauss_width/w(gauss_width, wavelength, z))**2 * np.exp(-2*(X**2+Y**2)/(w(gauss_width, wavelength, z)**2))

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(theoretical_start), np.min(theoretical_prop)])
vmax = np.max([np.max(theoretical_start), np.max(theoretical_prop)])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(theoretical_start, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(theoretical_prop, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, theoretical_start[int(len(theoretical_start)/2)])
ax3.plot(x*1e6, theoretical_prop[int(len(theoretical_prop)/2)], color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True

axins = inset_axes(ax3, width="35%", height="35%", loc='upper right', borderpad=1)
axins.plot(x*1e6, theoretical_start[int(len(theoretical_start)/2)])
axins.plot(x*1e6, theoretical_prop[int(len(theoretical_prop)/2)], '--', c="black")

axins.tick_params(labelleft=False,labelbottom=False)

# Set the x and y limits for the inset
x1, x2, y1, y2 = -25, 25, 0.95e6, 1.02e6  # Customize this to zoom into the area of interest
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

mark_inset(ax3, axins, loc1=3, loc2=1, fc="none", ec="0",zorder=10)

plt.show()

print("Tests for energy conservation...")
print(f"START: {np.sum(theoretical_start)}")
print(f"END: {np.sum(theoretical_prop)}")

# %%
# My solution for intensity
E_in = np.sqrt(I0)*np.ones((grid_dimension, grid_dimension), dtype=np.complex128)
E_in *= create_gaussian_mask(X, Y, gauss_width)

E_out, start, end = fresnel_propagation(E_in, wavelength, z, dx, padding_factor=2)
E_out = E_out[start:end, start:end]

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.abs(E_in)**2), np.min(np.abs(E_out)**2)])
vmax = np.max([np.max(np.abs(E_in)**2), np.max(np.abs(E_out)**2)])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.abs(E_in)**2, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.abs(E_out)**2, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, np.abs(E_in[int(len(E_in)/2)])**2)
ax3.plot(x*1e6, np.abs(E_out[int(len(E_out)/2)])**2, color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
# ax3.axhline(y=1/(np.exp(1)**2)*I0)
# ax3.axvline(x=gauss_width)

axins = inset_axes(ax3, width="35%", height="35%", loc='upper right', borderpad=1)
axins.plot(x*1e6, np.abs(E_in[int(len(E_in)/2)])**2)
axins.plot(x*1e6, np.abs(E_out[int(len(E_out)/2)])**2, '--', c="black")

axins.tick_params(labelleft=False,labelbottom=False)

# Set the x and y limits for the inset
x1, x2, y1, y2 = -25, 25, 0.95e6, 1.02e6  # Customize this to zoom into the area of interest
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

mark_inset(ax3, axins, loc1=3, loc2=1, fc="none", ec="0",zorder=10)

plt.show()

print("Tests for energy conservation...")
print(f"START: {np.sum(np.abs(E_in)**2)}")
print(f"END: {np.sum(np.abs(E_out)**2)}")

E_out_copy = E_out.copy()

# %%
# LightPipes solution for intensity
LP_in = lp.Begin(grid_size, wavelength, grid_dimension, dtype=np.complex128)
LP_in.field *= np.sqrt(I0)
LP_in = lp.GaussAperture(LP_in, 1/np.sqrt(2)*gauss_width)

LP_out = lp.Forvard(LP_in, z)

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.abs(LP_in.field)**2), np.min(np.abs(LP_out.field)**2)])
vmax = np.max([np.max(np.abs(LP_in.field)**2), np.max(np.abs(LP_out.field)**2)])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.abs(LP_in.field)**2, extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.abs(LP_out.field)**2, extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(LP_in.xvalues*1e6, np.abs(LP_in.field[int(len(LP_in.field)/2)])**2)
ax3.plot(LP_in.xvalues*1e6, np.abs(LP_out.field[int(len(LP_out.field)/2)])**2, color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
# ax3.axhline(y=1/(np.exp(1)**2)*I0)
# ax3.axvline(x=gauss_width)

axins = inset_axes(ax3, width="35%", height="35%", loc='upper right', borderpad=1)
axins.plot(LP_in.xvalues*1e6, np.abs(LP_in.field[int(len(LP_in.field)/2)])**2)
axins.plot(LP_in.xvalues*1e6, np.abs(LP_out.field[int(len(LP_out.field)/2)])**2, '--', c="black")

axins.tick_params(labelleft=False,labelbottom=False)

# Set the x and y limits for the inset
x1, x2, y1, y2 = -25, 25, 0.95e6, 1.02e6  # Customize this to zoom into the area of interest
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

mark_inset(ax3, axins, loc1=3, loc2=1, fc="none", ec="0",zorder=10)

plt.show()

print("Tests for energy conservation...")
print(f"START: {np.sum(np.abs(LP_in.field)**2)}")
print(f"END: {np.sum(np.abs(LP_out.field)**2)}")
print(np.sum(np.abs(LP_out.field)**2)/np.sum(np.abs(LP_in.field)**2))

# %%
# computing theoretical solutions (full field)
theoretical_start_2 = np.sqrt(I0) * np.exp(-(X**2+Y**2)/(gauss_width*gauss_width))

def zR(w0, lambda_):
    return np.pi*w0*w0/lambda_

# computing theoretical waist value after propagation
def w(w0, lambda_, z):
    return w0 * np.sqrt(1 + (lambda_*z/(np.pi*w0*w0))**2)

def final_field(x, y, I0, w0, z, lambda_):
    ww = w(w0, lambda_, z)
    zzR = zR(w0, lambda_)
    R = z * (1 + (zzR/z)**2)
    psi = np.atan2(z, zzR)
    print(psi)

    return np.sqrt(I0) * w0/ww * np.exp(-(x**2+y**2)/(ww*ww)) * np.exp(-1j * (k*z + k*(x**2+y**2)/(2*R) - psi))

theoretical_prop_2 = final_field(X, Y, I0, gauss_width, z, wavelength)

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.angle(theoretical_start_2)), np.min(np.angle(theoretical_prop_2))])
vmax = np.max([np.max(np.angle(theoretical_start_2)), np.max(np.angle(theoretical_prop_2))])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.angle(theoretical_start_2), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.angle(theoretical_prop_2), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, np.angle(theoretical_start_2[int(len(theoretical_start_2)/2)]))
ax3.plot(x*1e6, np.angle(theoretical_prop_2[int(len(theoretical_prop_2)/2)]), color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Phase [rad]")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True

plt.show()

# %%
# my method (phase)
fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.angle(E_in)), np.min(np.angle(E_out))])
vmax = np.max([np.max(np.angle(E_in)), np.max(np.angle(E_out))])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.angle(E_in), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.angle(E_out), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, np.angle(E_in[int(len(E_in)/2)]))
ax3.plot(x*1e6, np.angle(E_out[int(len(E_out)/2)]), color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
# ax3.axhline(y=1/(np.exp(1))*I0)
# ax3.axvline(x=gauss_width)

plt.show()

# %%
# LightPipes method (phase)
fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.angle(LP_in.field)), np.min(np.angle(LP_out.field))])
vmax = np.max([np.max(np.angle(LP_in.field)), np.max(np.angle(LP_out.field))])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.angle(LP_in.field), extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.angle(LP_out.field), extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(LP_in.xvalues*1e6, np.angle(LP_in.field[int(len(LP_in.field)/2)]))
ax3.plot(LP_in.xvalues*1e6, np.angle(LP_out.field[int(len(LP_out.field)/2)]), color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
# ax3.axhline(y=1/(np.exp(1))*I0)
# ax3.axvline(x=gauss_width)

plt.show()

# %%
# new functions
def zR(w0, lambda_):
    return np.pi*w0*w0/lambda_

def w(w0, lambda_, z):
    return w0 * np.sqrt(1 + (lambda_*z/(np.pi*w0*w0))**2)

def final_field(x, y, I0, w0, z, lambda_, random_phase):
    ww = w(w0, lambda_, z)
    zzR = zR(w0, lambda_)
    R = z * (1 + (zzR/z)**2)
    psi = np.atan2(z, zzR)
    print(psi)

    return np.sqrt(I0) * w0/ww * np.exp(-(x**2+y**2)/(ww*ww)) * np.exp(-1j * (k*z + k*(x**2+y**2)/(2*R) - psi - random_phase))

# %%
# randomized phase
random_phase = np.random.uniform(-np.pi, np.pi, (grid_dimension, grid_dimension))

# %%
# theoretical solutions with random phase (intensity plots)
theoretical_start = np.sqrt(I0) * np.exp(-(X**2+Y**2)/(gauss_width*gauss_width)) * np.exp(1j * random_phase)
theoretical_prop = final_field(X, Y, I0, gauss_width, z, wavelength, random_phase)

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.abs(theoretical_start)**2), np.min(np.abs(theoretical_prop)**2)])
vmax = np.max([np.max(np.abs(theoretical_start)**2), np.max(np.abs(theoretical_prop)**2)])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.abs(theoretical_start)**2, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.abs(theoretical_prop)**2, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, np.abs(theoretical_start[int(len(theoretical_start)/2)])**2)
ax3.plot(x*1e6, np.abs(theoretical_prop[int(len(theoretical_prop)/2)])**2, color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True

axins = inset_axes(ax3, width="35%", height="35%", loc='upper right', borderpad=1)
axins.plot(x*1e6, np.abs(theoretical_start[int(len(theoretical_start)/2)])**2)
axins.plot(x*1e6, np.abs(theoretical_prop[int(len(theoretical_prop)/2)])**2, '--', c="black")

axins.tick_params(labelleft=False,labelbottom=False)

# Set the x and y limits for the inset
x1, x2, y1, y2 = -25, 25, 0.95e6, 1.02e6  # Customize this to zoom into the area of interest
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

mark_inset(ax3, axins, loc1=3, loc2=1, fc="none", ec="0",zorder=10)

plt.show()

print("Tests for energy conservation...")
print(f"START: {np.sum(np.abs(theoretical_start)**2)}")
print(f"END: {np.sum(np.abs(theoretical_prop)**2)}")

# %%
# custom code (intensity plots)
def cosine_taper_2d(x, y, gauss_width, Lx, gaussian):
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    taper = np.ones((len(x), len(y)), dtype=np.complex128)

    taper[r > (Lx/2)] = 0 + 0j
    taper[r <= (Lx/2)] = 0.5 * (1 + np.cos(np.pi * (r[r <= Lx/2]) / (Lx/2)))

    return taper

def cosine_taper_2d_backup(x, y, gauss_width, Lx, gaussian):
    reduced_lx = (gauss_width + Lx/2)/2

    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    taper = np.ones((len(x), len(y)), dtype=np.complex128)

    taper[r > reduced_lx] = 0 + 0j
    taper[(r <= reduced_lx) & (r > gauss_width)] = 0.5 * (1 + np.cos(np.pi * (r[(r <= reduced_lx) & (r > gauss_width)] - gauss_width) / (reduced_lx - gauss_width)))

    return taper

E_in = np.sqrt(I0)*np.ones((grid_dimension, grid_dimension), dtype=np.complex128)
E_in *= create_gaussian_mask(X, Y, gauss_width)

mask = cosine_taper_2d_backup(x, y, gauss_width, grid_size, create_gaussian_mask(X, Y, gauss_width))
print(mask[int(len(mask)/2)])

E_in *= mask
E_in *= np.exp(1j * random_phase)

E_out, start, end = fresnel_propagation(E_in, wavelength, z, dx, padding_factor=2)
E_out = E_out[start:end, start:end]

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.abs(E_in)**2), np.min(np.abs(E_out)**2)])
vmax = 1e6 #np.max([np.max(np.abs(E_in)**2), np.max(np.abs(E_out)**2)])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.abs(E_in)**2, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.abs(E_out)**2, extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, np.abs(E_in[int(len(E_in)/2)])**2)
ax3.plot(x*1e6, np.abs(E_out[int(len(E_out)/2)])**2, color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
ax3.plot(x*1e6, mask[int(len(mask)/2)]*1e6, color="red", linestyle="dashed")
# ax3.axhline(y=1/(np.exp(1)**2)*I0)
# ax3.axvline(x=gauss_width)

plt.show()

print("Tests for energy conservation...")
print(f"START: {np.sum(np.abs(E_in)**2)}")
print(f"END: {np.sum(np.abs(E_out)**2)}")

E_out_copy = E_out.copy()

# plt.figure()
# plt.plot(np.abs(E_in[int(len(E_in)/2)])**2)
# plt.ylim(0,0.3)
# %%
# LightPipes method (intensity plots)

LP_in = lp.Begin(grid_size, wavelength, grid_dimension, dtype=np.complex128)
LP_in.field *= np.sqrt(I0) * np.exp(1j * random_phase)
LP_in = lp.GaussAperture(LP_in, 1/np.sqrt(2)*gauss_width)

LP_out = lp.Forvard(LP_in, z)

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.abs(LP_in.field)**2), np.min(np.abs(LP_out.field)**2)])
vmax = 1e6 #np.max([np.max(np.abs(LP_in.field)**2), np.max(np.abs(LP_out.field)**2)])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.abs(LP_in.field)**2, extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.abs(LP_out.field)**2, extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(LP_in.xvalues*1e6, np.abs(LP_in.field[int(len(LP_in.field)/2)])**2)
ax3.plot(LP_in.xvalues*1e6, np.abs(LP_out.field[int(len(LP_out.field)/2)])**2, color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
ax3.plot(x*1e6, mask[int(len(mask)/2)]*1e6, color="red", linestyle="dashed")
# ax3.axhline(y=1/(np.exp(1)**2)*I0)
# ax3.axvline(x=gauss_width)

plt.show()

print("Tests for energy conservation...")
print(f"START: {np.sum(np.abs(LP_in.field)**2)}")
print(f"END: {np.sum(np.abs(LP_out.field)**2)}")
print(np.sum(np.abs(LP_out.field)**2)/np.sum(np.abs(LP_in.field)**2))

# %%
# theoretical solutions with random phase (phase plots)

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.angle(theoretical_start)), np.min(np.angle(theoretical_prop))])
vmax = np.max([np.max(np.angle(theoretical_start)), np.max(np.angle(theoretical_prop))])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.angle(theoretical_start), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.angle(theoretical_prop), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, np.angle(theoretical_start[int(len(theoretical_start)/2)]))
ax3.plot(x*1e6, np.angle(theoretical_prop[int(len(theoretical_prop)/2)]), color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Phase [rad]")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True

plt.show()

# %%
# custom code (phase plots)

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.angle(E_in)), np.min(np.angle(E_out))])
vmax = np.max([np.max(np.angle(E_in)), np.max(np.angle(E_out))])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.angle(E_in), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.angle(E_out), extent=(np.min(x)*1e6, np.max(x)*1e6, np.min(y)*1e6, np.max(y)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(x*1e6, np.angle(E_in[int(len(E_in)/2)]))
ax3.plot(x*1e6, np.angle(E_out[int(len(E_out)/2)]), color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
# ax3.axhline(y=1/(np.exp(1))*I0)
# ax3.axvline(x=gauss_width)

plt.show()

# %%
# LightPipes method (phase plots)

fig = plt.figure(figsize=(6,9))

vmin = np.min([np.min(np.angle(LP_in.field)), np.min(np.angle(LP_out.field))])
vmax = np.max([np.max(np.angle(LP_in.field)), np.max(np.angle(LP_out.field))])

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0:2,0])
im0 = ax1.imshow(np.angle(LP_in.field), extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax1.set_ylabel("y [$\mu$m]")
ax1.set_xlabel("x [$\mu$m]")

ax2 = fig.add_subplot(gs[0:2,1])
im1 = ax2.imshow(np.angle(LP_out.field), extent=(np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6, np.min(LP_in.xvalues)*1e6, np.max(LP_in.xvalues)*1e6), cmap="Greys_r", vmin=vmin, vmax=vmax)
ax2.set_xlabel("x [$\mu$m]")
ax2.tick_params(labelleft=False)

im0.set_clim(vmin, vmax)
im1.set_clim(vmin, vmax)

fig.subplots_adjust(hspace=-0.375, wspace=0.05)
fig.colorbar(im0, ax=[ax1,ax2], location='top')

ax3 = fig.add_subplot(gs[2,:])
ax3.plot(LP_in.xvalues*1e6, np.angle(LP_in.field[int(len(LP_in.field)/2)]))
ax3.plot(LP_in.xvalues*1e6, np.angle(LP_out.field[int(len(LP_out.field)/2)]), color="black", linestyle="dashed")
ax3.set_xlabel("x [$\mu$m]")
ax3.set_ylabel("Intensity")
ax3.ticklabel_format(style="sci", axis='y', scilimits=(-3,2))
ax3.yaxis.major.formatter._useMathText = True
# ax3.axhline(y=1/(np.exp(1))*I0)
# ax3.axvline(x=gauss_width)

plt.show()
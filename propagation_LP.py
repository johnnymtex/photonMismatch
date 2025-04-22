import numpy as np
import matplotlib.pyplot as plt

from scipy.special import fresnel as _fresnel

def combined_propagation(field, lam, z, dx):
    return fresnel_propagation(field, lam, z, dx)

def fresnel_propagation(field, lam, z, dx, dtype=np.complex128):
    """
    Separated the "math" logic out so that only standard and numpy types
    are used.
    
    Parameters
    ----------
    z : float
        Propagation distance.
    field : ndarray
        2d complex numpy array (NxN) of the field.
    dx : float
        In units of sim (usually [m]), spacing of grid points in field.
    lam : float
        Wavelength lambda in sim units (usually [m]).
    dtype : dtype
        complex dtype of the array.
    usepyFFTW : bool
        Use the pyFFTW package if True
        Use numpy FFT if False

    Returns
    -------
    ndarray (2d, NxN, complex)
        The propagated field.

    """

    N = field.shape[0] #assert square
    
    kz = 2.*3.141592654/lam * z
    siz = N*dx
    dx = siz/(N-1) #like old Cpp code, even though unlogical
    
    cokz = np.cos(kz)
    sikz = np.sin(kz)
    
    No2 = int(N/2) #"N over 2"

    in_outF = np.zeros((2*N, 2*N),dtype=dtype)
    in_outK = np.zeros((2*N, 2*N),dtype=dtype)

    # Create the sign-flip pattern for largest use case and 
    # reference smaller grids with a view to the same data for
    # memory saving.
    ii2N = np.ones((2*N),dtype=float)
    ii2N[1::2] = -1 #alternating pattern +,-,+,-,+,-,...
    iiij2N = np.outer(ii2N, ii2N)
    iiij2No2 = iiij2N[:2*No2,:2*No2] #slice to size used below
    iiijN = iiij2N[:N, :N]

    RR = np.sqrt(1/(2*lam*z))*dx*2
    io = np.arange(0, (2*No2)+1) #add one extra to stride fresnel integrals
    R1 = RR*(io - No2)
    fs, fc = _fresnel(R1)
    fss = np.outer(fs, fs) #    out[i, j] = a[i] * b[j]
    fsc = np.outer(fs, fc)
    fcs = np.outer(fc, fs)
    fcc = np.outer(fc, fc)

    temp_re = (fsc[1:, 1:] #s[i+1]c[j+1]
               + fcs[1:, 1:]) #c[+1]s[+1]
    temp_re -= fsc[:-1, 1:] #-scp [p=+1, without letter =+0]
    temp_re -= fcs[:-1, 1:] #-csp
    temp_re -= fsc[1:, :-1] #-spc
    temp_re -= fcs[1:, :-1] #-cps
    temp_re += fsc[:-1, :-1] #sc
    temp_re += fcs[:-1, :-1] #cs
    
    temp_im = (-fcc[1:, 1:] #-cpcp
               + fss[1:, 1:]) # +spsp
    temp_im += fcc[:-1, 1:] # +ccp
    temp_im -= fss[:-1, 1:] # -ssp
    temp_im += fcc[1:, :-1] # +cpc
    temp_im -= fss[1:, :-1] # -sps
    temp_im -= fcc[:-1, :-1] # -cc
    temp_im += fss[:-1, :-1]# +ss
    
    temp_K = 1j * temp_im # a * b creates copy and casts to complex
    temp_K += temp_re
    temp_K *= iiij2No2
    temp_K *= 0.5
    in_outK[(N-No2):(N+No2), (N-No2):(N+No2)] = temp_K
    
    in_outF[(N-No2):(N+No2), (N-No2):(N+No2)] \
        = field[(N-2*No2):N,(N-2*No2):N] #cutting off field if N odd (!)
    in_outF[(N-No2):(N+No2), (N-No2):(N+No2)] *= iiij2No2
    
    in_outK = np.fft.fft2(in_outK)
    in_outF = np.fft.fft2(in_outF)
    
    in_outF *= in_outK
    
    in_outF *= iiij2N
    in_outF = np.fft.ifft2(in_outF)
    
    Ftemp = (in_outF[No2:N+No2, No2:N+No2]
             - in_outF[No2-1:N+No2-1, No2:N+No2])
    Ftemp += in_outF[No2-1:N+No2-1, No2-1:N+No2-1]
    Ftemp -= in_outF[No2:N+No2, No2-1:N+No2-1]
    comp = complex(cokz, sikz)
    Ftemp *= 0.25 * comp
    Ftemp *= iiijN

    field = Ftemp #reassign without data copy

    return field
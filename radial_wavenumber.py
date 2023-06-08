def radial_wavenumber(kx,ky,dkx,dky):
    wv = np.sqrt(kx[:,None]**2+ky[None,:]**2)
    if kx.max()>ky.max():
        kmax = ky.max()
    else:
        kmax = kx.max()
        dkr = np.sqrt(dkx**2 + dky**2)
        kr =  np.arange(dkr/2.,kmax+dkr/2.,dkr)
    return wv, kr, dkr

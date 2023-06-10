def compute_KE_time_series_bands(u,v,dx,dy,kmin,kmax):
    u_fs = fourier_series(np.array(u),dx,dy,kmin,kmax)
    v_fs = fourier_series(np.array(v),dx,dy,kmin,kmax)
    ke = np.mean(0.5*u_fs**2+v_fs**2, axis=(0,1))
    del u, v, u_fs, v_fs
    return ke
  
  def fourier_series(var,dx,dy,kmin,kmax):
    nx, ny, nt = var.shape
    kx, ky = fft.fftfreq(nx,dx), fft.fftfreq(ny,dy)
    dkx, dky = 1/(dx*nx), 1/(dy*ny)
    wv = np.sqrt(kx[:,None]**2+ky[None,:]**2)
    if kx.max()>ky.max():
        kmax = ky.max()
    else:
        kmax = kx.max()
    dkr = np.sqrt(dkx**2 + dky**2)
    kr =  np.arange(dkr/2.,kmax+dkr/2.,dkr)
    msk = (wv >= kmin) & (wv <= kmax)
    fft_var = fft.fft2(np.array(var),axes=(0,1))
    del var, kx, ky
    var_fs = fft.ifft2(fft_var*msk[:,:,None],axes=(0,1)).real
    del fft_var
    return var_fs

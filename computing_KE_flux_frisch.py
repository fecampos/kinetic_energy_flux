def computing_KE_flux_frisch(u, v, dist_x, dist_y, dx, dy):    
    nx, ny,nt = u.shape
    kx, ky = fft.fftfreq(nx,dx), fft.fftfreq(ny,dy)
    dkx, dky = 1/(dx*nx), 1/(dy*ny)

    u_detrend = signal.detrend(u.data,axis=0,type="linear")
    u_detrend = signal.detrend(u_detrend,axis=1,type="linear")    
    v_detrend = signal.detrend(v.data,axis=0,type="linear")
    v_detrend = signal.detrend(v_detrend,axis=1,type="linear")
    
    win1, win2 =  np.hanning(nx), np.hanning(ny)       
    win1, win2= (nx/(win1**2).sum())*win1, (ny/(win2**2).sum())*win2
    win = win1[:,None,None]*win2[None,:,None]

    u_detrend, v_detrend = u_detrend*win, v_detrend*win     

    fftu, fftv = fft.fft2(u_detrend,axes=(0,1)), fft.fft2(v_detrend,axes=(0,1))
    area = dist_x[:,:,None]*dist_y[:,:,None]

    wv, kr, dkr = radial_wavenumber(kx,ky,dkx,dky)    
    ke_flux_ns, ke_flux_ss = [], []
    for ii in range(kr.size):
        kr_g, kr_l =  wv>kr[ii], wv<kr[ii]
        ug = fft.ifft2(fftu*kr_g[:,:,None],axes=(0,1)).real
        vg = fft.ifft2(fftv*kr_g[:,:,None],axes=(0,1)).real
        ul = fft.ifft2(fftu*kr_l[:,:,None],axes=(0,1)).real
        vl = fft.ifft2(fftv*kr_l[:,:,None],axes=(0,1)).real
        ns = (ul*(ug+ul)*np.gradient(dist_y[:,:,None]*ug, axis=0, edge_order=2)+\
              vl*(vg+vl)*np.gradient(dist_x[:,:,None]*vg, axis=1, edge_order=2))/area
        ss = (vl*(ug+ul)*np.gradient(dist_y[:,:,None]*vg, axis=0, edge_order=2)+\
              ul*(vg+vl)*np.gradient(dist_x[:,:,None]*ug, axis=1, edge_order=2))/area               
        ke_flux_ns.append(np.mean(ns))
        ke_flux_ss.append(np.mean(ss))

    return kr, np.array(ke_flux_ss), np.array(ke_flux_ns)

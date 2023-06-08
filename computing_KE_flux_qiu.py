def computing_KE_flux_qiu(u, v, dist_x, dist_y, dx, dy):
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
       
    advx = (u_detrend*np.gradient(dist_y[:,:,None]*u_detrend, axis=0, edge_order=2)+\
            v_detrend*np.gradient(dist_x[:,:,None]*u_detrend, axis=1, edge_order=2))/area
    advy = (u_detrend*np.gradient(dist_y[:,:,None]*v_detrend, axis=0, edge_order=2)+\
            v_detrend*np.gradient(dist_x[:,:,None]*v_detrend, axis=1, edge_order=2))/area    
    
    fft_advx, fft_advy = fft.fft2(advx, axes=(0,1)), fft.fft2(advy, axes=(0,1))
        
    fft_ke_adv = -fft.fftshift((fftu.conj()*fft_advx+fftv.conj()*fft_advy).real/(nx*ny)**2,axes=(0,1))
        
    kr, ke_flux = computing_pi_qiu(fft_ke_adv,fft.fftshift(kx), fft.fftshift(ky),dkx,dky)

    return kr, ke_flux

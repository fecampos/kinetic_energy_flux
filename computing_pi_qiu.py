def computing_pi_qiu(ke_flux,kx,ky,dkx,dky):
    wv, kr, dkr = radial_wavenumber(kx,ky,dkx,dky)
    nx, ny, nt = ke_flux.shape
    nr = kr.size
    Er = np.zeros((nr,nt))
    for i in range(kr.size):
        fkr = (wv>=kr[i]-kr[0]) & (wv<=kr[i])
        Er[i,:] = np.sum(ke_flux*fkr[:,:,None],axis=(0,1))        
    ke_flux = np.mean(np.cumsum(Er[::-1],axis=0),axis=1)[::-1]
    return kr, ke_flux

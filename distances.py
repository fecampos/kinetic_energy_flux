def distances(longitude, latitude):
    nx, ny = longitude.size, latitude.size
    lony, laty = np.meshgrid(longitude, latitude)
    latx, lonx = np.meshgrid(latitude, longitude)
    dx, dy = gsw.geostrophy.distance(lonx, latx), gsw.geostrophy.distance(lony, laty)
    distx, disty = np.zeros((ny,nx)), np.zeros((ny,nx))
    disty[:,1:] = dy
    disty[:,0] = dy[:,0]
    distx[1:,:] = np.transpose(dx)
    distx[0,:] = dx[:,0]
    return distx, disty

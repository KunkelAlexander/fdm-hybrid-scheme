import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate the Schrodinger-Poisson system with the Spectral method
"""

def transferFunction(x):

    x1 = 1.8
    x2 = 2.0
    x3 = 2.2

    r = 1.0*np.exp( -(x - x1)**2/0.1 ) +  0.1*np.exp( -(x - x2)**2/0.1 ) +  0.1*np.exp( -(x - x3)**2/0.1 )
    g = 1.0*np.exp( -(x - x1)**2/0.1 ) +  1.0*np.exp( -(x - x2)**2/0.1 ) +  0.1*np.exp( -(x - x3)**2/0.1 )
    b = 0.1*np.exp( -(x - x1)**2/0.1 ) +  0.1*np.exp( -(x - x2)**2/0.1 ) +  1.0*np.exp( -(x - x3)**2/0.1 )
    a = 0.6*np.exp( -(x - x1)**2/0.1 ) +  0.1*np.exp( -(x - x2)**2/0.1 ) + 0.01*np.exp( -(x - x3)**2/0.1 )
    return r,g,b,a


""" Volume Rendering """

def load(data):
    # Load Datacube
    #data = np.loadtxt(filename, skiprows=1)
    print(data[data[:,0]==1].shape)
    pid1 = data[data[:,0]==1]
    pid1[:, 1:4] += 6
    print(pid1[:10, 3])
    datacube = np.zeros((28, 28, 28))
    datacube[pid1[:, 1].astype(int), pid1[:, 2].astype(int), pid1[:, 3].astype(int)] = pid1[:, 4]
    #datacube = np.array(f['density'])
    return datacube


def render(datacube, filename):
    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx/2, Nx/2, Nx)
    y = np.linspace(-Ny/2, Ny/2, Ny)
    z = np.linspace(-Nz/2, Nz/2, Nz)
    points = (x, y, z)

    # Do Volume Rendering at Different Veiwing Angles
    Nangles = 4

    for i in range(Nangles):

        print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')

        # Camera Grid / Query Points -- rotate camera view
        angle = np.pi/2 * i / Nangles
        N = 10
        c = np.linspace(-N/2, N/2, N)
        qx, qy, qz = np.meshgrid(c,c,c)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle) 
        qzR = qy * np.sin(angle) + qz * np.cos(angle)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
        
        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
        
        # Do Volume Rendering
        image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

        for dataslice in camera_grid:
            r,g,b,a = transferFunction(np.log(dataslice))
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]
        
        image = np.clip(image,0.0,1.0)
        
        # Plot Volume Rendering
        plt.figure(figsize=(4,4), dpi=80)
        
        plt.imshow(image)
        plt.axis('off')
        
        # Save figure
        plt.savefig('3d_plots/volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)



    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4,4), dpi=80)

    plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
    plt.clim(-5, 5)
    plt.axis('off')

    # Save figure
    plt.savefig('3d_plots/projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    plt.show()
	
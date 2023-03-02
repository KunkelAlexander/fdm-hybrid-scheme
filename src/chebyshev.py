import scipy
import numpy as np

def grid(N, x0, x1):
    i=np.linspace(0,N,N+1)
    return (x1-x0)/2.*(-np.cos(i*np.pi/N))+(x0+x1)/2.0

def dx_fft_real(v, x0, x1):
    '''Chebyshev differentiation via fft.
    Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    N = len(v)-1
    if N == 0:
        w = 0.0 # only when N is even!
        return w
    x  = np.cos(np.pi*np.arange(0,N+1)/N)
    ii = np.arange(0,N)
    V = np.flipud(v[1:N]); V = list(v) + list(V);
    U = np.real(np.fft.fft(V))
    b = list(ii); b.append(0); b = b + list(np.arange(1-N,0));
    w_hat = 1j*np.array(b)
    w_hat = w_hat * U
    W = np.real(np.fft.ifft(w_hat))
    w = np.zeros(N+1)
    w[1:N] = -W[1:N]/np.sqrt(1-x[1:N]**2)
    w[0] = sum(ii**2*U[ii])/N + 0.5*N*U[N]
    w[N] = sum((-1)**(ii+1)*ii**2*U[ii])/N + \
            0.5*(-1)**(N+1)*N*U[N]
    w *= -2/(x1-x0)
    return w

def dxn_fft_real(v, x0, x1, n):
    result = np.copy(v)
    for i in range(n):
        result = dx_fft_real(result, x0, x1)
    return result

def dxn_fft_complex(v, x0, x1, n):
    dre = np.real(v)
    dim = np.imag(v)
    for i in range(n):
        dre = dx_fft_real(dre, x0, x1)
        dim = dx_fft_real(dim, x0, x1)
    return dre + 1j* dim

def matrix(N, x0, x1):
    '''Chebushev polynomial differentiation matrix.
    Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    x      = np.cos(np.pi*np.arange(0,N+1)/N)
    if N%2 == 0:
        x[N//2] = 0.0 # only when N is even!
    c      = np.ones(N+1); c[0] = 2.0; c[N] = 2.0
    c      = c * (-1.0)**np.arange(0,N+1)
    c      = c.reshape(N+1,1)
    X      = np.tile(x.reshape(N+1,1), (1,N+1))
    dX     = X - X.T
    D      = np.dot(c, 1.0/c.T) / (dX+np.eye(N+1))
    D      = D - np.diag( D.sum(axis=1) )
    D     *= -2/(x1-x0)
    return D

def dxn_mat_complex(v, mat, n):
    dre = np.real(v)
    dim = np.imag(v)
    for i in range(n):
        dre = np.matmul(mat, dre)
        dim = np.matmul(mat, dim)
    return dre + 1j* dim

def poly(x,n):

    T0 = 1.0

    T1 = x

    if n == 0:

        return  T0

    elif n == 1:

        return  T1

    elif n > 1:

        for i in range(n-1):

            T = 2*x*T1-T0

            T0 = T1

            T1 = T

        return T

def coeff(y):
    N = len(y) - 1

    a = []

    y[0] *= 0.5

    y[N] *= 0.5

    x_chebgrid = grid(N,-1.0,1.0)

    for j in range(N+1):

        sum = 2.0 / N  * np.sum(y*poly(x_chebgrid,j))

        a.append(sum)
    y[0] /= 0.5

    y[N] /= 0.5

    return a

def interp(x,coeff, xorg):

    a, b   = xorg[0], xorg[-1]
    N = len(xorg) - 1

    a0, b0 = x[0], x[-1]
    if a0 < a or b0 > b:
        raise ValueError(f"Interpolation on [{a0}, {b0}] when original domain is [{a},{b}]")

    x_chebgrid = ( 2.0 * x - a  - b ) / ( b-a)

    sum  = coeff[0]*poly(x_chebgrid,0) * 0.5

    for j in range(1,N-1):

        sum += coeff[j]*poly(x_chebgrid,j)

    sum += coeff[N]*poly(x_chebgrid,N) * 0.5

    return sum
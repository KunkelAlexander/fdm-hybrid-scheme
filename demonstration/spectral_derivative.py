import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres

DCT1 = 1    # WSWS
DCT2 = 2    # HSHS
DCT3 = 3    # WSWA
DCT4 = 4    # HSHA
DST1 = 5    # WAWA
DST2 = 6    # HAHA
DST3 = 7    # WAWS
DST4 = 8    # HAHS
DFT1 = 9    # PERIODIC

NN       = DCT1
ND       = DCT3
DN       = DST3
DD       = DST1
PERIODIC = DFT1


DCT1 = 1    # WSWS
DCT2 = 2    # HSHS
DCT3 = 3    # WSWA
DCT4 = 4    # HSHA
DST1 = 5    # WAWA
DST2 = 6    # HAHA
DST3 = 7    # WAWS
DST4 = 8    # HAHS
DFT1 = 9    # PERIODIC

NN       = DCT1
ND       = DCT3
DN       = DST3
DD       = DST1
PERIODIC = DFT1

M_LINEAR      = 1
M_POLYNOMIAL  = 2
M_PLANE_WAVE  = 3
M_EVEN        = 4
M_DISABLED    = 5
M_COSINE      = 6

ONE_SIDED   = 1
CENTRAL     = 2

# options for finite difference functions
MODE_FORWARD = 0
MODE_CENTERED = 1
MODE_BACKWARD = 2
MODE_CUSTOM = 3


#Note:
def computeX(L0, L1, N):
    L  = L1 - L0
    dx = L/(N - 1)
    xx = np.arange(0, N) * dx + L0
    return xx, dx

# Return representative sample and boundary indices for psi depending on boundary conditions bc_ind
def selectBC(psi, bc_ind):
    Nx = len(psi)

    # define which transforms to use
    if bc_ind == NN:

        # Neumann-Neumann / WSWS
        T1 = DCT1

        # set indices for representative sample
        ind1 = 0
        ind2 = Nx

    elif bc_ind == ND:

        # Neumann-Dirichlet / WSWA
        T1 = DCT3

        # set indices for representative sample
        ind1 = 0
        ind2 = Nx - 1

    elif bc_ind == DN:

        # Dirichlet-Neumann / WAWS
        T1 = DST3

        # set indices for representative sample
        ind1 = 1
        ind2 = Nx

    elif bc_ind == DD:

        # Dirichlet-Dirichlet / WAWA
        T1 = DST1

        # set indices for representative sample
        ind1 = 1
        ind2 = Nx - 1

    elif bc_ind == PERIODIC:

        # Periodic
        T1 = DFT1

        # set indices for representative sample
        ind1 = 0
        ind2 = Nx - 1


    # assign initial condition for p, just selecting representative sample
    p = psi[ind1 : ind2]
    return p, ind1, ind2

# Restore full function from representative sample depending on boundary conditions
def restoreBC(psi, bc_ind):
    #Nx = len(psi)

    # define which transforms to use
    if bc_ind == NN:

        # Neumann-Neumann / WSWS
        #T1 = DCT1

        # set indices for representative sample
        #ind1 = 0
        #ind2 = Nx

        return psi

    elif bc_ind == ND:

        # Neumann-Dirichlet / WSWA
        #T1 = DCT3

        # set indices for representative sample
        #ind1 = 0
        #ind2 = Nx - 1

        # mode = "constant" pads with zeros by default
        psi = np.pad(psi, (0, 1), mode="constant")

    elif bc_ind == DN:

        # Dirichlet-Neumann / WAWS
        #T1 = DST3

        # set indices for representative sample
        #ind1 = 1
        #ind2 = Nx

        # mode = "constant" pads with zeros by default
        psi = np.pad(psi, (1, 0), mode="constant")

    elif bc_ind == DD:

        # Dirichlet-Dirichlet / WAWA
        #T1 = DST1

        # set indices for representative sample
        #ind1 = 1
        #ind2 = Nx - 1
        psi = np.pad(psi, (1, 1), mode="constant")

    elif bc_ind == PERIODIC:

        # Periodic
        #T1 = DFT1

        # set indices for representative sample
        #ind1 = 0
        #ind2 = Nx - 1

        # mode = "wrap" fills with the values from the other side of the array
        psi = np.pad(psi, (0, 1), mode="wrap")

    return psi

def computeK(p, dx, T1):
    N = len(p)

    #WSWS
    if T1 == DCT1:
        M = 2 * ( N - 1 )
        n = np.arange(0, int(M/2) + 1)
        k = 2 * np.pi / ( M * dx ) * n
    #HSHS
    elif T1 == DCT2:
        M = 2 * N
        n = np.arange(0, int(M/2)    )
        k = 2 * np.pi / ( M * dx ) * n
    #WAWA
    elif T1 == DST1:
        M = 2 * ( N + 1 )
        n = np.arange(1, int(M/2)    )
        k = 2 * np.pi / ( M * dx ) * n
    #HAHA
    elif T1 == DST2:
        M = 2 * N
        n = np.arange(1, int(M/2) + 1)
        k = 2 * np.pi / ( M * dx ) * n
    #WSWA or HSHA or WAWS or HAHS
    elif T1 == DCT3 or T1 == DCT4 or T1 == DST3 or T1 == DST4:
        M = 2 * N
        n = np.arange(0, int(M/2)   )
        k = 2 * np.pi / ( M * dx ) * (n + 0.5)
    elif T1 == DFT1:
        n = np.arange(-N/2, N/2)
        k = 2 * np.pi / ( N * dx ) * n
        k = np.fft.ifftshift(k)

    return k

def laplacianDtt1D(p, dx, T1, debug = False):
    k = computeK(p, dx, T1)

    if   T1 <= DCT4:
        p_hat = scipy.fft.dct(p, type = T1)
    elif T1 <= DST4:
        p_hat = scipy.fft.dst(p, type = T1 - 4)
    else:
        if len(p) % 2 != 0:
            raise ValueError("Fourier transform does not work well for an uneven number of grid points")
        p_hat = scipy.fft.fft(p)

    if debug:
        plt.title("p_hat")
        plt.plot(p_hat.real, label="real")
        plt.plot(p_hat.imag, label="imag")
        plt.legend()
        plt.show()

        for i in range(1, 11):
            plt.title(f"{i}-th derivative of p")

            if   T1 <= DCT4:
                dip = scipy.fft.idct(p_hat * k**i, type = T1)
            elif T1 <= DST4:
                dip = scipy.fft.idst(p_hat * k**i, type = T1 - 4)
            else:
                dip = scipy.fft.ifft(p_hat * k**i)

            plt.plot(dip.real, label="real")
            plt.plot(dip.imag, label="imag")
            plt.legend()
            plt.show()



    p_hat = p_hat * (-1) * k**2

    if   T1 <= DCT4:
        pn = scipy.fft.idct(p_hat, type = T1)
    elif T1 <= DST4:
        pn = scipy.fft.idst(p_hat, type = T1 - 4)
    else:
        pn = scipy.fft.ifft(p_hat)

    return pn, k

def getSingleDerivative(f, j, dx, stencil, derivative_order=1, debug=False):
    shifts, coeff = stencil
    f_dx = 0
    for i, shift in enumerate(shifts):
        f_dx += f[j + shift] * coeff[i]
    return f_dx / dx ** derivative_order


def getDerivative(f, dx, stencil, derivative_order=1, axis=0, debug=False):
    # directions for np.roll()
    f_dx = np.zeros(f.shape, dtype=f.dtype)
    shifts, coeff = stencil
    for i, shift in enumerate(shifts):
        if debug:
            print(
                "Derivative order",
                derivative_order,
                "Order = ",
                len(stencil) - 1,
                "shift ",
                shift,
                " coefficient = ",
                coeff[i],
            )
        f_dx += np.roll(f, shift * -1, axis=axis) * coeff[i]
    return f_dx / dx ** derivative_order

# Return array with finite difference coefficients for approximations of order (stencil_length - derivative_order)
def getFiniteDifferenceCoefficients(derivative_order, accuracy, mode, stencil=None, debug=False):
    stencil_length = derivative_order + accuracy

    if mode == MODE_FORWARD:
        stencil = np.array(range(0, stencil_length), dtype=int)
    elif mode == MODE_BACKWARD:
        stencil = np.array(range(-stencil_length + 1, 1), dtype=int)
    elif mode == MODE_CENTERED:
        if accuracy % 2 != 0:
            raise ValueError(
                "Centered stencils only available with even accuracy orders"
            )
        if (stencil_length % 2 == 0) and stencil_length >= 4:
            stencil_length -= 1
        half_stencil_length = int((stencil_length - 1) / 2)
        stencil = np.array(
            range(-half_stencil_length, half_stencil_length + 1), dtype=int
        )
    elif mode == MODE_CUSTOM:
        if stencil is None:
            raise ValueError("Need to provide custom stencil in MODE_CUSTOM")
        stencil_length = len(stencil)
        if derivative_order >= stencil_length:
            raise ValueError("Derivative order must be smaller than stencil length")

    A = np.zeros((stencil_length, stencil_length))
    b = np.zeros(stencil_length)

    for i in range(stencil_length):
        A[i, :] = stencil ** i

    b[derivative_order] = np.math.factorial(derivative_order)

    if debug:
        print("A", A)
        print("b", b)

    coefficients = np.linalg.solve(A, b)
    return stencil, coefficients


MAX_SMOOTHING_ORDER = 30
fstencils = []
bstencils = []
cstencils = []

for i in range(0, MAX_SMOOTHING_ORDER):
    N_MAX = i + 2
    fstencils_at_order_i = []
    bstencils_at_order_i = []
    cstencils_at_order_i = []
    for order in range(1, N_MAX):
        c = getFiniteDifferenceCoefficients(order, N_MAX - order + ((N_MAX - order) % 2 != 0), mode=MODE_CENTERED)
        f = getFiniteDifferenceCoefficients(order, N_MAX - order, mode= MODE_FORWARD)
        b = getFiniteDifferenceCoefficients(order, N_MAX - order, mode= MODE_BACKWARD)
        fstencils_at_order_i.append(f)
        bstencils_at_order_i.append(b)
        cstencils_at_order_i.append(c)
    fstencils.append(fstencils_at_order_i)
    bstencils.append(bstencils_at_order_i)
    cstencils.append(cstencils_at_order_i)

stencils      = [fstencils, cstencils, bstencils]
stencil_names = ["forward", "centered", "backward"]

class PlaneWave:
    def __init__(self, amplitudes, momenta):
        self.amplitudes = amplitudes
        self.momenta = momenta
        self.omega   = self.momenta**2 / 2
        self.N       = len(amplitudes)

    def __call__(self, xx, derivative_order=0, t=0):
        waves = []
        sum   = 0
        for i in range(self.N):
            wave = self.amplitudes[i] * (1j * self.momenta[i])**derivative_order * np.exp(1j * (self.momenta[i] * xx - self.omega[i] * t))
            waves.append(wave)
            sum += wave
        return sum

def make_plane_wave_superposition(points, f, k0, bc_type = None):
    if bc_type is None:
        bc_type = ([], [])

    N          = len(bc_type[0]) + 1

    #All plane waves with a maximum momentum of k0/2
    scaling = 1/2**np.arange(N)
    positive_momenta = k0 * scaling
    momenta    = np.array([-positive_momenta, positive_momenta]).flatten()

    def operator(amplitudes):
        A          = np.zeros((2 * N), complex)
        for j in range(N):
            A[2 * j    ] = np.sum(amplitudes * (1j * momenta)**j * np.exp(1j * momenta * points[0]))
            A[2 * j + 1] = np.sum(amplitudes * (1j * momenta)**j * np.exp(1j * momenta * points[1]))
        return A

    b    = np.zeros(2 * N, dtype=complex)
    b[0] = f[0]
    b[1] = f[1]
    for j in range(1, N):
        b[2 * j    ] = bc_type[0][j-1][1]
        b[2 * j + 1] = bc_type[1][j-1][1]


    A       = LinearOperator((2 * N, 2 * N), matvec =  operator)
    amplitudes, exitCode = gmres(A,  b, tol=1e-15)
    return PlaneWave(amplitudes, momenta)


class IPRReconstruction:
    def __init__(self, x, lam = 1, cutoff = 100):
        self.cutoff                 = cutoff
        self.lam                    = lam
        W                           = self.directW(self.shiftx(x), int(len(x)/2), lam)
        self.p, self.l, self.u      = scipy.linalg.lu(W, permute_l=False)

    def poly(self, N):
        return scipy.special.chebyt(N)

    def reconstruct(self, g):
        f = np.poly1d([])
        for l, coeff in enumerate(g):
            f += coeff * self.poly(l)

        return f

    def __call__(self, x, order = 0):
        return self.rec.deriv(order)((x - self.s)*self.c) * self.c**order


    #Construct matrices T and V recursively for arbitrary lambda
    def directW(self, x, N, lam):
        # Even
        W = np.zeros((2*N, 2*N), dtype=complex)

        for l in range(2*N):
            W[:, l] = scipy.fft.fft(self.poly(l)(x))

        return W


    def shiftx(self, x):
        dx = x[1] - x[0]
        self.a = x[0]
        self.b = x[-1]
        self.s = (self.a + self.b)/2
        self.c = 1 / ((self.b - self.a)/2)
        sx = (x - self.s) * self.c
        return  sx

    def gaussWithTruncation(self, A, B):
        """
        Solve Ax = B using Gaussian elimination and LU decomposition with truncation for stability of IPR
        """
        # LU decomposition with pivot
        p, l, u = scipy.linalg.lu(A, permute_l=False)
        return self.solveLUWithTruncation(B, p, l, u)


    def compute(self, psi, p = None, l = None, u = None):
        B = scipy.fft.fft(psi)

        if p is None:
            p = self.p
        if l is None:
            l = self.l
        if u is None:
            u = self.u

        # forward substitution to solve for Ly = B
        y = np.zeros(B.size, dtype=complex)
        for m, b in enumerate((p.T @ B).flatten()):
            y[m] = b
            # skip for loop if m == 0
            if m:
                for n in range(m):
                    y[m] -= y[n] * l[m,n]
            y[m] /= l[m, m]

        # truncation for IPR
        c = np.abs(y) < self.cutoff * np.finfo(float).eps
        y[c] = 0

        # backward substitution to solve for y = Ux
        x = np.zeros(B.size, dtype=complex)
        lastidx = B.size - 1  # last index
        for midx in range(B.size):
            m = B.size - 1 - midx  # backwards index
            x[m] = y[m]
            if midx:
                for nidx in range(midx):
                    n = B.size - 1  - nidx
                    x[m] -= x[n] * u[m,n]
            x[m] /= u[m, m]
        self.rec = self.reconstruct(x)

def fCosineDiffMat(order, dx):
    s = order
    mat = np.zeros((s, s), dtype=complex)
    for k in range(1, s+1):
        for j in range(1, s+1):
            mat[j-1, k-1] = (j * dx)**k / np.math.factorial(k)

    return mat

def bCosineDiffMat(order, dx):
    s = order
    mat = np.zeros((s, s), dtype=complex)
    for k in range(1, s+1):
        for j in range(1, s+1):
            mat[j-1, k-1] = (-j * dx)**k / np.math.factorial(k)

    return mat

def fCosineDiffVec(order, f):
    diff = np.zeros(order, dtype=complex)
    for j in range(1, order + 1):
        diff[j-1] = f[j] - f[0]
    return diff

def bCosineDiffVec(order, f):
    diff = np.zeros(order, dtype=complex)
    for j in range(1, order + 1):
        diff[j-1] = (f[-1-j] - f[-1])
    return diff


def iterativeRefinement(A, b, tolerance = 1e-9):
    C = np.linalg.solve(A, b)
    residual      = b - A @ C
    residualError = np.sum(np.abs(residual))

    iteration = 0
    while residualError > tolerance:
        correction = np.linalg.solve(A, residual)
        C += correction
        residual = b - A @ C
        residualError = np.sum(np.abs(residual))
        iteration += 1
        #print(f"After {iteration} iterations with residual error {residualError}")
        if iteration > 100:
            break

    #print(f"Finished in {iteration} iterations with residual error {residualError}")
    return C

def shiftx(x):
    return (x - x[0])/(x[-1] - x[0])

def cosineDiffVec(order, f, Dl, Dr):
    b = np.zeros(2*order, dtype=complex)
    b[0] = f[ 0]
    b[1] = f[-1]
    for i in range(1, order):
        b[i*2    ] = Dl[2*i-1]/(np.pi)**(2*i)

    for i in range(1, order):
        b[i*2 + 1] = Dr[2*i-1]/(np.pi)**(2*i)

    return b

def cosineDiffMat(order):
    A = np.zeros((order*2, order*2), dtype=complex)
    for i in range(order):
        derivative  = 2 * i
        for j in range(1, 2*order+1):
            #Every derivative gives a factor of j -- j**derivative
            #Every second derivative gives a minus sign -- (-1)**i
            #Cosine evaluated at pi gives negative sign depending on wavelength -- (-1)**j
            A[2*i  , j-1] = j**derivative * (-1)**i
            A[2*i+1, j-1] = j**derivative * (-1)**i * (-1)**j

    return A


def reconstructCosine(C, x, derivative_order = 0, t = 0):
    f = np.zeros(x.shape, dtype=complex)
    L = x[-1] - x[0]
    xeval = shiftx(x)
    for k in range(1, len(C) + 1):
        f += C[k-1] * (1j * k * np.pi / L) ** derivative_order * (np.exp(1j * (k * np.pi * xeval - (k*np.pi / L)**2 / 2 * t)) + np.exp(1j * (-k * np.pi * xeval - (k*np.pi / L)**2 / 2 * t)))/2

    return f

class CosineWave:
    def __init__(self, C):
        self.C = C

    def __call__(self, xx, derivative_order=0, t = 0):
        return reconstructCosine(self.C, xx, derivative_order, t = t)

def getCosineShiftFunction(f, order, x):
    xeval = shiftx(x)
    dx = xeval[1] - xeval[0]

    if 0:
        A = fCosineDiffMat (order, dx)
        b = fCosineDiffVec (order, f)
        Dl = iterativeRefinement(A, b)

        A = bCosineDiffMat (order, dx)
        b = bCosineDiffVec (order,  f)

        Dr = iterativeRefinement(A, b)
    else:
        Nrec = 16
        lam = 1
        lipr = IPRReconstruction(x[:Nrec])
        lipr.compute(f[:Nrec])
        ripr = IPRReconstruction(x[-Nrec:])
        ripr.compute(f[-Nrec:])
        Dl = []
        Dr = []
        for i in range(1, order + 1):
            Dl.append(lipr(x[0], i))
            Dr.append(ripr(x[-1], i))



    A = cosineDiffMat(int(order/2) + 1)
    b = cosineDiffVec(int(order/2) + 1, f, Dl, Dr)
    C = iterativeRefinement(A, b)


    shift = reconstructCosine(C, xeval)
    return shift, C


def getShiftFunction(x, f, mode, derivative_mode, lb, rb, chop = True, N = 0, debug = False, fd_f_stencil = None, fd_c_stencil = None, fd_b_stencil = None):

    dx       = x [ 1 ] - x[ 0 ]
    x0       = x [ 0 + lb ]
    x1       = x [-1 - rb ]
    f0       = f [ 0 + lb ]
    f1       = f [-1 - rb ]
    L        = x1 - x0


    lind = 0
    rind = len(x)

    if chop:
        rind = len(x) - rb
        lind = lb

    N_columns = 1

    if mode == M_LINEAR:
        N_columns = 1
    else:
        N_columns = 1 + N
        if fd_f_stencil is None:
            fd_f_stencil = fstencils[N - 1]
        if fd_b_stencil is None:
            fd_b_stencil = bstencils[N - 1]
        if fd_c_stencil is None:
            fd_c_stencil = cstencils[N - 1]

    #N_columns = 11

    B = np.zeros((N_columns, len(f)), f.dtype)

    if mode == M_LINEAR:
        #Compute linear shift function
        slope = (f1 - f0) / (x1 - x0)
        B[0]  = f0 + slope * ( x - x0 )
        poly = 0
    elif mode == M_POLYNOMIAL:
        bc_l = []
        bc_r = []
        for i in range(N):
            if derivative_mode == CENTRAL:
                bc_l.append((i + 1,   getSingleDerivative( f,  lb,     dx, fd_c_stencil[i], i + 1)))
                bc_r.append((i + 1,   getSingleDerivative( f, -rb - 1, dx, fd_c_stencil[i], i + 1)))
            elif derivative_mode == ONE_SIDED:
                bc_l.append((i + 1,   getSingleDerivative( f,  lb    , dx, fd_f_stencil[i], i + 1)))
                bc_r.append((i + 1,   getSingleDerivative( f, -rb - 1, dx, fd_b_stencil[i], i + 1)))
            elif derivative_mode == PERIODIC:
                bc_l.append((i + 1, - getSingleDerivative( f, -rb - 1, dx, fd_b_stencil[i], i + 1) + getSingleDerivative( f,  lb    , dx, fd_f_stencil[i], i + 1)))
                bc_r.append((i + 1, 0))
                f0 = +f0 - f1
                f1 = 0
            else:
                raise ValueError(f"Unsupported derivative_mode {derivative_mode} in getShiftFunction!")
        bc_type=(bc_l, bc_r)

        if N == 0:
            bc_type = None


        if 0:
            for i in range(N):
                print(f"{i+1}th derivative of f at left boundary:  {getSingleDerivative( f,  lb    , dx, fd_f_stencil[i], i + 1):3.3e}")
                print(f"{i+1}th derivative of f at right boundary: {getSingleDerivative( f, -rb - 1, dx, fd_b_stencil[i], i + 1):3.3e}")

        poly  = scipy.interpolate.make_interp_spline([x0, x1], [f0, f1], k = 2 * N + 1, bc_type=bc_type, axis=0)
        #print("Poly reality: ")
        #print(poly(x0), f0)
        #print(poly(x1), f1)
        for i in range(N + 1):
            B[i] = poly( x, i * 2)

        if 0:
            for i in range(N):
                print(f"{i+1}th derivative of hom at left boundary:  {getSingleDerivative( f-B[0],  lb    , dx, fd_f_stencil[i], i + 1):3.3e}")
                print(f"{i+1}th derivative of hom at right boundary: {getSingleDerivative( f-B[0], -rb - 1, dx, fd_b_stencil[i], i + 1):3.3e}")

    elif mode == M_PLANE_WAVE:
        bc_l = []
        bc_r = []
        for i in range(N):
            if derivative_mode == CENTRAL:
                bc_l.append((i + 1,   getSingleDerivative( f,  lb,     dx, fd_c_stencil[i], i + 1)))
                bc_r.append((i + 1,   getSingleDerivative( f, -rb - 1, dx, fd_c_stencil[i], i + 1)))
            elif derivative_mode == ONE_SIDED:
                bc_l.append((i + 1,   getSingleDerivative( f,  lb    , dx, fd_f_stencil[i], i + 1)))
                bc_r.append((i + 1,   getSingleDerivative( f, -rb - 1, dx, fd_b_stencil[i], i + 1)))
            elif derivative_mode == PERIODIC:
                bc_l.append((i + 1, - getSingleDerivative( f, -rb - 1, dx, fd_b_stencil[i], i + 1) + getSingleDerivative( f,  lb    , dx, fd_f_stencil[i], i + 1)))
                bc_r.append((i + 1, 0))
                f0 = +f0 - f1
                f1 = 0
            else:
                raise ValueError(f"Unsupported derivative_mode {derivative_mode} in getShiftFunction!")
        bc_type=(bc_l, bc_r)

        if N == 0:
            bc_type = None

        poly  = make_plane_wave_superposition([x0, x1], [f0, f1], k0 = 2*np.pi/L, bc_type=bc_type)
        #print("Poly reality: ")
        #print(poly(x0), f0)
        #print(poly(x1), f1)
        for i in range(N + 1):
            B[i] = poly( x, i * 2)
    elif mode == M_COSINE:

        if 0:
            for i in range(N):
                print(f"{i+1}th derivative of f at left boundary:  {getSingleDerivative( f,  lb    , dx, fd_f_stencil[i], i + 1):3.3e}")
                print(f"{i+1}th derivative of f at right boundary: {getSingleDerivative( f, -rb - 1, dx, fd_b_stencil[i], i + 1):3.3e}")

        shift, C = getCosineShiftFunction(f, N, x)
        poly = CosineWave(C)
        for i in range(N + 1):
            B[i] = poly( x, i * 2)


        if 0:
            for i in range(N):
                print(f"{i+1}th derivative of hom at left boundary:  {getSingleDerivative( f-shift,  lb    , dx, fd_f_stencil[i], i + 1):3.3e}")
                print(f"{i+1}th derivative of hom at right boundary: {getSingleDerivative( f-shift, -rb - 1, dx, fd_b_stencil[i], i + 1):3.3e}")

    elif mode == M_EVEN:
        bc_l = []
        bc_r = []
        for i in range(N):
            if derivative_mode == ONE_SIDED:
                bc_l.append(((i + 1) * 2,   getSingleDerivative( f,  lb    , dx, fd_f_stencil[i], (i + 1) * 2)))
                bc_r.append(((i + 1) * 2,   getSingleDerivative( f, -rb - 1, dx, fd_b_stencil[i], (i + 1) * 2)))
            else:
                raise ValueError(f"Unsupported derivative_mode {derivative_mode} in getShiftFunction!")
        bc_type=(bc_l, bc_r)
        print(bc_type)

        if N == 0:
            bc_type = None

        poly  = scipy.interpolate.make_interp_spline([x0, x1], [f0, f1], k = 2 * N + 1, bc_type=bc_type, axis=0)
        #print("Poly reality: ")
        #print(poly(x0), f0)
        #print(poly(x1), f1)
        for i in range(N + 1):
            B[i] = poly( x, i * 2)

    elif mode == M_DISABLED:
        poly = 0
        pass
    else:
        raise ValueError(f"Unsupported mode {mode} in getShiftFunction!")

    if debug:
        plt.title("f")
        plt.plot(f.real, label = "real" )
        plt.plot(f.imag, label = "imag" )
        plt.legend()
        plt.show()
        plt.title("B")
        plt.plot(B[0].real, label = "real" )
        plt.plot(B[0].imag, label = "imag" )
        plt.legend()
        plt.show()
        plt.title("f - B")
        plt.plot((f - B[0]).real, label = "real" )
        plt.plot((f - B[0]).imag, label = "imag" )
        plt.legend()
        plt.show()

        f0 = f[lind]
        f1 = f[rind - 1]
        B0 = B[0][lind]
        B1 = B[0][rind - 1]
        h0 = f0 - B0
        h1 = f1 - B1

        print(f"f[lb] = {f0} f[rb] = {f1} B[0][lb] = {B0}  B[0][rb] = {B1} homf[lb] = {h0} homf[rb] = {h1}")


    return B [:,  lind : rind ], poly


class FourierExtension:
    def __init__(self, N, Ncoll, theta, chi, cutoff):
        self.N, self.Ncoll, self.theta, self.chi, self.cutoff = N, Ncoll, theta, chi, cutoff
        self.Meven = self.getFPICSUEvenMatrix(N, Ncoll, theta, chi)
        self.Modd  = self.getFPICSUOddMatrix (N, Ncoll, theta, chi)

        self.dx    = chi / (Ncoll - 1)
        self.x     = np.arange(-Ncoll + 1, Ncoll) * self.dx

        self.Meveninv  = self.invertRealM(self.Meven, cutoff)
        self.Moddinv   = self.invertRealM(self.Modd,  cutoff)

    def getX(self):
        return self.x, self.dx

    def getFPICSUEvenMatrix(self, N, Ncoll, theta, chi):
        M  = np.zeros((Ncoll, N))
        dx = chi / (Ncoll - 1)
        for i in range(Ncoll):
            for j in range(N):
                #Collocation points uniformly distributed over the positive half
                #of the physical interval x in [0, chi]
                M[i, j] = np.cos(j * np.pi / theta * i * dx)
        return M

    def getFPICSUOddMatrix(self, N, Ncoll, theta, chi):
        M = np.zeros((Ncoll, N))
        dx = chi / (Ncoll - 1)
        for i in range(Ncoll):
            for j in range(N):
                #Collocation points uniformly distributed over the positive half
                #of the physical interval x in [0, chi]
                M[i, j] = np.sin(j * np.pi / theta * i * dx)
        return M

    def invertRealM(self, M, cutoff):
        U, s, Vh = scipy.linalg.svd(M)
        sinv = np.zeros(M.T.shape)
        for i in range(np.min(M.shape)):
            if s[i] < cutoff:
                sinv[i, i] = 0
            else:
                sinv[i, i] = 1/s[i]
        return Vh.T @ sinv @ U.T


    def rescaleToPhysical(self, x, getSlope = False):
        a = x[0]
        b = x[-1]
        L = b -a
        sx = ( x - a ) / L
        sx = sx * (2*self.chi) - self.chi
        if getSlope:

            return sx, a * 2 * self.chi / L + self.chi, 2 * self.chi / L
        else:
            return sx

    def rescaleToExtended(self, x):
        a = x[0]
        b = x[-1]
        sx = ( x - a ) / ( b - a )
        sx = sx * (2*self.theta) - self.theta
        return sx


    def convertToFourierCoeff(self, aodd, aeven):
        k = np.arange(-self.N, self.N) * np.pi / self.theta
        fhat = np.zeros(2*self.N, dtype=complex)

        for j, (oddcoeff, evecoeff) in enumerate(zip(aodd, aeven)):
            fhat[ j + self.N] +=   oddcoeff / (2j) + evecoeff / (2)
            fhat[-j + self.N] += - oddcoeff / (2j) + evecoeff / (2)


        return np.fft.ifftshift(fhat), np.fft.ifftshift(k)

    def reconstructFourier(self, x, fhat):

        rec  = np.zeros (   x.shape, dtype=complex)
        ks   = np.fft.ifftshift(np.arange( -self.N, self.N) * np.pi / self.theta)

        for k, coeff in zip(ks, fhat):
            rec += coeff * np.exp(1j * (k * x))
        return rec

    def iterativeRefinement(self, M, Minv, f, threshold = 100, maxiter = 5):
        a       = Minv @ f
        r       = M @ a - f
        counter = 0
        while np.linalg.norm(r) > 100 * np.finfo(float).eps * np.linalg.norm(a) and counter < maxiter:
            delta    = Minv @ r
            a        = a - delta
            r        = M @ a - f
            counter += 1
        return a

    def computeExtension(self, f, Ni, threshold = 10, maxiter = 3):
        refeven  = ((f + np.flip(f)).real/2)[self.Ncoll-1:]
        refodd   = ((f - np.flip(f)).real/2)[self.Ncoll-1:]
        imfeven  = ((f + np.flip(f)).imag/2)[self.Ncoll-1:]
        imfodd   = ((f - np.flip(f)).imag/2)[self.Ncoll-1:]
        aeven    = self.iterativeRefinement(self.Meven, self.Meveninv, refeven, threshold = threshold, maxiter = maxiter) + 1j * self.iterativeRefinement(self.Meven, self.Meveninv, imfeven, threshold = threshold, maxiter = maxiter)
        aodd     = self.iterativeRefinement(self.Modd,  self.Moddinv,  refodd,  threshold = threshold, maxiter = maxiter) + 1j * self.iterativeRefinement(self.Modd,  self.Moddinv,  imfodd,  threshold = threshold, maxiter = maxiter)
        fhat, k  = self.convertToFourierCoeff(aodd, aeven)
        frec     = self.fourierInterpolation(fhat, Ni)
        return frec, fhat

    def evolve(self, f, dt, threshold = 10, maxiter = 3):
        refeven  = ((f + np.flip(f)).real/2)[self.Ncoll-1:]
        refodd   = ((f - np.flip(f)).real/2)[self.Ncoll-1:]
        imfeven  = ((f + np.flip(f)).imag/2)[self.Ncoll-1:]
        imfodd   = ((f - np.flip(f)).imag/2)[self.Ncoll-1:]
        aeven    = self.iterativeRefinement(self.Meven, self.Meveninv, refeven, threshold = threshold, maxiter = maxiter) + 1j * self.iterativeRefinement(self.Meven, self.Meveninv, imfeven, threshold = threshold, maxiter = maxiter)
        aodd     = self.iterativeRefinement(self.Modd,  self.Moddinv,  refodd,  threshold = threshold, maxiter = maxiter) + 1j * self.iterativeRefinement(self.Modd,  self.Moddinv,  imfodd,  threshold = threshold, maxiter = maxiter)
        fhat, k  = self.convertToFourierCoeff(aodd, aeven)
        frec     = self.reconstructFourier(self.x, fhat, dt)
        return frec

    def plotApproximationErorr(self, xext, forg, frec):
        plt.title("Approximation error of f")
        plt.yscale("log")
        plt.plot(xext, np.abs(forg - frec))
        plt.show()


    def fourierInterpolation(self, fhat, Ni):
        N = len(fhat)
        Npad = int(Ni/2 - N/2)
        ft   = np.fft.fftshift(fhat)
        ft_pad = np.concatenate([np.zeros(Npad), ft, np.zeros(Npad)])
        fint = scipy.fft.ifft(np.fft.fftshift(ft_pad), norm="forward")
        return fint
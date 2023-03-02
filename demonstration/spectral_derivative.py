import numpy as np
import scipy
import matplotlib.pyplot as plt

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

M_LINEAR    = 1
M_NTH_ORDER = 2
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

def getShiftFunction(x, f, mode, derivative_mode, lb, rb, chop = True, N = 0, debug = False, fd_f_stencil = None, fd_c_stencil = None, fd_b_stencil = None):

    dx       = x [ 1 ] - x[ 0 ]
    x0       = x [ 0 + lb ]
    x1       = x [-1 - rb ]
    f0       = f [ 0 + lb ]
    f1       = f [-1 - rb ]


    lind = 0
    rind = len(x)

    if chop:
        rind = len(x) - rb
        lind = lb

    N_columns = 1

    if mode == M_LINEAR:
        N_columns = 1
    elif mode == M_NTH_ORDER:
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
    elif mode == M_NTH_ORDER:
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

        poly  = scipy.interpolate.make_interp_spline([x0, x1], [f0, f1], k = 2 * N + 1, bc_type=bc_type, axis=0)
        #print("Poly reality: ")
        #print(poly(x0), f0)
        #print(poly(x1), f1)
        for i in range(N + 1):
            B[i] = poly( x, i * 2)

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


    return B [:,  lind : rind ]
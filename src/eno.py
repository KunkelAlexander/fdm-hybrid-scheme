import numpy as np 
import matplotlib.pyplot as plt

# Get forward divided difference of degree m for direction = 1, backward for direction = -1
def dividedDifference(f, dx, i, m, direction=1):
    N = len(f)
    if m == 0:
        return f[i]
    else:
        return (
            direction
            * (
                dividedDifference(f, dx, (i + direction) % N, m - 1, direction)
                - dividedDifference(f, dx, i, m - 1, direction)
            )
            / (m * dx)
        )

#Essentially nonoscillatory scheme
class ENO:
    #Optionally pass suitable numpy arrays for lpis and coef to avoid memory allocation
    def __init__(self, xx, y, L, order, direction, lpis = None, coef = None):
        dx = xx[ 1] - xx[0]
        #NOTE: L  != xx[-1] - xx[0] because of periodic boundary conditions, it is actually (N + 1)*dx
        N  = len(y)

        self.dx, self.xx, self.y, self.L, self.N, self.order, self.direction = dx, xx, y, L, N, order, direction

        coef    = np.zeros((order, N), dtype=float)
        lpis    = np.zeros((order, N), dtype=int)

        # Construct two-point stencial polynomial P^(f, 1)_(j + 1/2)(x) = self.psi +  b (x - xj)
        lpis[0] = np.array(range(N), dtype=int)
        coef[0] = dividedDifference(y, dx, lpis[0], 1, direction)
        

        for m in range(1, order):
            # direction = 1 : Add right neighbouring point to current stencil and compute divided difference
            # direction = -1: Add left  neighbouring point to current stencil and compute divided difference

            #a = self.dividedDifference(y, dx,  lpis[m - 1]                 , m + 1, direction)
            #b = self.dividedDifference(y, dx, (lpis[m - 1] - direction) % N, m + 1, direction)
            
            
            a = dividedDifference(y, dx,  lpis[m-1], m + 1, direction)
            b = dividedDifference(y, dx, (lpis[m-1] - direction) % N, m + 1, direction)

            # direction =  1: Whereever np.abs(a) < b == Whereever adding the right neighbour leads to a smaller divided difference, Add right neighbour to current stencil
            # direction = -1: Whereever np.abs(a) < b == Whereever adding the left  neighbour leads to a smaller divided difference, Add left  neighbour to current stencil

            coef[m, :] = a
            lpis[m, :] = lpis[m - 1, :]

            # direction =  1: Else, add left  neighbour to current stencil
            # direction = -1: Else, add right neighbour to current stencil

            add_other = (np.abs(a) >= np.abs(b))
            coef[m, add_other] = b[add_other]
            lpis[m, add_other] = (lpis[m, add_other] - direction) % N

        
        self.lpis, self.coef = lpis, coef


    def P(self, x):
        dx, xx, y, L, N, order, direction, lpis, coef = self.dx, self.xx, self.y, self.L, self.N, self.order, self.direction, self.lpis, self.coef
    
        epsilon = 1e-8

        #Compute interval indices for points in x-array
        i        = np.floor(x/dx + epsilon).astype(int) % N
        #Compute distances to nearest support
        d        = x - xx[lpis[0, i]]
        #Account for periodic boundaries
        cond     = (np.abs(d) > L/2)
        d[cond] -= np.sign(d[cond])*L

        #Approximate function by linear polynomial at lowest order
        p = y[lpis[0, i]] + coef[0, i] * d

        for m in range(1, order):
            poly    = coef[m,     i]
            ks      = lpis[m - 1, i]
            indices = np.zeros((len(ks), m + 1), dtype=int)

            if   direction ==  1:
                for j, k in enumerate(ks):
                    indices[j, :] = np.array(range(k    , k + m + 1)) % N
            elif direction == -1:
                for j, k in enumerate(ks):
                    indices[j, :] = np.array(range(k - m, k + 1    )) % N

            for j in range(m + 1):
                d        = (x - xx[indices[:, j]])
                cond     = (np.abs(d) > L/2)
                d[cond] -= np.sign(d[cond])*L
                poly *= d
            p += poly
        return p


    def dP(self, x):
        dx, xx, y, L, N, order, direction, lpis, coef = self.dx, self.xx, self.y, self.L, self.N, self.order, self.direction, self.lpis, self.coef
        epsilon = 1e-8
        i = np.floor(x/dx + epsilon).astype(int) % N
        p = coef[0, i]

        for m in range(1, order):
            poly    = coef[m,     i]
            ks      = lpis[m - 1, i]
            indices = np.zeros((len(ks), m + 1), dtype=int)

            if direction == 1:
                for j, k in enumerate(ks):
                    indices[j, :] = np.array(range(k, k + m + 1)) % N
            elif direction == -1:
                for j, k in enumerate(ks):
                    indices[j, :] = np.array(range(k - m, k + 1)) % N

            # Sum
            derivative_sum = 0

            for k in range(m + 1):
                derivative_prod = 1
                # Product
                for j in range(m + 1):
                    # Skip derivative we are summing over
                    if k == j:
                        continue
                    d                = x - xx[indices[:, j]]
                    cond             = (np.abs(d) > L/2)
                    d[cond]         -= np.sign(d[cond])*L
                    derivative_prod *= d
                derivative_sum += derivative_prod

            p += poly * derivative_sum
        return p

def plotENO( xx, xx_HR, L, f, df):
    y    = f(xx)[0:-1]
    y_HR = f(xx_HR)[0:-1]
    xx = xx[0:-1]
    xx_HR = xx_HR[0:-1]

    direction_dict = {
        '-1': 'backward',
        '1': 'forward'
    }

    Ps  = []
    dPs = []

    directions = [1, -1]
    enoOrders  = [1, 2, 3, 7]


    for direction in directions:
        plt.title(f"The approximated function in {direction_dict[str(direction)]} direction")
        plt.plot(xx, y, label="Original function")

        for enoOrder in enoOrders:
            eno = ENO(xx = xx, y = y, order = enoOrder, L = L, direction = direction)
            z = eno.P(xx_HR)
            Ps.append(z)
            plt.plot(xx_HR, z, label=f"Order {enoOrder}")

        plt.legend()
        plt.show()

    for direction in directions:
        plt.title(f"The approximated derivative in {direction_dict[str(direction)]} direction")
        plt.plot(xx, df(xx), label="Original function")

        for enoOrder in enoOrders:
            eno = ENO(xx = xx, y = y, order = enoOrder, L = L, direction = direction)
            dz = eno.dP(xx_HR)
            dPs.append(dz)
            plt.plot(xx_HR, dz, label=f"Order {enoOrder}")

        plt.legend()
        plt.show()

    
    plt.title(f"The approximated function as average")
    plt.plot(xx, y, label="Original function")

    for i, enoOrder in enumerate(enoOrders):
        plt.plot(xx_HR, (Ps[i] + Ps[i + 3])/2, label=f"Order {enoOrder}")

    plt.legend()
    plt.show()


    plt.title(f"The approximated derivative as average")
    plt.plot(xx, df(xx), label="Original function")

    for i, enoOrder in enumerate(enoOrders):
        plt.plot(xx_HR, (dPs[i] + dPs[i + 3])/2, label=f"Order {enoOrder}")

    plt.legend()
    plt.show()
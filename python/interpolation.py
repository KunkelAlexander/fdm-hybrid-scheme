import numpy as np 

def stencil_ijk(r, j, k):
    result = 0
    for m in range(j + 1, k + 1):
        s1 = 0
        for l in range(k + 1):
            if l == m:
                continue
            p = 1
            for q in range(k + 1):
                if (q == m) or (q == l):
                    continue
                p *= (r - q + 1)
            s1 += p
            

        s2 = 1
        for l in range(k + 1):
            if l == m:
                continue 
            s2 *= (m - l)

        result += s1/s2
    return result 

def stencil_ij(k):
    c = np.zeros((k + 1, k))

    for r in range(-1, k):
        for j in range(k):
            c[r + 1, j] = stencil_ijk(r, j, k)

    return c

#left_shift = r
def fixed_stencil_reconstruction(f, stencil, left_shift, axis, p2 = True, debug = False):
    R = -1   # right
    L =  1    # left
    f_rec  = np.zeros(f.shape)
    order  = stencil.shape[1]
        
    if debug:
        print("Order", order, " and left_shift ", left_shift)

    for j in range(order):
        if debug:
            print("Coefficient and roll (1 = L, -1 = R)", stencil[left_shift + int(p2), j], left_shift * L + R * j)
        f_rec += stencil[left_shift + int(p2), j] * np.roll(f, left_shift * L + R * j, axis = axis)

    return f_rec
import numpy as np
import matplotlib.pyplot as plt

def cheby_exgrid(N,a,b):
    i=np.linspace(0,N,N+1)
    return (b-a)/2.*(-np.cos(i*np.pi/N))+(b+a)/2.0

def cheby_poly(x,n):
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

def cheby_coeff(x,y,N):
    a = []
    y[0] *= 0.5
    y[N] *= 0.5

    x_chebgrid = cheby_exgrid(N,-1.0,1.0)
    print(x,x_chebgrid)
    for j in range(N+1):
        sum = 2.0 / N  * np.sum(y*cheby_poly(x_chebgrid,j))
        a.append(sum)

    y[0] /= 0.5
    y[N] /= 0.5
    return a

def cheby_interp(x,coeff,N,a,b):

    x_chebgrid = ( 2.0 * x - a  - b ) / ( b-a);
    sum  = coeff[0]*cheby_poly(x_chebgrid,0) * 0.5    
    for j in range(1,N-1):
        sum += coeff[j]*cheby_poly(x_chebgrid,j)
    sum += coeff[N]*cheby_poly(x_chebgrid,N) * 0.5
    return sum

f = np.loadtxt('test')[:,[0,1]]
N = np.shape(f)[0]-1

xx = cheby_exgrid(40,f[0,0],f[-1,0])
coeff = cheby_coeff(f[:,0],f[:,1],N)
interp =  cheby_interp(xx,coeff,N,f[0,0],f[-1,0])

print( cheby_interp(0.504897,coeff,N,f[0,0],f[-1,0]))
plt.plot(f[:,0],f[:,1],'>',xx,interp,'.')
plt.show()

# test it with sin(x)
# a = 1.
# b = 2.
# x = cheby_exgrid(N,a,b)
# f = np.sin(5.*x)
# xx = np.linspace(a,b,11)

# coeff = cheby_coeff(x,f,N)
# interp =  cheby_interp(xx,coeff,N,a,b)
# plt.plot(x,f,'-',xx,interp,'.')
# plt.show()


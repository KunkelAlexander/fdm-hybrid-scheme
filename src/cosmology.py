def getScaleFactor(time):
    return (3 / 2 * time) ** (2 / 3)

def getTime(a):
    return a**(3/2) * 2/3

#dt' = dt a(t)^2
def getDashedTime(t):
    return 9/14 * (3/2)**(1/3) * t**(7/3)
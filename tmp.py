# import scipy.optimize
# import sympy

# print(scipy.optimize.fsolve(lambda x: x**2 + 2*x + 1, 0))
# print(sympy.solve('x**2 + 2*x + 1'))
import numpy as np 

def quadratic(a, b, c):
    p=np.poly1d([a,b,c])
    print (p.r)

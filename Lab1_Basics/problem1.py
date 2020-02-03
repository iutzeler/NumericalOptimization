#!/usr/bin/env python
# coding: utf-8

# # Problem 1
# 
# 
# The objective of Problem 1 is to minimize a simple quadratic function $f$ on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# f: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  & 4 (x_1-3)^2 + 2(x_2-1)^2
# \end{array}$$



##### Function definition
def f(x):
    x1 = x[0]
    x2 = x[1]
    return 4*(x1-3)**2+2*(x2-1)**2
####

##### Plot parameters f
x1_min = -0.5
x1_max = 5.5
x2_min = -0.5
x2_max = 5.5
nb_points = 200
vmin = 0
vmax = 80
levels = [0.5,1,2,5,10,15]
title = 'f: a simple function'
####


# ### Some parameters
# 
# Before solving things numerically, some useful things can be computed:
# * Properties of $f$: lower bounds, Lipschitz constant of $\nabla f$, strong convexity constant, etc
# * Good starting points (for hot starting e.g.)



###### Useful Parameters
L = 8        # Lipschitz constant of the gradient


# ### Oracles
# 
# Numerical optimization methods need callable *oracles* for properties of $f$, that is a function that, given a point $x$ in the domain of $f$, returns $f$ and/or gradient, Hessian of $f$ at point $x$. We talk about the *order* of an oracle as the number of differentiations given (0th order for just $f$, 1st order for the gradient, 2nd for gradient + Hessian).



# Q: Observe the first order oracle `f_grad`.



import numpy as np

##### Gradient oracle
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-3) 
    gy = 4*(x2-1)
    return np.array( [ gx  ,  gy  ] )
####


# Q: Observe the second order oracle `f_grad_hessian`.


import numpy as np

##### Hessian scaled Gradient computation
def f_grad_hessian(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-3) 
    gy = 4*(x2-1) 
    g = np.array( [ gx  ,  gy  ] )
    H = np.array(  [ (8.0 , 0.0 )  ,  ( 0.0 , 4.0 )  ]  )  
    return g,H
####


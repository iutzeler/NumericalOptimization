#!/usr/bin/env python
# coding: utf-8

# # Problem 2
# 
# 
# The objective of Problem 2 is to minimize a more involved but very smooth function function $g$ on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# g: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  & \log( 1 + \exp(4 (x_1-3)^2 ) + \exp( 2(x_2-1)^2 ) ) - \log(3)
# \end{array}$$




##### Function definition
def f(x):
    x1 = x[0]
    x2 = x[1]
    return np.log( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) ) - np.log(3)
####

##### Plot parameters f
x1_min = -0.5
x1_max = 5.5
x2_min = -0.5
x2_max = 5.5
nb_points = 500
vmin = 0
vmax = 100
levels = [0.5,1,2,5,10,15]
title = 'a Harder function: g'
####



###### Useful Parameters
L = 8        # Lipschitz constant of the gradient


# ### Oracles


# Q: Complete the first order oracle `f_grad`.



import numpy as np

##### Gradient oracle
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    gx = 0  ## To complete
    gy = 0  ## To complete
    return np.array( [ gx  ,  gy  ] )
####


# Q: Fill the following second order oracle `f_grad_hessian`.


import numpy as np

##### Hessian scaled Gradient computation
def f_grad_hessian(x):

    return g,H
####






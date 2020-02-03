#!/usr/bin/env python
# coding: utf-8

# # Problem 21
# 
# 
# The objective of Problem 21 is to minimize a simple quadratic function $f$ on $\mathbb{R}^2$, constrained to $x\ge 0$: 
# 
# $$\begin{array}{rrcll}
# f: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  & 4 (x_1-1)^2 + 2(x_2+0.5)^2
# \end{array}$$
# <center><img src="Fig/1.png" width="50%"></center>

# ### Function definition 

# In[1]:


##### Function definition
def f(x):
    x1 = x[0]
    x2 = x[1]
    return 4*(x1-1)**2+2*(x2+0.5)**2
####

##### Plot parameters f
x1_min = -4.
x1_max = 3.
x2_min = -4.
x2_max = 3.
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

# In[2]:


###### Useful Parameters
L = 8        # Lipschitz constant of the gradient


# ### Oracles
# 
# Numerical optimization methods need callable *oracles* for properties of $f$, that is a function that, given a point $x$ in the domain of $f$, returns $f$ and/or gradient, Hessian of $f$ at point $x$. We talk about the *order* of an oracle as the number of differentiations given (0th order for just $f$, 1st order for the gradient, 2nd for gradient + Hessian).
# 
# > Observe the first order oracle `f_grad`.
# 

# In[3]:


import numpy as np

##### Gradient oracle
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-1) 
    gy = 4*(x2+0.5)
    return np.array( [ gx  ,  gy  ] )
####


# > Fill the following second order oracle `f_grad_hessian`.

# In[4]:


import numpy as np

##### Hessian scaled Gradient computation
def f_grad_hessian(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-1) 
    gy = 4*(x2+0.5) 
    g = np.array( [ gx  ,  gy  ] )
    H = np.array(  [ ( 8.0 , 0 )  ,  ( 0 , 4.0 )  ]  )  ### -> To complete DONE 
    return g,H
####


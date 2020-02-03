#!/usr/bin/env python
# coding: utf-8

# # Problem 3
# 
# 
# The objective of Problem 3 is to minimize non-convex smooth Rosenbrock function $r$ on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# r: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  &  (1-x_1)^2 + 100(x_2-x_1^2)^2
# \end{array}$$
# <center><img src="Fig/3.png" width="50%"></center>

# ### Function definition 

# In[1]:


##### Function definition
def f(x):
	"""Rosenbrock."""
	x1 = x[0]
	x2 = x[1]
	return (1-x1)**2+100*(x2-x1**2)**2
####

##### Plot parameters f
x1_min = -1.5
x1_max = 1.55
x2_min = -0.2
x2_max = 1.5
nb_points = 200
vmin = 0
vmax = 120
levels = [0.05,1,5,15,50,100,200]
title = 'Rosenbrock function'
####


# ### Some parameters
# 
# Before solving things numerically, some useful things can be computed:
# * Properties of $f$: lower bounds, Lipschitz constant of $\nabla f$, strong convexity constant, etc
# * Good starting points (for hot starting e.g.)

# In[1]:


###### Useful Parameters


# ### Oracles
# 
# Numerical optimization methods need callable *oracles* for properties of $f$, that is a function that, given a point $x$ in the domain of $f$, returns $f$ and/or gradient, Hessian of $f$ at point $x$. We talk about the *order* of an oracle as the number of differentiations given (0th order for just $f$, 1st order for the gradient, 2nd for gradient + Hessian).
# 
# > Complete the first order oracle `f_grad`.
# 

# In[1]:


import numpy as np

##### Gradient oracle ##### return  grandient ### To complete
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    return np.array( ( 2*(x1-1) + 400*x1*(x1**2-x2) , 200*( x2 - x1**2)  ) )
####


# > Fill the following second order oracle `f_grad_hessian`.

# In[4]:


import numpy as np

##### Hessian scaled Gradient computation, ####	return g,H ### To complete
def f_grad_hessian(x):
    x1 = x[0]
    x2 = x[1]
    g = np.array(  [ 2*(x1-1) + 400*x1*(x1**2-x2) , 200*( x2 - x1**2)  ] )
    H = np.array(  [ ( 2 - 400*x2 + 3*400*x1**2 , -400*x1 )  ,  ( -400*x1 , 200 )  ]  )
    return g,H
####


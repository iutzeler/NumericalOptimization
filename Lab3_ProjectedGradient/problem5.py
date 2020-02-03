#!/usr/bin/env python
# coding: utf-8

# # Problem 5
# 
# 
# The objective of Problem 5 is to minimize a polyhedral function $p$ on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# p: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  &  \left| x_1-3 \right|  + 2\left| x_2-1\right| .
# \end{array}$$
# <center><img src="Fig/5.png" width="50%"></center>

# ### Function definition 

# In[1]:


##### Function definition
def f(x):
    x1 = x[0]
    x2 = x[1]
    return np.abs(x1-3)+2*np.abs(x2-1)
####

##### Plot parameters f
x1_min = -0.5
x1_max = 5.5
x2_min = -0.5
x2_max = 5.5
nb_points = 200
levels = [0.05,0.5,1,2,5]
vmin = 0
vmax = 5
title = 'polyhedral'
####


# ### Some parameters
# 
# Before solving things numerically, some useful things can be computed:
# * Properties of $f$: lower bounds, Lipschitz constant of $\nabla f$, strong convexity constant, etc
# * Good starting points (for hot starting e.g.)

# In[2]:


###### Useful Parameters


# ### Oracles
# 
# Numerical optimization methods need callable *oracles* for properties of $f$, that is a function that, given a point $x$ in the domain of $f$, returns $f$ and/or gradient, Hessian of $f$ at point $x$. We talk about the *order* of an oracle as the number of differentiations given (0th order for just $f$, 1st order for the gradient, 2nd for gradient + Hessian).
# 
# > Compute a first order oracle `f_grad`. Is it unique?
# 

# In[3]:


import numpy as np

##### Gradient oracle
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    g = np.array( [ 0.0  ,   0.0 ] )
    if x1 < 3:
        g[0] = -1.0
    elif x1 > 3:
        g[0] = 1.0
    if x2 < 1:
        g[1] = -2.0
    elif x2 > 1:
        g[1] = 2.0
    return g
###### return g ### To complete
####


# > What about a second order oracle?

# In[ ]:





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
# <center><img src="Fig/2.png" width="50%"></center>

# ### Function definition 

# In[1]:


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
# > Complete the first order oracle `f_grad`.
# 

# In[2]:


import numpy as np

##### Gradient oracle
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-3)*np.exp(4*(x1-3)**2)/( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) ) ## To complete
    gy = 4*(x2-1)*np.exp(2*(x2-1)**2)/( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) )  ## To complete
    return np.array( [ gx  ,  gy  ] )
####


# > Fill the following second order oracle `f_grad_hessian`.

# In[1]:


import numpy as np

##### Hessian scaled Gradient computation
def f_grad_hessian(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-3)*np.exp(4*(x1-3)**2)/( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) ) ## To complete
    gy = 4*(x2-1)*np.exp(2*(x2-1)**2)/( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) )  ## To complete
    
    hxx = (1+ 8*(x1-3)**2)*np.exp(4*(x1-3)**2)*( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) )
    hxx = hxx -8* ((x1-3)*np.exp(4*(x1-3)**2))**2
    hxx = 8 * hxx/( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) )**2
    hxy = -32*(x1-3)*(x2-1)*np.exp(4*(x1-3)**2)*np.exp(2*(x2-1)**2)
    hxy=hxy/( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) )**2 
    ## H is symetric thus hyx=hxy
    hyy = (1+4*(x2-1)**2)*np.exp(2*(x2-1)**2)*( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) )
    hyy= hyy -4* ((x2-1)*np.exp(2*(x2-1)**2))**2
    hyy= 4* hyy / ( 1 + np.exp(4*(x1-3)**2) + np.exp(2*(x2-1)**2) )**2 
    
    g = np.array( [ gx  ,  gy  ] )
    H = np.array(  [ ( hxx , hxy )  ,  ( hxy , hyy )  ]  )  ### -> To complete DONE 

    return g,H
####


# In[ ]:





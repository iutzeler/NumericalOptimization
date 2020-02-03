#!/usr/bin/env python
# coding: utf-8

# # Problem 4
# 
# 
# The objective of Problem 4 is to minimize a non-convex function $t$ with two minimizers on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# t: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  & (0.6 x_1 + 0.2 x_2)^2 \left((0.6 x_1 + 0.2 x_2)^2 - 4 (0.6 x_1 + 0.2 x_2)+4\right) + (-0.2 x_1 + 0.6 x_2)^2
# \end{array}$$
# <center><img src="Fig/4.png" width="50%"></center>

# ### Function definition 

# In[1]:


##### Function definition
def f(x):
    x1 = x[0]
    x2 = x[1]
    return (0.6*x1 + 0.2*x2)**2 * ((0.6*x1 + 0.2*x2)**2 - 4*(0.6*x1 + 0.2*x2)+4) + (-0.2*x1 + 0.6*x2)**2
####

##### Plot parameters f
x1_min = -1
x1_max = 4
x2_min = -1
x2_max = 4
nb_points = 200
levels = [0.05,0.5,1,2,5]
vmin = 0
vmax = 5
title = 'two pits'
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

# In[3]:


import numpy as np

##### Gradient oracle ### To complete
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    return np.array( (  0.5184*x1**3+x1**2*(-2.592+0.5184*x2)+ x2*(0.72-0.288*x2+0.0192*x2**2)+ x1*(2.96-1.728*x2+0.1728*x2**2) , 0.1728*x1**3+x1**2*(-0.864+0.1728*x2)+x2*(1.04-0.096*x2+0.0064*x2**2)+x1*(0.72-0.576*x2+0.0576*x2**2)  ) )
####


# > Does a second order oracle exist for any point?

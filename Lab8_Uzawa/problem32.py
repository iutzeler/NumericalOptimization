#!/usr/bin/env python
# coding: utf-8

# # Problem 32
# 
# 
# The objective of Problem 31 is to minimize a simple quadratic function $f$ on $\mathbb{R}^2$ under affine constraints: 
# 
# $$\begin{array}{rrcll}
# f: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  & 4 (x_1-3)^2 + 2(x_2-1)^2
# \end{array}$$
# <center><img src="Fig/1.png" width="50%"></center>
# 
# The affine constraints are given on the form $\theta_i \cdot x \le d_i, i=1,\ldots, m$ where $\theta_i$ is a vector. We have here two constraints($m=1$),  
# $\theta_1=(1,-1)$ and $d_1=1$ ; $\theta_2=(2,-1)$ and $d_2=3$
# 
# 

# ### Function definition 



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


# ### Constraint definition 

# In[3]:


import numpy as np


### Constraint definition 
d = np.array( [ 1 ] )
C = np.array(  [ ( 1.0 , -1.0 )  ]  )  
### -> To complete DONE 
alpha = 4
normeCsquared = 2
rhomax = 2*alpha/normeCsquared


# ### Constraint function definition


import numpy as np

##### Constraint functions definition phi1 and phi2
def phi1(x):
    x1 = x[0]
    x2 = x[1]
##    d = np.array( [ 1 ] )
##    C = np.array(  [ ( 1.0 , -1.0 )  ]  )  
    C11=1
    C12=-1
    d1=1
    return C11*x1 + C12*x2 - d1;
####    return np.dot(C, x) - d[0];

def phi2(x):
    x1 = x[0]
    x2 = x[1]
    C21=2
    C22=-1
    d2=3
    return C21*x1 + C22*x2 - d2;
####    return np.dot(C, x) - d[0];


##### Plot parameters phi1 and phi2 (as for f)
phi1nb_points = 100
phi1vmin = -10
phi1vmax = 10
phi1levels = [-3,-2,-1,0,1,2,3]
phi1title = 'phi1: a first simple affine constraint'
phi2title = 'phi2: a second simple affine constraint'
####

def phi(x):
    p1 = phi1(x)
    p2 = phi2(x)
    return np.array( [ p1 , p2 ] )
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



import numpy as np

##### Gradient oracle
def f_grad(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-3) 
    gy = 4*(x2-1)
    return np.array( [ gx  ,  gy  ] )
####



import numpy as np

##### Hessian scaled Gradient computation
def f_grad_hessian(x):
    x1 = x[0]
    x2 = x[1]
    gx = 8*(x1-3) 
    gy = 4*(x2-1) 
    g = np.array( [ gx  ,  gy  ] )
    H = np.array(  [ ( 8.0 , 0 )  ,  ( 0 , 4.0 )  ]  )  ### -> To complete DONE 
    return g,H
####


# ### Lagrangian oracles
# definition of the Lagrangian

##### Lagrangian definition
def lagrangian(x,mu):
    x1 = x[0]
    x2 = x[1]
    mu1 = mu[0]
    mu2 = mu[1]
    return f(x) + mu1*phi1(x)+ mu2*phi2(x)
####################



import numpy as np

##### Lagrangian Hessian and Lagrangian Gradient computation of the Lagrangian 
def lagrangian_grad_hessian(x,mu):
    x1 = x[0]
    x2 = x[1]
    mu1 = mu[0]
    mu2 = mu[1]
    C11=1
    C12=-1
    C21=2
    C22=-1
#gx = 8*(x1-3) + mu1*C11 
#gy = 4*(x2-1) + mu1*C12
    gx = 8*(x1-3) + mu1*C11 + mu2*C21 
    gy = 4*(x2-1) + mu1*C12 + mu2*C22 
    g = np.array( [ gx  ,  gy  ] )
    H = np.array(  [ ( 8.0 , 0 )  ,  ( 0 , 4.0 )  ]  )  ### -> To complete DONE 
    return g,H
####


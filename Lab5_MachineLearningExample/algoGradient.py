#!/usr/bin/env python
# coding: utf-8

# # Gradient-based algorithms
# 
# In this notebook, we code our gradient-based optimization algorithms.

# # 1. Gradient algorithms
# 
# For minimizing a differentiable function $f:\mathbb{R}^n \to \mathbb{R}$, given:
# * the function to minimize `f`
# * a 1st order oracle `f_grad` (see `problem1.ipynb` for instance)
# * an initialization point `x0`
# * the sought precision `PREC` 
# * a maximal number of iterations `ITE_MAX` 
# 
# 
# these algorithms perform iterations of the form
# $$ x_{k+1} = x_k - \gamma_k \nabla f(x_k) $$
# where $\gamma_k$ is a stepsize to choose.

# ### 1.a. Constant stepsize gradient algorithm
# 


import numpy as np
import timeit

def gradient_algorithm(f , f_grad , x0 , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )

    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        g = f_grad(x)
        x = x - step*g  

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_tab


# ### 1.b. Adaptive stepsize gradient algorithm
# 

import numpy as np
import timeit


def gradient_adaptive_algorithm(f , f_grad , x0 , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )

    x_tab = np.copy(x)
    print("------------------------------------\nAdaptative Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        g = f_grad(x)
        x_prev = np.copy(x)

        x = x - step*g  #######  ITERATION

        if f(x)>f(x_prev):
            x = np.copy(x_prev)
            step = step/2
            print("stepsize: = {:0}".format(step))

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_tab


# ### 1.c. Wolfe Line search
# 
# 



import numpy as np
import timeit
from scipy.optimize import line_search

def gradient_Wolfe(f , f_grad , x0 , PREC , ITE_MAX ):
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )

    x_tab = np.copy(x)
    print("------------------------------------\n Gradient with Wolfe line search\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        g = f_grad(x)

        res = line_search(f, f_grad, x, -g, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=50)

        x = x - res[0]*g 

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_tab


# ### 1.d. Nesterov's Fast gradient algorithm
# 
# In a series of papers published in the 80's, Yu. Nesterov proposed an acceleration technique in order to make the worst case rate of the gradient algorithm from $\mathcal{O}(1/k)$ to  $\mathcal{O}(1/k^2)$. This technique is now immensely popular, notably in the machine learning and image processing communities.
# 
# The iterations of Nesterov's accelerated gradient are as such:
# $$ \left\{  \begin{array}{ll}  x_{k+1} = y_k - \gamma \nabla f(y_k) \\ y_{k+1} = x_{k+1} + \alpha_{k+1} (x_{k+1} - x_k )  \end{array}           \right. $$
# with 
# $$ \alpha_{k+1} = \frac{\lambda_k -1 }{\lambda_{k+1}} \text{ with } \lambda_0 = 0 \text{ and } \lambda_{k+1} = \frac{1+\sqrt{1+4\lambda_k^2}}{2} . $$
# 
# Although no clear intuition can be drawn, the extended point can be seen as an extension by inertia of the last points.
# 
# 
# Q. Fill the function below accordingly. 



import numpy as np
import timeit

def fast_gradient_algorithm(f , f_grad , x0 , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    y = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )



    x_tab = np.copy(x)
    print("------------------------------------\n Fast gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        g = f_grad(x)
        # TO FILL

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_tab


# # 2. Second Order algorithms
# 
# For minimizing a *twice* differentiable function $f:\mathbb{R}^n \to \mathbb{R}$, given:
# * the function to minimize `f`
# * a 2nd order oracle `f_grad_hessian` (see `problem1.ipynb` for instance)
# * an initialization point `x0`
# * the sought precision `PREC` 
# * a maximal number of iterations `ITE_MAX` 
# 
# 
# these algorithms perform iterations of the form
# $$ x_{k+1} = x_k - [\nabla^2 f(x_k) ]^{-1} \nabla f(x_k) .$$



import numpy as np
import timeit

def newton_algorithm(f , f_grad_hessian , x0 , PREC , ITE_MAX ):
    x = np.copy(x0)
    g0,H0 = f_grad_hessian(x0)
    stop = PREC*np.linalg.norm(g0 )

    x_tab = np.copy(x)
    print("------------------------------------\nNewton's algorithm\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        g,H = f_grad_hessian(x)
        x = x - np.linalg.solve(H,g)  

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_tab


# # 3. Quasi Newton algorithms
# 
# **BFGS.** (Broyden-Fletcher-Goldfarb-Shanno, 1970) The popular BFGS algorithm consist in performing the following iteration
# $$ x_{k+1}=x_k - \gamma_k W_k \nabla f(x_k)$$
# where $\gamma_k$ is given by Wolfe's line-search and positive definite matrix $W_k$ is computed as
# $$ W_{k+1}=W_k - \frac{s_k y_k^T W_k+W_k y_k s_k^T}{y_k^T s_k} +\left[1+\frac{y_k^T W_k y_k}{y_k^T s_k}\right]\frac{s_k s_k^T}{y_k^T s_k} $$
# with $s_k=x_{k+1}-x_{k}$ and $y_k=\nabla f(x_{k+1}) - \nabla f(x_{k})$.

# Q. Implement BFGS

import numpy as np
import timeit
from scipy.optimize import line_search

def bfgs(f , f_grad , x0 , PREC , ITE_MAX ):
    x = np.copy(x0)
    n = x0.size
    g =  f_grad(x0)
    sim_eval = 1
    stop = PREC*np.linalg.norm( g )

    W = np.eye(n)

    x_tab = np.copy(x)
    print("------------------------------------\n BFGS\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX): 

        x = x    # To fill

        x_tab = np.vstack((x_tab,x))

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_tab


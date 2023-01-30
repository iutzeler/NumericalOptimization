#!/usr/bin/env python
# coding: utf-8

# # Gradient-based algorithms
# 
# In this file, we code our gradient-based optimization algorithms.

#################################
# # 1. Gradient algorithms
##################################
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
# First, we consider the case where the stepsize is fixed over iterations and passed an argument `step` to the algorithm.



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
        x = x - step*g  #######  ITERATION

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab

# # 1.b Newton algorithm
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


# Q: Complete the code for the Newton iterations

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

        g, H = f_grad_hessian(x)
    
        ### COMPLETE

        x_tab = np.vstack((x_tab,x))
        
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab

#################################
# # 2. More involved functions
##################################

# # 2.b Adaptive stepsize gradient algorithm

# Q: Complete the adaptive gradient below using your intuition


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

	### COMPLETE

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab







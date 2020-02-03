#!/usr/bin/env python
# coding: utf-8

# # Proximal algorithms
# 
# In this notebook, we code our proximal optimization algorithms.

# # 1. Proximal Gradient algorithm
# 
# For minimizing a function $F:\mathbb{R}^n \to \mathbb{R}$ equal to $f+g$ where $f$ is differentiable and the $\mathbf{prox}$ of $g$ is known, given:
# * the function to minimize `F`
# * a 1st order oracle for $f$ `f_grad` 
# * a proximity operator for $g$ `g_prox` 
# * an initialization point `x0`
# * the sought precision `PREC` 
# * a maximal number of iterations `ITE_MAX` 
# * a display boolean variable `PRINT` 
# 
# these algorithms perform iterations of the form
# $$ x_{k+1} = \mathbf{prox}_{\gamma g}\left( x_k - \gamma \nabla f(x_k) \right) $$
# where $\gamma$ is a stepsize to choose.

# 
# 
# Q. How would you implement the precision stopping criterion?



import numpy as np
import timeit

def proximal_gradient_algorithm(F , f_grad , g_prox , x0 , step , PREC , ITE_MAX , PRINT ):
    x = np.copy(x0)
    x_tab = np.copy(x)
    if PRINT:
        print("------------------------------------\n Proximal gradient algorithm\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        g = f_grad(x)
        x = g_prox(x - step*g , step)  #######  ITERATION

        x_tab = np.vstack((x_tab,x))


    t_e =  timeit.default_timer()
    if PRINT:
        print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,F(x)))
    return x,x_tab


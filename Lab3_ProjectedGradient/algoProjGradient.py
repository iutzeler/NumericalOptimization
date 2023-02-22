#!/usr/bin/env python
# coding: utf-8

# # Projected Gradient-based algorithms
# 
# In this notebook, we code our Projected gradient-based optimization algorithms.
# We consider here 
# * Positivity constraints
# * Interval constraints

# # 1. Projected Gradient algorithms (for positivity or interval constraints)
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
# $$ x_{k+1} = P\left(x_k - \gamma_k \nabla f(x_k)\right) $$
# where $\gamma_k$ is a stepsize to choose and $P$ is the projector onto the convex constraint set. We only consider positivity and interval constraints.

# ### 1.a. Constant stepsize projected gradient algorithm for positivity constraints
# 
# First, we consider the case where the stepsize is fixed over iterations and passed an argument `step` to the algorithm.


# Q. Fill the function below accordingly. 



import numpy as np
import timeit

def positivity_gradient_algorithm(f , f_grad , x0 , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    g = f_grad(x) # we initialize both x and f_grad(x)
    stop = PREC*np.linalg.norm(g)

    epsilon = PREC*np.ones_like(x0)
    
    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
       
        x = x         #######  ITERATION --> To complete by the projection onto the set "x >= 0"
    
        ####### 
        x_tab = np.vstack((x_tab,x))
        ####### 
        ##########################################################
        #######  Why must the following stopping criteria be changed ? Propose a correct stopping rule
        #if np.linalg.norm(g) < stop:
        #    break
        ###############################################

        # To complete
        if ... :
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab


# ### 1.b. Constant stepsize projected gradient algorithm for interval constraints
# 
# First, we consider the case where the stepsize is fixed over iterations and passed an argument `step` to the algorithm.

# Q. Fill the function below accordingly. Then, test you algorithm in `2_Optimization100.ipynb [Sec. 1a]` for Problem 1.



import numpy as np
import timeit

def interval_gradient_algorithm(f , f_grad , x0 , infbound , supbound , step , PREC , ITE_MAX ):
    # compute the min of f with a gradient method with constant step under the constraint 
    # borninf < x < bornesup
    x = np.copy(x0)
    g = f_grad(x)
    stop = PREC*np.linalg.norm(g)
    zero = np.zeros_like(x0) 
    epsilon = PREC*np.ones_like(x0)

    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        x = x         #######  ITERATION --> To complete by the projection onto the set "x >= 0"
     

        x_tab = np.vstack((x_tab,x))

        #######  Why must the following stopping criteria be changed ? Propose a correct stopping rule
        #if np.linalg.norm(g) < stop:
        #    break
        
        # To complete
        if ... :
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab







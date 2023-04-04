#!/usr/bin/env python
# coding: utf-8

# # Uzawa algorithm
# 
# In this notebook, we code the Uzawa algorithm.
# We consider here 
# * Affine constraints (qualified)
# * Other (qualified) convex constraints may be tested but are out of the scope of the convergence theorem that we have proven

# # 1. Uzawa algorithm
# 
# For minimizing a differentiable function $f:\mathbb{R}^n \to \mathbb{R}$, under affine constraints given:
# * the function to minimize `f`
# * a 1st order oracle `f_grad` (see `problem1.ipynb` for instance)
# * the $m$ constraints are given by `phi` in the form $\phi(x)\le 0$ ; affine constraints are in the form of a matrix `C` $m\times n$ and a vector $d\in\mathbb{R}^m$, $\phi(x)=C x - d$.
# * an initialization point `x0`
# * the sought precision `PREC` 
# * a maximal number of iterations `ITE_MAX` 
# 
# 
# %
# 
# 
# This algorithm perform iterations of the form
# 
# Solve the minimization problem 
# $$ x^{k} = \mbox{arg}\min_{x} J(x) + \lambda^k \cdot \phi(x)$$
# or equivalently solve in $x_k$ with the Newton method
# $$ \nabla J(x^{k}) + C^t \lambda^k= 0 $$
# then one step of the projected gradient (projection onto $\mathbb{R_+}^m$) 
# $$ \lambda^{k+1} = P\left(\lambda^k + \rho \phi(x^k)\right) $$
# where $\rho$ is a stepsize to choose and $P$ is the projector onto the positivity constraint set. 
# 
# 
# %%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
# For more general (qualified) convex constraints $\phi_i, i=1,\ldots,m$, the first part of the iteration is
# 
# Solve in $x^k$ with the Newton method
# $$ \nabla J(x^{k}) +  \sum_{i=1}^m\lambda^k_i \nabla \phi_i(x^{k})= 0 $$
# or in the vectorial form
# $$ \nabla J(x^{k}) +  \lambda^k \cdot \nabla \phi(x^{k})= 0 $$
# the projected gradient step would be identical
# 

# ### 1.a. Uzawa method for affine constraints
# 
# We consider the case of affine constraints. They are passed to the algorithm as a matrix. The stepsize is fixed over iterations and passed an argument `step` to the algorithm.
# 
# S. Fill the function below accordingly. Then, test you algorithm.



import numpy as np
import timeit

def uzawa_affine_algorithm(f , lagrangian, lagrangian_grad_hessian,  phi, x0 , lambda0, rho , PREC , ITE_MAX):
    xk = np.copy(x0)
    lambdak = np.copy(lambda0)
    g,H = lagrangian_grad_hessian(xk,lambdak) # we initialize both x and f_grad(x)
    #stop = should be the verification of the KKT !!!!
    stop = PREC*np.linalg.norm(g)

    zerolambda = np.zeros_like(lambdak) #could be usefull... 
    zerox = np.zeros_like(xk)
    vphiprec = PREC * np.ones_like(lambdak)
    # print("vphiprec = ",vphiprec)
    
    x_tab = np.copy(xk)
    print("--------------------------\n Constant Stepsize projected gradient\n-------------------\nSTART    -- stepsize = {:0}".format(rho))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        # compute the min in x of L(x^k,lambda^k)=J(x^k) + sum_i lambda^k_i phi_(x_k)
        # (solving with with the Newton method  \nabla J(x_{k}) + C^t \lambda_k= 0 )
        # COMPLETE first the  newton_lagrangian_algo(lagrangian, lagrangian_grad_hessian, x0 , mu, PREC , ITE_MAX ) 
        # the Newton algorithm adapted to the lagrangian at FIXED mu (see appendix below)
        xk=newton_lagrangian_algo(lagrangian, lagrangian_grad_hessian , xk , lambdak , PREC , ITE_MAX )
       

        #######  ITERATION on lambda^k--> To complete by the projection onto the set "x >= 0"


        ####### 
        x_tab = np.vstack((x_tab,xk))

        #######  Why must the following stopping criteria be changed ? Propose a correct stopping rule
# convergence test ??? Should be KKT
#        print("lambdak = ({:.2f})  \n".format(lambdak[0]))
   

    
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(xk),xk[0],xk[1]))
    return lambdak,xk,x_tab


# ### 1.b. Uzawa method for convex constraints
# 
# We could consider the case of convex constraints. They are passed to the algorithm as a function $\phi$, whose components are convex functions $phi_i, i=1,\ldots,m$. 
# The stepsize is fixed over iterations and passed an argument `step` to the algorithm.
# 
# Obviously the gradient and the Hessian matrices of $J$ and $\phi_i$ are necessary.
# 
# > Could you use the previous function in its form to test more general (qualified) convex constraints ?
# 
# 
# 

# ## 2. Annex: Newton method for Min_x=L(x,mu) (Second Order algorithms)
# 
# For minimizing according to $x$ a *twice* differentiable function 
# $L(x,\mu):\mathbb{R}^n \times \mathbb{R}_+^m \to \mathbb{R}$,
# at fixed $\mu$ given:
# * the function to minimize `Lagrangian(x,mu)`
# * a 2nd order oracle `Lagrangian_grad_hessian` (see `problem31.ipynb` for instance)
# * an initialization point `x0`
# * the fixed vector `mu`
# * the sought precision `PREC` 
# * a maximal number of iterations `ITE_MAX` 
# 
# 
# these algorithms perform iterations of the form
# $$ x^{k+1} = x^k - [\nabla^2 L(x^k,\mu) ]^{-1} \nabla L(x^k,\mu) .$$
# where the Hessian and gradient are according to $x$ at fixed $\mu$



import numpy as np
import timeit

def newton_lagrangian_algo(lagrangian, lagrangian_grad_hessian , x0 , mu , PREC , ITE_MAX ):
    x = np.copy(x0)
    g0,H0 = lagrangian_grad_hessian(x0,mu)
    stop = PREC*np.linalg.norm(g0 )
    
    #### x_tab = np.copy(x)
    print("------------------------------------\nNewton's algorithm\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
    

        x = x   #######  ITERATION -> to complete

        #### x_tab = np.vstack((x_tab,x))
        
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("Newton FINISHED -- {:d} iterations / {:.6f}s -- langrangian final value: {:f} at point x ({:.2f},{:.2f}) \n\n".format(k,t_e-t_s,lagrangian(x,mu),x[0],x[1]))
    #### return x,x_tab 
    return x








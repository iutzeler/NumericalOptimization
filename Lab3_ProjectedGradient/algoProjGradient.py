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
    #stop = PREC*np.linalg.norm(f_grad(x0) )
    stop = PREC*np.linalg.norm(g)

    zero = np.zeros_like(x0) #could be usefull... 
    epsilon = PREC*np.ones_like(x0)
    
    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
       
        x = x         #######  ITERATION --> To complete by the projection onto the set "x >= 0"
    
        ####### 
        x_tab = np.vstack((x_tab,x))
        ####### 
        ## un test basé sur l'annulation des comp de grad (gistested)        
        # gtested = np.copy(g)
        # if((x[0]<=0.000000001) and (g[0]>0.0)):
        #    gtested[0] = 0. 
        #if((x[1]<=0.000000001) and (g[1]>0.0)):
        #    gtested[1] = 0. 
        #print("x = ({:.2f},{:.2f}) ; gradf = ({:.2f},{:.2f}) ; gtested = ({:.2f},{:.2f})\n".format(x[0],x[1],g[0],g[1],gtested[0],gtested[1]))
        #if np.linalg.norm(gtested) < stop: ### L’opérateur * ne fait que multiplier terme à terme deux tableaux de même dimension:
        #    break
        ##########################################################
        #######  Why must the following stopping criteria be changed ? Propose a correct stopping rule
        #if np.linalg.norm(g) < stop:
        #    break
        ###############################################
        #a = np.array([1, 2, 4, 6])

        #####################################
        # we want to use the KKT as stopping rules
        # x >=0 (by construction of the projected gradient thus not tested
        # g >=0 => must be tested !
        # sum_i x(i)*g(i)=0 or np.dot(x,g)=0
        positive_grad = np.all(g>-epsilon) #numerical test for g >=0
        null_dot = np.absolute(np.dot(x,g)) < PREC
        if(positive_grad and null_dot) :
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
    zero = np.zeros_like(x0) #could be usefull...
    epsilon = PREC*np.ones_like(x0)
    #stop = PREC*np.linalg.norm(f_grad(x0) )

    x_tab = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):

        x = x         #######  ITERATION --> To complete by the projection onto the set "x >= 0"
     

        x_tab = np.vstack((x_tab,x))

        ####### 
        #gistested = ((g < zero) or (x>zero))
        #gistested.astype(np.int)
        #if np.linalg.norm(g*gistested) < stop: ### L’opérateur * ne fait que multiplier terme à terme deux tableaux de même dimension:
        #    break
        # gtested = np.zeros_like(g)
        #if((g[0]<0.) or (x[0])>0):
        #    gtested[0] = g[0] 
        #if((g[1]<0.) or (x[1])>0):
        #    gtested[1] = g[1] 
        gtested = np.copy(g)
        if((x[0]<= (infbound[0] + 0.000000001)) and (g[0]>0.0)):
            gtested[0] = 0. 
        if((x[1]<=(infbound[1] +  0.000000001)) and (g[1]>0.0)):
            gtested[1] = 0. 
        if((x[0]>= (supbound[0] - 0.000000001)) and (g[0]<0.0)):
            gtested[0] = 0. 
        if((x[1]>=(supbound[1] -  0.000000001)) and (g[1]<0.0)):
            gtested[1] = 0. 
        print("gtested = ({:.2f},{:.2f})\n".format(gtested[0],gtested[1]))
        if np.linalg.norm(gtested) < stop: ### L’opérateur * ne fait que multiplier terme à terme deux tableaux de même dimension:
            break
        
        #######  Why must the following stopping criteria be changed ? Propose a correct stopping rule
        #if np.linalg.norm(g) < stop:
        #    break
        # we want to use the KKT as stopping rules
        # x >=0 (by construction of the projected gradient thus not tested
        # here for all i
        #  either x(i)==infbound(i) and g(i)>=0 
        #      or x(i)==supbound(i) and g(i)<=0
        #      or g(i)==0 (by construction infbound(i)<=x(i)<= supbound(i))
        cond1 = np.logical_and(x <= (infbound+epsilon) , g >= -epsilon)
        cond2 = np.logical_and(x >= (supbound+epsilon) , g <= epsilon)
        cond3 = np.absolute(g) < epsilon 
        if np.all(np.logical_or(cond1, np.logical_or(cond2,cond3))) :
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab


# ### 2.a. Adaptive stepsize gradient algorithm
# 
# Now, we consider the case where the stepsize is fixed over iterations and passed an argument `step` to the algorithm.


# Q. Examine the behavior of the constant stepsize gradient algorithm and try to solve the problem by changing the stepsizes.



def interval_gradient_adaptive_algorithm(f , f_grad , x0 , infbound , supbound , step , PREC , ITE_MAX ):
    x = np.copy(x0)
    g = f_grad(x)
    stop = PREC*np.linalg.norm(g)

    x_tab = np.copy(x)
    print("------------------------------------\nAdaptative Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        
        x_prev = np.copy(x)
        
        x = x - step*g  ## ITERATION -> To adapt in order to introduce interval constraints x in [infbound , supbound] 

            
        x_tab = np.vstack((x_tab,x))
        
        g = f_grad(x)
        
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f} at point ({:.2f},{:.2f})\n\n".format(k,t_e-t_s,f(x),x[0],x[1]))
    return x,x_tab






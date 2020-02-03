#!/usr/bin/env python
# coding: utf-8

# # Problem 3
# 
# 
# The objective of Problem 3 is to minimize non-convex smooth Rosenbrock function $r$ on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# r: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  &  (1-x_1)^2 + 100(x_2-x_1^2)^2
# \end{array}$$




##### Function definition
def f(x):
	"""Rosenbrock."""
	x1 = x[0]
	x2 = x[1]
	return (1-x1)**2+100*(x2-x1**2)**2
####

##### Plot parameters f
x1_min = -1.5
x1_max = 1.55
x2_min = -0.2
x2_max = 1.5
nb_points = 200
vmin = 0
vmax = 120
levels = [0.05,1,5,15,50,100,200]
title = 'Rosenbrock function'
####





### Oracles


# Q: Complete the first order oracle `f_grad`.




import numpy as np

##### Gradient oracle
def f_grad(x):

	return  0.0 ### To complete
####


# Q: Fill the following second order oracle `f_grad_hessian`.


import numpy as np

##### Hessian scaled Gradient computation
def f_grad_hessian(x):
                
	return g,H ### To complete
####


#!/usr/bin/env python
# coding: utf-8

# # Problem 5
# 
# 
# The objective of Problem 5 is to minimize a polyhedral function $p$ on $\mathbb{R}^2$ (unconstrained): 
# 
# $$\begin{array}{rrcll}
# p: & \mathbb{R}^2 & \to &\mathbb{R}\\
# & (x_1,x_2) & \mapsto  &  \left| x_1-3 \right|  + 2\left| x_2-1\right| .
# \end{array}$$




##### Function definition
def f(x):
	x1 = x[0]
	x2 = x[1]
	return np.abs(x1-3)+2*np.abs(x2-1)
####

##### Plot parameters f
x1_min = -0.5
x1_max = 5.5
x2_min = -0.5
x2_max = 5.5
nb_points = 200
levels = [0.05,0.5,1,2,5]
vmin = 0
vmax = 5
title = 'polyhedral'
####


### Oracles


# Q: Compute a first order oracle `f_grad`. Is it unique?



import numpy as np

##### Gradient oracle
def f_grad(x):

	return g ### To complete
####


# Q: What about a second order oracle?






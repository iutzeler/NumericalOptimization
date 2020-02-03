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





###### Useful Parameters
L = 8        # Lipschitz constant of the gradient


### Oracles

# Q: Complete the first order oracle `f_grad`.




import numpy as np

##### Gradient oracle
def f_grad(x):

	return  ### To complete


# Q: Does a second order oracle exist for any point?



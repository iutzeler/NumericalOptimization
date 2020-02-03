#!/usr/bin/env python
# coding: utf-8

# # a Toy problem

# We consider the first illustrative example of the original paper
# 
#     Candes, E., Tao, T. "The Dantzig selector: Statistical estimation when p is much larger than n". 
#     The Annals of Statistics, 2007

# In this first example, the design matrix $X$ has $m = 72$ rows and $n = 256$ columns, with independent Gaussian entries (and then normalized so that the columns have unit-norm). We then select $\theta$ with $S := |\{i : \theta_i = 0\}| = 8$, and form $y = X\theta + \xi$, where the $\xi_i$’s are i.i.d. $\mathcal{N}(0, \sigma^2 )$. The noise level is adjusted so that 
# $$ \sigma = \frac{1}{3} \sqrt{\frac{S}{n}} .$$

# ### Problem


import numpy as np

# Parameters
m = 72
n = 256

S = 8

sigma = 1/3.0 * np.sqrt(S/float(m))

# X creation
X = np.random.randn(m, n)

n_col = np.linalg.norm(X, axis=0)
X = np.dot(X,np.diag(1/n_col))    # Normalization per column [Get rid of it for the "To go further" part!]

# theta creation
theta = np.zeros(n)
non_null = np.random.choice(n, S)
theta[non_null] = np.random.randn(S)


# y creation
y = np.dot(X,theta) + sigma*np.random.randn(m)


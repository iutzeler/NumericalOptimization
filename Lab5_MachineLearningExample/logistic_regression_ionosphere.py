#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Problem
# 
# 
# 
# ### Machine Learning as an Optimization problem
# 
# We have some *data*  $\mathcal{D}$ consisting of $m$ *examples* $\{d_i\}$; each example consisting of a *feature* vector $a_i\in\mathbb{R}^d$ and an *observation* $b_i\in \mathcal{O}$: $\mathcal{D} = \{[a_i,b_i]\}_{i=1..m}$. In this lab, we will consider the <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.names">ionosphere</a> dataset.
# 
# 
# The goal of *supervised learning* is to construct a predictor for the observations when given feature vectors.
# 
# 
# A popular approach is based on *linear models* which are based on finding a *parameter* $x$ such that the real number $\langle a_i , x \rangle$ is used to predict the value of the observation through a *predictor function* $g:\mathbb{R}\to \mathcal{O}$: $g(\langle a_i , x \rangle)$ is the predicted value from $a_i$.
# 
# 
# In order to find such a parameter, we use the available data and a *loss* $\ell$ that penalizes the error made between the predicted $g(\langle a_i , x \rangle)$ and observed $b_i$ values. For each example $i$, the corresponding error function for a parameter $x$ is $f_i(x) =   \ell( g(\langle a_i , x \rangle) ; b_i )$. Using the whole data, the parameter that minimizes the total error is the solution of the minimization problem
# $$ \min_{x\in\mathbb{R}^d}  \frac{1}{m} \sum_{i=1}^m f_i(x) = \frac{1}{m} \sum_{i=1}^m  \ell( g(\langle a_i , x \rangle) ; b_i ). $$
# 
# 
# ### Binary Classification with Logisitic Regression
# 
# In our setup, the observations are binary: $\mathcal{O} = \{-1 , +1 \}$, and the *Logistic loss* is used to form the following optimization problem
# \begin{align*}
# \min_{x\in\mathbb{R}^d } f(x) := \frac{1}{m}  \sum_{i=1}^m  \log( 1+\exp(-b_i \langle a_i,x \rangle) ) + \frac{\lam2bda}{2} \|x\|_2^2.
# \end{align*}
# where the last term is added as a regularization (of type $\ell_2$, aka Tikhnov) to prevent overfitting.
# 
# Under some statistical hypotheses, $x^\star = \arg\min f(x)$ maximizes the likelihood of the labels knowing the features vector. Then, for a new point $d$ with features vector $a$, 
# $$ p_1(a) = \mathbb{P}[d\in \text{ class }  +1] = \frac{1}{1+\exp(-\langle a;x^\star \rangle)} $$
# Thus, from $a$, if $p_1(a)$ is close to $1$, one can decide that $d$ belongs to class $1$; and the opposite decision if $p(a)$ is close to $0$. Between the two, the appreciation is left to the data scientist depending on the application.
# 
# 
# # Objective of the optimizer
# 
# Given oracles for the function and its gradient, as well as an upper-bound of the Lipschitz constant $L$ of the gradient, find a minimizer of $f$.
# 

# ### Function definition 



import numpy as np
import csv
from sklearn import preprocessing

file = open('ionosphere.data')

d = 34
n = d+1 # Variable size + intercept

m = 351 # Number of examples

lam2 = 0.001 # regularization best:0.001

A = np.zeros((m,d))
b = np.zeros(m)

reader = csv.reader(file, delimiter=',')
i = 0
for row in reader:
    A[i] = np.array(row[:d]) 
    if row[d] == 'b':
        b[i] = -1.0
    else:
        b[i] =  1.0 
    i+=1

scaler = preprocessing.StandardScaler().fit(A)
A = scaler.transform(A)  

# Adding an intercept
A_inter = np.ones((m,n))
A_inter[:,:-1] = A
A = A_inter


L = 0.25*max(np.linalg.norm(A,2,axis=1))**2 + lam2


# ## Oracles




def f(x):
    l = 0.0
    for i in range(A.shape[0]):
        if b[i] > 0 :
            l += np.log( 1 + np.exp(-np.dot( A[i] , x ) ) ) 
        else:
            l += np.log( 1 + np.exp(np.dot( A[i] , x ) ) ) 
    return l/m + lam2/2.0*np.dot(x,x)

def f_grad(x):
    g = np.zeros(n)
    for i in range(A.shape[0]):
        if b[i] > 0:
            g += -A[i]/( 1 + np.exp(np.dot( A[i] , x ) ) ) 
        else:
            g += A[i]/( 1 + np.exp(-np.dot( A[i] , x ) ) ) 
    return g/m + lam2*x

def f_grad_hessian(x):
    g = np.zeros(n)
    H = np.zeros((n,n))
    for i in range(A.shape[0]):
        if b[i] > 0:
            g += -A[i]/( 1 + np.exp(np.dot( A[i] , x ) ) ) 
            H +=  (np.exp(np.dot( A[i] , x ))/( 1 + np.exp(np.dot( A[i] , x ) ) )**2)*np.outer(A[i],A[i])
        else:
            g += A[i]/( 1 + np.exp(-np.dot( A[i] , x ) ) ) 
            H +=  (np.exp(-np.dot( A[i] , x ))/( 1 + np.exp(-np.dot( A[i] , x ) ) )**2)*np.outer(A[i],A[i])
    g =  g/m + lam2*x
    H = H/m + lam2*np.eye(n)
    return g,H


# ## Prediction Function



def prediction(w,PRINT=False):
    pred = np.zeros(A.shape[0])
    perf = 0
    for i in range(A.shape[0]):
        p = 1.0/( 1 + np.exp(-np.dot( A[i] , w ) ) )
        if p>0.5:
            pred[i] = 1.0
            if b[i]>0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),1,(p-0.5)*200,correct))
        else:
            pred[i] = -1.0
            if b[i]<0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),-1,100-(0.5-p)*200,correct))
    return pred,float(perf)/A.shape[0]



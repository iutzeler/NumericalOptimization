#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Problem
# 
# 
# 
# ### Machine Learning as an Optimization problem
# 
# We have some *data*  $\mathcal{D}$ consisting of $m$ *examples* $\{d_i\}$; each example consisting of a *feature* vector $a_i\in\mathbb{R}^d$ and an *observation* $b_i\in \mathcal{O}$: $\mathcal{D} = \{[a_i,b_i]\}_{i=1..m}$. In this lab, we will consider the <a href="http://archive.ics.uci.edu/ml/datasets/Student+Performance">student performance</a> dataset.
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
# 
# # Regularized Problem 
# 
# In this lab, we will consider an $\ell_1$ regularization to promote sparsity of the iterates. A sparse final solution would select the most important features. The new function (below) is non-smooth but it has a smooth part, $f$, the same as in Lab3; and a non-smooth part, $g$, that we will treat with proximal operations.
# 
# \begin{align*}
# \min_{x\in\mathbb{R}^d } F(x) := \underbrace{ \frac{1}{m}  \sum_{i=1}^m  \log( 1+\exp(-b_i \langle a_i,x \rangle) ) + \frac{\lambda_2}{2} \|x\|_2^2}_{f(x)} + \underbrace{\lambda_1 \|x\|_1 }_{g(x)}.
# \end{align*}
# 
# 


# ### Function definition 



import numpy as np
import csv
from sklearn import preprocessing

#### File reading
dat_file = np.load('student.npz')
A = dat_file['A_learn']
final_grades = dat_file['b_learn']
m = final_grades.size
b = np.zeros(m)
for i in range(m):
    if final_grades[i]>11:
        b[i] = 1.0
    else:
        b[i] = -1.0

A_test = dat_file['A_test']
final_grades_test = dat_file['b_test']
m_test = final_grades_test.size
b_test = np.zeros(m_test)
for i in range(m_test):
    if final_grades_test[i]>11:
        b_test[i] = 1.0
    else:
        b_test[i] = -1.0


d = 27 # features
n = d+1 # with the intercept




lam2 = 0.1 # for the 2-norm regularization best:0.1
lam1 = 0.03 # for the 1-norm regularization best:0.03


L = 0.25*max(np.linalg.norm(A,2,axis=1))**2 + lam2


# ## Oracles
# 
# ### Related to function $f$



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


# ## Related to function $f_i$ (one example)

# Q. To Fill



def f_grad_ex(x,i):
    g = np.zeros(n)
    
    #### TODO
    
    return g


# ### Related to function $g$


def g(x):
    return lam1*np.linalg.norm(x,1)

def g_prox(x,gamma):
    p = np.zeros(n)
    for i in range(n):
        if x[i] < - lam1*gamma:
            p[i] = x[i] + lam1*gamma
        if x[i] > lam1*gamma:
            p[i] = x[i] - lam1*gamma
    return p


# ### Related to function $F$



def F(x):
    return f(x) + g(x)


# ## Prediction Function



def prediction_train(w,PRINT):
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

def prediction_test(w,PRINT):
    pred = np.zeros(A_test.shape[0])
    perf = 0
    for i in range(A_test.shape[0]):
        p = 1.0/( 1 + np.exp(-np.dot( A_test[i] , w ) ) )
        if p>0.5:
            pred[i] = 1.0
            if b_test[i]>0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),1,(p-0.5)*200,correct))
        else:
            pred[i] = -1.0
            if b_test[i]<0:
                correct = "True"
                perf += 1
            else:
                correct = "False"
            if PRINT:
                print("True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(int(b[i]),-1,100-(0.5-p)*200,correct))
    return pred,float(perf)/A_test.shape[0]


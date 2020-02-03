#!/usr/bin/env python
# coding: utf-8

# # Regularized Problem
# 
# In this lab, we add an $\ell_1$ regularization to promote sparsity of the iterates. The function (below) is non-smooth but it has a smooth part, $f$, and a non-smooth part, $g$, that we will treat with proximal operations.
# 
# \begin{align*}
# \min_{x\in\mathbb{R}^d } F(x) := \underbrace{ \frac{1}{m}  \sum_{i=1}^m  \log( 1+\exp(-b_i \langle a_i,x \rangle) ) + \frac{\lambda_2}{2} \|x\|_2^2}_{f(x)} + \underbrace{\lambda_1 \|x\|_1 }_{g(x)}.
# \end{align*}

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


L = 0.25*max(np.linalg.norm(A,2,axis=1))**2 + lam2


d = 27 # features
n = d+1 # with the intercept



lam1 = 0.03 # for the 1-norm regularization best:0.03
lam2 = 0.0



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


# ### Related to function $g$ [TODO]


def g(x):
    return lam1*np.linalg.norm(x,1)

def g_prox(x,gamma):
    p = np.zeros(n)
#TODO
    return p


# ### Related to function $F$



def F(x):
    return f(x) + g(x)


# ## Prediction Functions


def prediction_train(w,PRINT=False):
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

def prediction_test(w,PRINT=False):
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




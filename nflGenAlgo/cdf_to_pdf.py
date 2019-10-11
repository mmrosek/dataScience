#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:58:24 2019

@author: miller
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import interpolate

### Numerical derivation of CDF --> PDF ###
x=np.array(list(range(-1000,1000,2)))
y=np.array(x)**2
# np.diff --> discrete difference b/w consecutive elements 
# --> needs to be divided by the length b/w elements to yield derivative
deriv=np.diff(y) / (x[-1] - x[-2])
plt.plot(deriv)
plt.plot(x,y)

x = np.linspace(5,15,11)
cdf = np.array([0,0.02,0.06,0.14,0.3,0.52,0.72,0.84,0.94,0.98,1])
plt.plot(x,cdf)
pdf=np.diff(cdf) / (cdf[-1] - cdf[-2]) # derivative of cdf = pdf
plt.plot(x[:-1], pdf)

###########################################

### Interpolation ###

# Interpolation - pdf
# k --> degree of spline
tck = interpolate.splrep(x[:-1], pdf, s=0, k=2)
xnew = np.arange(5, 15, 0.01)
ynew = interpolate.splev(xnew, tck, der=0)
plt.plot(xnew, ynew)


# Interpolation - cdf
tck = interpolate.splrep(x, cdf, s=0)
xnew = np.arange(5, 15, 0.02)
ynew = interpolate.splev(xnew, tck, der=0)
plt.plot(xnew, ynew)

############################################
### Inverse transform sampling ###

# Interpolation - cdf
tck = interpolate.splrep(x, cdf, s=0)
xnew = np.arange(5, 15, 0.02)
ynew = interpolate.splev(xnew, tck, der=0) # need to make sure this has enough values to cover 1-100
plt.plot(xnew, ynew)

min_key=1
max_key=100
keys=np.rint(ynew*100)
# taking mean of all cdf values belonging to same integer key 
#(i.e. if 25th percentile (0.246, 0.251, 0.253, etc.) ~ [12.1,12.2,12.3, etc.])
vals=[ xnew[ keys == i].mean() for i in range(min_key,max_key+1)] 
int_keys=[i for i in range(min_key,max_key+1)]
inv_cdf_dict=dict(zip(int_keys,vals))

n_draws=10
draws = [inv_cdf_dict[round(np.random.randint(1,100))] for i in range(n_draws)]
draws


### Generating random normal draws using inverse transform sampling ###
# Essence: F_inv_x(U) ~ f(x)
# https://en.wikipedia.org/wiki/Inverse_transform_sampling

# norm.ppf(x) = percent point function = quantile function = inverse CDF
# --> returns value needed to be greater than x% of standard normal distribution
# --> Ex: norm.ppf(0.5) = 0 --> 0 > standard normally distributed RV 50% of time
draws = np.array([norm.ppf( (i+1e-6)/1000) for i in range(1,1000)])
plt.hist(draws)


### Interpolation of empirical cdf ###
x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
y = np.sin(x)
plt.plot(x,y)

tck = interpolate.splrep(x, y, s=0)
tck

xnew = np.arange(0, 2*np.pi, np.pi/50)
ynew = interpolate.splev(xnew, tck, der=0)
plt.plot(xnew, ynew)


plt.figure()
plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation')
plt.show()



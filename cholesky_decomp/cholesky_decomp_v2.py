#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:54:11 2019

@author: miller
"""
import numpy as np
import matplotlib.pyplot as plt

'''
Numerical calculation of PDF from CDF (numerical derivation) --> done

Get real correlations --> done
How will correlation change when having multiple players on same team? 
--> Will they eventually be negatively correlated at some point? I.e. WRs eating into each other workload

Generate player distributions
Test on simulated example (simulate players as gaussian, variance proportional to mean projection)
Use QRF to get more accurate distributions
--> Hopefully use available data in projection csvs to generate, may need to scrape

Code genetic algorithm to evaluate C-VaR
--> Best data structure(s) for players / correlations to efficiently correlate distributions

Is it useful to have more than just mean/stdev for each player? Is drawing from actual distribution beneficial?
https://math.stackexchange.com/questions/2075300/does-using-a-cholesky-decomposition-to-generate-correlated-samples-preserve-the?rq=1
'''

# Drawing samples
n_samples=1000
p1=np.random.normal(10, 5, size=(1,n_samples))
p2=np.random.normal(12, 4, size=(1,n_samples))
p3=np.random.normal(8, 3, size=(1,n_samples))
X = np.concatenate( (p1,p2,p3), axis=0) 
mn_init=X.mean(axis=1)
std_init=X.std(axis=1)
print(np.cov(X))

# Plotting distribution over lineup total score
totals=X.sum(axis=0)
plt.hist(totals, bins = [i for i in range(6,54,3)]) 
plt.title("histogram") 
plt.show()

# Standardizing X
X_cent = X - X.mean(axis=1).reshape(-1,1)
print(f'Mean by row/player: {np.mean(X_cent, axis=1)}')
X_std = X_cent/np.std(X_cent, axis=1).reshape(-1,1)

# -------------------------------------------------------------------------

### Deriving covariance matrix from known correlations b/w players/positions ###



corr_dict = {'qb-wr1':0.47, 'qb-wr2':0.38,
             'qb-te1':0.41, 'qb-
            '
            }

# Ex: Player 1 and 3 are on the same team and historically 
# have correlation of 0.3
corr12 = 0.1
corr13 = 0.3
corr23 = 0.2

s1=std_init[0]
s2=std_init[1]
s3=std_init[2]

des_cov = np.array( [ [s1**2, corr12*s1*s2, corr13*s1*s3], 
                       [corr12*s1*s2, s2**2 , corr23*s2*s3], 
                      [corr13*s1*s3, corr23*s2*s3, s3**2]  ])

# Cholesky decomp of desired covariance matrix
L=np.linalg.cholesky(des_cov)

# Multiplying X_std by CD(cov)
X_chol = L@X_std
print(f'Covariance after CD, before adding back means:\n {np.cov(X_chol)}')
X_final = X_chol + X.mean(axis=1).reshape(-1,1)
# X_final = mu + Z*L; where mu = player means, L = mean 0, unit variance RVs
print(f'Covariance after CD adding back means:\n {np.cov(X_final)}')

# Calculating correlation matrix
non_centered_outer_prod=np.std(X_final,axis=1).reshape(1,-1) * np.std(X_final,axis=1).reshape(-1,1)
print(f'Final correlation matrix:\n {np.cov(X_final) / non_centered_outer_prod}')

# Compare to original, mean seems to be the same but not stdev?
X_final.mean(axis=1)==mn_init
print(f'Stdev diffs, some seem to change a bit, maybe not in the limit: \n{X_final.std(axis=1)-std_init}')

# Plotting to compare to joint distribution (totals)
totals=X_final.sum(axis=0)
plt.hist(totals, bins = [i for i in range(6,54,3)]) 
plt.title("Correlated") 
plt.show()

# Investigating individual players
idx=2
totals=X_final[idx,:]
plt.hist(totals, bins = [i for i in range(0,26,1)]) 
plt.title("Player 0 Post-CD") 
plt.show()

totals=X[idx,:]
plt.hist(totals, bins = [i for i in range(0,26,1)]) 
plt.title("Pre-CD") 
plt.show()

### Proving Expectation of outer product of matrices w/ rows as uncorrelated, mean 0 unit variance vectors = Identity matrix ###
# Traditional case is where n_obs = 1 (i.e. single observation of each random variable)
# Theoretically, with 1 obs still = identity, numerically limits to identity as n_obs increase 
# Must divide by n_obs to recover identity / covariance matrix
from scipy.stats import norm
n_obs=1000
n_rvs=2
r = norm.rvs(size=(n_rvs, n_obs))
r_centered = r-(r.mean(axis=1).reshape(-1,1))
(r_centered @ r_centered.T) / (n_obs-1) # Covariance matrix
np.cov(r_centered) # Same result as line above

### Numerical derivation of CDF --> PDF ###
x=np.array(list(range(-1000,1000,2)))
y=np.array(x)**2
# np.diff --> discrete difference b/w consecutive elements 
# --> needs to be divided by the length b/w elements to yield derivative
deriv=np.diff(y) / (x[-1] - x[-2])
plt.plot(deriv)
plt.plot(x,y)

cdf_x = np.linspace(5,15,11)
cdf_y = np.array([0,0.02,0.06,0.14,0.3,0.52,0.72,0.84,0.94,0.98,1])
plt.plot(cdf_x,cdf_y)
deriv=np.diff(cdf_y) / (cdf_x[-1] - cdf_x[-2])
plt.plot(deriv)










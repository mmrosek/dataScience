#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:54:11 2019

@author: miller
"""
import numpy as np
import matplotlib.pyplot as plt

'''
How to use correlation to get covariance?
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
std_init[0]

# Plotting distribution over lineup total score
totals=X.sum(axis=0)
plt.hist(totals, bins = [9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]) 
plt.title("histogram") 
plt.show()


# Standardizing X
center=True
if center: X_cent = X - X.mean(axis=1).reshape(-1,1)
else: X_cent = X.copy()
print(f'Mean by row/player: {np.mean(X_cent, axis=1)}')
X_std = X_cent/np.std(X_cent, axis=1).reshape(-1,1)
np.var(X_std, axis=1)

np.cov(X) / np.cov(X_std) # Interesting

# -------------------------------------------

# Desired cov matrix derived from correlation


# Ex: Player 1 and 3 are on the same team and historically 
# have correlation of 0.3

### NEED TO VERIFY THIS LOGIC ###
#from scipy.stats import pearsonr
## calculate Pearson's correlation
#corr, _ = pearsonr(rands.reshape(-1,), rands2.reshape(-1,))
#corr
#covariance(X, Y) / (stdv(X) * stdv(Y))

corr12 = 0.1
corr13 = 0.3
corr23 = 0.2

s1=std_init[0]
s2=std_init[1]
s3=std_init[2]

des_cov = np.array( [ [s1**2, corr12*s1*s2, corr13*s1*s3], 
                       [corr12*s1*s2, s2**2 , corr23*s2*s3], 
                      [corr13*s1*s3, corr23*s2*s3, s3**2]  ])

L=np.linalg.cholesky(des_cov)

# Multiplying X_std by CD(cov)
X_chol = L@X_std
print(f'Covariance after CD, before un-normalizing:\n {np.cov(X_chol)}')
X_final = X_chol + X.mean(axis=1).reshape(-1,1)
print(f'Covariance after CD and un-normalizing:\n {np.cov(X_final)}')

non_centered_outer_prod=np.std(X_final,axis=1).reshape(1,-1) * np.std(X_final,axis=1).reshape(-1,1)
print(f'Final correlation matrix:\n {np.cov(X_final) / non_centered_outer_prod}')


# Compare to original, mean seems to be the same but not stdev?
X_final.mean(axis=1)==mn_init
print(f'Stdev diffs, some seem to change a bit, maybe not in the limit: \n{X_final.std(axis=1)-std_init}')

# Plotting to compare to joint distribution (totals)
totals=X_final.sum(axis=0)
plt.hist(totals, bins = [5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]) 
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


X[2,:20]
X_final[2,:20]









import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid


# Choice of cholesky or eigenvector method.
method = 'cholesky'
#method = 'eigenvectors'

num_samples = 400

# The desired covariance matrix.
r = np.array([
        [  3.40, -2.75, -2.00],
        [ -2.75,  5.50,  1.50],
        [ -2.00,  1.50,  1.25]
    ])

# Generate samples from three independent normally distributed random
# variables (with mean 0 and std. dev. 1).
x = norm.rvs(size=(3, num_samples))


# We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
# the Cholesky decomposition, or the we can construct `c` from the
# eigenvectors and eigenvalues.

if method == 'cholesky':
    # Compute the Cholesky decomposition.
    c = cholesky(r, lower=True)
else:
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = eigh(r)
    # Construct c, so c*c^T = r.
    c = np.dot(evecs, np.diag(np.sqrt(evals)))

# Convert the data to correlated random variables. 
y = np.dot(c, x)

np.cov(y)

#
# Plot various projections of the samples.
#
subplot(2,2,1)
plot(y[0], y[1], 'b.')
ylabel('y[1]')
axis('equal')
grid(True)

subplot(2,2,3)
plot(y[0], y[2], 'b.')
xlabel('y[0]')
ylabel('y[2]')
axis('equal')
grid(True)

subplot(2,2,4)
plot(y[1], y[2], 'b.')
xlabel('y[1]')
axis('equal')
grid(True)

show()

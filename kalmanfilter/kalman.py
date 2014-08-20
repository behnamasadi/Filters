#http://ascratchpad.blogspot.com/2010/03/kalman-filter-in-python.html
# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Adapted by Burak Bayramli, 2010


from pylab import *
from numpy import *

class Kalman:
    def __init__(self, ndim):
        self.ndim = ndim
        self.Sigma_x = eye(ndim)*1e-5
        self.A = eye(ndim)
        self.H = eye(ndim)
        self.mu_hat = 0
        self.cov = eye(ndim)
        self.R = eye(ndim)*0.01

    def update(self, obs):

        # Make prediction
        self.mu_hat_est = dot(self.A,self.mu_hat)
        self.cov_est = dot(self.A,dot(self.cov,transpose(self.A))) + self.Sigma_x

        # Update estimate
        self.error_mu = obs - dot(self.H,self.mu_hat_est)
        self.error_cov = dot(self.H,dot(self.cov,transpose(self.H))) + self.R
        self.K = dot(dot(self.cov_est,transpose(self.H)),linalg.inv(self.error_cov))
        self.mu_hat = self.mu_hat_est + dot(self.K,self.error_mu)
        if ndim>1:
            self.cov = dot((eye(self.ndim) - dot(self.K,self.H)),self.cov_est)
        else:
            self.cov = (1-self.K)*self.cov_est 
            
if __name__ == "__main__":		
    print "***** 1d ***********"
    ndim = 1
    nsteps = 50
    k = Kalman(ndim)    
    mu_init=array([-0.37727])
    cov_init=0.1*ones((ndim))
    obs = random.normal(mu_init,cov_init,(ndim, nsteps))
    for t in range(ndim,nsteps):
        k.update(obs[:,t])
    print k.mu_hat_est

    print "***** 2d ***********"
    ndim = 2
    nsteps = 50
    k = Kalman(ndim)    
    mu_init=array([-0.37727, 2.333])
    cov_init=0.1*ones((ndim))
    obs = zeros((ndim, nsteps))
    for t in range(nsteps):
        obs[:, t] = random.normal(mu_init,cov_init)
    for t in range(ndim,nsteps):
        k.update(obs[:,t])
    print k.mu_hat_est
    

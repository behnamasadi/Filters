# Copyright (c) 2010 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''A simple particle filter implementation.'''

# Much of the inspiration for this code came from the small sample at
# http://www.scipy.org/Cookbook/ParticleFilter.
#
# http://homepages.inf.ed.ac.uk/mdewar1/python_tutorial/bootstrap.html enhanced
# the implementation by bridging it a bit with the literature.

import numpy
import logging
from numpy import random as rng


def uniform_displacement(width=1):
    '''Return a callable that displaces the particles in a particle filter.

    This particular model simply displaces the particles in a filter by a
    uniformly distributed random value between -width and width.
    '''
    def displace(particles):
        particles += rng.uniform(-width, width, particles.shape)
    return displace


def euclidean_weights(particles, observation):
    '''Return the inverse Euclidean distance between particles and observation.

    particles: A two-dimensional numpy array of particles that represent the
      current estimate of a distribution. There is one particle per row in the
      array.
    observation: A one-dimensional numpy array containing a new observation.
    '''
    delta = particles - observation
    return (delta * delta).sum(axis=1) ** -0.5


class Filter(object):
    '''A particle filter is a discrete estimate of a probability distribution.

    The filter implementation here maintains a small-ish set of discrete
    particles to represent an estimate of a probability distribution that
    evolves according to some unobserved dynamics.

    Each particle is positioned somewhere in the space being observed, and is
    explicitly weighted with an estimate of the likelihood of that particle.

    The weighted sum of the particles can be used as the expectation of the
    underlying distribution, or---more generally---the weighted particles may
    be used to calculate the expected value of any function under this
    distribution.
    '''

    def __init__(self,
                 dimension,
                 num_particles,
                 displace_particles=None,
                 assess_particles=None,
                 resample_threshold=1.5):
        '''Initialize this filter.

        dimension: The dimensionality of the space from which we draw
          observations.
        num_particles: The number of particles to use. Smaller values result in
          lower accuracy but require less computational effort.
        displace_particles: A callable that takes one argument---the current
          particles---and displaces them according to the system being modeled.
        assess_particles: A callable that takes two arguments---the current
          particles and an observation---and returns an array of weights to be
          normalized and mixed with the current weights.
        resample_threshold: A numeric threshold that determines how often
          particles should be resampled. Values <= 1 will cause resampling with
          every call to update(), and values > 1 will cause increasingly less
          frequent resampling.
        '''
        self._particles = rng.randn(num_particles, dimension)
        self._weights = numpy.ones((num_particles, ), 'd') / num_particles
        self._displace = displace_particles or uniform_displacement()
        self._assess = assess_particles or euclidean_weights
        self._resample_threshold = resample_threshold

    def sample(self):
        '''Return a sample from our filter's distribution.'''
        offset = numpy.searchsorted(self._weights.cumsum(), rng.random())
        return self._particles[offset]

    def expectation(self, f=lambda x: x):
        '''Get the expected value of a function under our filter.

        f: A callable that takes one argument---a particle in our filter---and
          returns some value that implements __iadd__ (int, float, numpy.array,
          etc.).
        '''
        acc = None
        for p, w in self.iterparticles():
            v = f(p) * w
            if acc is None:
                acc = v
            else:
                acc += v
        return acc

    def _resample(self):
        '''Resample the particles in our filter using the current weights.'''
        n = len(self._particles)
        cdf = self._weights.cumsum()
        indices = numpy.searchsorted(cdf, rng.uniform(0, 1, n))
        self._particles = self._particles[indices]
        self._weights = numpy.ones(self._weights.shape, 'd') / n

    def observe(self, observation, confidence=1.0):
        '''Update the filter based on a new observation.

        observation: A numpy array containing a new observation.
        confidence: A value in [0, 1] that indicates the confidence of the new
          observation.
        '''
        self._displace(self._particles)

        weight = self._assess(self._particles, observation)
        weight /= weight.sum()

        # use the confidence to create a mixture of the two weights
        assert 0 <= confidence <= 1
        self._weights = confidence * weight + (1 - confidence) * self._weights

        w = (self._weights ** 2).sum()
        if w * len(self._particles) > self._resample_threshold:
            self._resample()

    def iterparticles(self):
        '''Iterate over the particles and weights in this filter.'''
        for i in xrange(len(self._particles)):
            yield self._particles[i], self._weights[i]


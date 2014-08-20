#http://wiki.esl.fim.uni-passau.de/index.php/Stochastic_Filtering_-_Particle_Filter
from pylab import *
 
class Model():
    def __init__(self,Q,R):
        self.Q = Q
        self.R = R
 
    def pi_0(self):
        # prior on x
        return normal(0,sqrt(10))
 
    def f(self,x,t,new_x=None):
        # dynamic equation
        mean = x/2 + 25* (x/(1+x**2)) + 8 * cos(1.2*t)
        if new_x: 
            return 1/(sqrt(self.Q)*sqrt(2*pi)) * exp((-(new_x-mean)**2)/(2.0*self.Q))
        else: 
            return mean + normal(0,sqrt(self.Q))        
 
    def g(self,x,y=None):
        # observation equation
        mean = (x**2)/20
        if y: 
            return (1/(sqrt(self.R)*sqrt(2*pi))) * exp((-(y-mean)**2)/(2.0*self.R))
        else: 
            return mean + normal(0,sqrt(self.R))    
 
    def sample(self,T):
        # sample a state sequence and its corresponding observations
        # storage
        x = empty(T)
        y = empty(T)
        # initialisation
        x[0] = self.pi_0()
        y[0] = self.g(x[0])
        for t in range(1,T):
            x[t] = self.f(x[t-1],t)
            y[t] = self.g(x[t])
        return x,y
 
 
class Bootstrap():
    def __init__(self,model,N):
        self.model = model
        self.q = model.f
        self.N = N
 
    def resample(self,x,w):
        N = len(w)
        Ninv = 1 / float(N)
        new_x = empty(N)
        c = cumsum(w)
        u = rand()*Ninv
        i = 0
        for j in range(N):
            uj = u + Ninv*j
            while uj > c[i]:
                i += 1
            new_x[j] = x[i]
        new_w = ones(self.N,dtype=float)/self.N
        return new_x, new_w
 
    def filter(self,y):
        # time
        T = len(y)
        t = 0
        xhat = empty(T)
        # distributions
        q, pi_0, g = self.q, self.model.pi_0, self.model.g
        # initial state
        x0 = pi_0()
        # initial particles
        x = [q(x0,t) for i in range(self.N)]
        xhat[0] = 0
        w = ones(self.N,dtype=float)/self.N
        for t in range(1,T):
            # importance sampling
            x = [q(xi,t) for xi in x]
            w = w*[g(xi,y[t]) for xi in x]
            w /= sum(w) # normalise
            # selection
            x,w = self.resample(x,w)
            xhat[t] = sum(x*w)
        return xhat
 
if __name__ == "__main__":        
    model = Model(Q=10,R=1)
    T = 100
    N = 10
    x,y = model.sample(T)
 
    bs = Bootstrap(model,N)
    xhat = bs.filter(y)
    plot(xhat,label='estimated state')
    plot(x,label='true state')
    legend()
    xlabel('t')
    show()
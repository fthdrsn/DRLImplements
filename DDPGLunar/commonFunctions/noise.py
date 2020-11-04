import numpy as np


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self,actionDim, params):
        self.mu = 0.
        self.theta = params.theta
        self.sigma = params.startSigma
        self.max_sigma = params.startSigma
        self.min_sigma = params.endSigma
        self.decay_period = params.decayLenSigma
        self.action_dim = actionDim
        self.Reset()

    def Reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def EvolveState(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def GetNoise(self, t=0):
        ou_state = self.EvolveState()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return ou_state
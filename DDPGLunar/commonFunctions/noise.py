
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,args):

        self.action_dimension = action_dimension
        self.mu = args.mu
        self.theta = args.theta
        self.sigma = args.fixedSigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


# import numpy as np
# # Ornstein-Ulhenbeck Process
# # Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
# class OUNoise(object):
#     def __init__(self,actionDim, params):
#         self.mu = 0.
#         self.theta = params.theta
#         self.sigma = params.startSigma
#         self.max_sigma = params.startSigma
#         self.min_sigma = params.endSigma
#         self.decay_period = params.decayLenSigma
#         self.action_dim = actionDim
#         self.Reset()
#
#     def Reset(self):
#         self.state = np.ones(self.action_dim) * self.mu
#
#     def EvolveState(self):
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
#         self.state = x + dx
#         return self.state
#
#     def GetNoise(self, t=0):
#         ou_state = self.EvolveState()
#         self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
#         return ou_state
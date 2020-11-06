from collections import namedtuple
import numpy as np

### Store Transition and Sample Uniformly

Transition=namedtuple("TRANSITIONS","States Actions Rewards NextStates Dones")

class UniformReplayBuffer():

    def __init__(self,bufferSize,stateShape,actionShape,stateDType,actionDType):

        if not isinstance(stateShape, (tuple, list)):
            raise ValueError("Tuple or list expected for state shape")

        self.bufferIndex=0
        self.size=0
        self.bufferCapacity=bufferSize
        ## self.buffer namedtuple (states array, actions array,reward array,next states array,done array)
        self.buffer=Transition(np.zeros((bufferSize,*stateShape),stateDType),
                               np.zeros((bufferSize, *actionShape), actionDType),
                               np.zeros((bufferSize, 1), np.float32),
                               np.zeros((bufferSize, *stateShape), stateDType),
                               np.zeros((bufferSize, 1), np.long))

    ##Push a transition. Transition is namedtuple(state,action,reward,nextstate,done)
    def push_transition(self,transition):
        self.buffer.States[self.bufferIndex]=transition.State
        self.buffer.Actions[self.bufferIndex] = transition.Action
        self.buffer.Rewards[self.bufferIndex] = transition.Reward
        self.buffer.NextStates[self.bufferIndex] = transition.NextState
        self.buffer.Dones[self.bufferIndex] = transition.Done
        self.bufferIndex+=1
        self.size+=1 if self.size<self.bufferCapacity else 0
        self.bufferIndex=self.bufferIndex%self.bufferCapacity

    ##Sample namedtuple with size batchsize(states,actions,rewards,nextstates,dones)
    def sample_batch(self,batchSize):
        if batchSize<self.size:
            sampleIndexes = np.random.choice(self.size, batchSize, replace=True)
            batch = Transition(self.buffer.States[sampleIndexes],
                                self.buffer.Actions[sampleIndexes],
                                self.buffer.Rewards[sampleIndexes],
                                self.buffer.NextStates[sampleIndexes],
                                self.buffer.Dones[sampleIndexes])
        else:
            batch=None
        return batch






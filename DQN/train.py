import numpy as np
import argparse
from collections import namedtuple
from commonFunctions.UniformBuffer import UniformReplayBuffer
from DQNAgent import DQNAgent
import gym
from torch.utils.tensorboard import SummaryWriter
Transition=namedtuple("Transitions","State Action Reward NextState Done")
def main(args):
    env=gym.make(args.env)
    writer = SummaryWriter(comment="CartPole-v0-DQN")
    totalReward=[]
    actionDim=env.action_space.n
    stateDim=env.observation_space.shape[0]
    hiddenDim=args.hiddenDim
    buffer=UniformReplayBuffer(args.maxCapacity,env.observation_space.shape,np.float32,np.long)

    dqnAgent=DQNAgent(buffer,stateDim,actionDim,hiddenDim,args)
    stepCounter=0
    epsilon=args.epsStart
    for e in range(args.numberOfEpisode):
        state=env.reset()
        episodeReward=0
        done=False

        while not done:
            stepCounter+=1
            action=dqnAgent.GetAction(state,epsilon)
            nextState,reward,done,_=env.step(action)
            buffer.push_transition(Transition(state,action,reward,nextState,done))

            episodeReward+=reward
            if stepCounter>2*args.batchSize:
                dqnAgent.Update(stepCounter)
                epsilon = max(epsilon * args.epsDecay, args.epsStop)

            state = nextState

        totalReward.append(episodeReward)
        meanReward = float(np.mean(totalReward[-100:]))
        writer.add_scalar("episodeReward", episodeReward, stepCounter)
        writer.add_scalar("meanReward", meanReward, stepCounter)
        writer.add_scalar("epsilon", epsilon, stepCounter)
        writer.add_scalar("episodes", e,stepCounter)
        print("Eps:{} Steps:{} Mean Reward: {} Episode Reward: {}  Epsilon: {}".format(e, stepCounter, meanReward, episodeReward, epsilon))


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DQN Algo-FD')
    parser.add_argument("--env", default="CartPole-v0", help="Training environment")
    parser.add_argument("--maxCapacity", type=int,default=1000, help="Maximum buffer capacity")
    parser.add_argument("--batchSize", type=int, default=16, help="Batch size for training")
    parser.add_argument("--maxStepCount", type=int, default=1000, help="Maximum step count in an episode")
    parser.add_argument("--numberOfEpisode", type=int, default=1000, help="Episode count ")
    parser.add_argument("--hiddenDim", type=int, default=48, help="Episode count ")
    parser.add_argument("--learningRate",  default=1e-3, help="Learning rate  ")
    parser.add_argument("--gamma", default=0.99, help="Discount factor for future values  ")
    parser.add_argument("--device", default="cuda", help=" ")
    parser.add_argument("--targetUpdatePeriod", default=10, help="Update target at this period")

    parser.add_argument("--epsStart", default=1, help="Start value of epsilon  ")
    parser.add_argument("--epsDecay", default=0.995, help="Decay rate of epsilon  ")
    parser.add_argument("--epsStop", default=0.02, help=" Final value of epsilon")
    args = parser.parse_args()
    main(args)













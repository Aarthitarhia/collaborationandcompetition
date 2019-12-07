## Introduction

For this project, you will work with the Tennis environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


ALGORITHM:
    The algorithm used for the agent is DDPG.
    
    model.py file contains neural network definitions for actor and critic models
    
    ddpg_agent.py file implements Agent and ReplayBuffer classes
    
    checkpoint_actor.pth file contains model checkpoint for actor neural network
    
    checkpoint_critic.pth file contains model checkpoint for critic neural network
    
HYPERPARAMETERS:

    BUFFER_SIZE = int(1e5) (replay buffer size)
    BATCH_SIZE = 128 (minibatch size)
    GAMMA = 0.99 (discount factor)
    TAU = 1e-3 (for soft update of target parameters)
    LR_ACTOR = 1e-4 (learning rate of the actor)
    LR_CRITIC = 1e-4 (learning rate of the critic)
    WEIGHT_DECAY = 0 (L2 weight decay)
    UPDATE_EVERY = 1 (how many steps to take before updating target networks)
    
ACTOR NEURAL NETWORK:

    The Actor neural network consists of three fully connected (FC) layers.
    The input has 24 channels (each agent observes a state with length: 24)
    The output channels of the first FC layer is: 256
    The input and output channels of the second FC layer are: 256, 128
    The input channels of the third FC layer are: 128
    The output has 2 channels (actions: movement toward (or away from) the net, and jumping)

CRITIC NEURAL NETWORK:

    The Critic neural network consists of three fully connected (FC) layers.
    The input has 24 channels (each agent observes a state with length: 24)
    The output channels of the first FC layer is: 256
    The input channels of the second FC layer is: 256 + 2 (actions)
    The output channels of the second FC layer is: 128
    The output has 1 channel.
    
PLOT OF REWARDS:

    The plot shows the reward received for each episode. In our case, environment was solved in 2800 episodes.
    
    
    




<img src = "plot.png"/>













## Future Work Ideas

I would implement experienced replay buffer.This technique prioritizes the experiences and chooses the best experience for further training when sampling from the buffer. This is known to reduce the training time and make the training more efficient.

I would probably try tuning with different hyperparameters and then visualise them to view a pattern as to see how the trend changes.


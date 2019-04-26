[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

This project is part of Udacity's Nanodegree on Deep Reinforcement Learning (https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).


# Project 3: Collaboration and Competition

### Introduction

For this project, I will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment which uses the Unity Machine Learning Agents Toolkit (https://github.com/Unity-Technologies/ml-agents), an open-source Unity plugin.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

Goal of the exercise is to train a multi-agent Reinforcement Learning Agent to keep the ball in play. Specifically, I have chosen an Agent based on Multi-Agent Deep Deterministic Policy Gradients (MADDPG).


The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the Agent must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received are added up (without discounting), to get a score for each agent. This yields two (potentially different) scores. The maximum of these two scores is then taken.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


The ipython notebook `Tennis.ipynb` contains the successful MADDPG Agent for performing this
task. The MADDPG Agent is implemented in `maddpg_agent.py` and the neural network in `model.py`. For a discussion of the findings and assessment of the Agent see the `Report.pdf`. The model weights of the Actor and Critic are stored in `checkpoint_Multi-Agent DDPG_agent_0_actor.pth`,  `checkpoint_Multi-Agent DDPG_agent_0_critic.pth` and `checkpoint_Multi-Agent DDPG_agent_1_actor.pth`,  `checkpoint_Multi-Agent DDPG_agent_1_critic.pth`, for agents 0 and 1, respectively.



# Prerequisites

Running this notebook requires Python 3.5 (or higher), the Tennis environment (see below) and the following Python libraries:

- NumPy
- PyTorch
- Unity Machine Learning Agents Toolkit
- Matplotlib



# To get started -

Below are the instructions on how to get started on this project as given in the original repository.
The original repository can be found here: https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet which this README.md file is based on.



### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `Udacity-Deep-RL-Project-3-Collab-Compet/` folder, and unzip (or decompress) the file.

3. Run `Tennis.ipynb` for the successful MADDPG Agent performing this task.



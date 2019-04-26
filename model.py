# based on my solution of assignment 2 and Udacity's implementation of ddpg-pendulum/model.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, use_BatchNorm=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            use_BatchNorm (bool): if 'True', then use batch normalization
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_BatchNorm = use_BatchNorm
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        if self.use_BatchNorm:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        #Apply batch normalization:
        # https://pytorch.org/docs/0.3.1/nn.html        
        # https://stackoverflow.com/questions/47197885/how-to-do-fully-connected-batch-norm-in-pytorch        
        
        # Reshape 'state' into a new tensor with a dimension of size one inserted at the specified position.       
        #  https://pytorch.org/docs/stable/torch.html
        if self.use_BatchNorm and state.dim()==1:
            state = torch.unsqueeze(state, 0)
        
        x = F.relu(self.fc1(state))
        
        if self.use_BatchNorm:
            x = self.bn1(x)
            
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""
    
    # There are two possibilities for inputting the actions:
    # - in the second layer (as previously done in Assignment 2)
    # - in the first layer (as suggested on Udacity's Knowledge Hub)
    # Here I use the latter approach, i.e. in the first layer.
    
    #%https://knowledge.udacity.com/questions/32892
    #Actor input layer size = state_size
    #Critic input layer size = num_agents*(state_size+action_size)
    

    def __init__(self, all_state_size, all_action_size, seed, fcs1_units=256, fc2_units=256, use_BatchNorm=True):
        """Initialize parameters and build model.
        Params
        ======
            all_state_size (int): Dimension of num_agents * state_size
            all_action_size (int): Dimension of num_agents * action_size
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            use_BatchNorm (bool): if 'True', then use batch normalization
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_BatchNorm = use_BatchNorm

#        self.fcs1 = nn.Linear(all_state_size, fcs1_units)
#        self.fc2 = nn.Linear(fcs1_units+all_action_size, fc2_units)

        self.fcs1 = nn.Linear(all_state_size+all_action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)

        self.fc3 = nn.Linear(fc2_units, 1)
        
        if self.use_BatchNorm:
            self.bn1 = nn.BatchNorm1d(fcs1_units)
            
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (states, actions) pairs -> Q-values."""
#        if self.use_BatchNorm and states.dim()==1:
#            states = torch.unsqueeze(states, 0)
#        xs = F.relu(self.fcs1(states))
#        if self.use_BatchNorm:
#            xs = self.bn1(xs)
#        x = torch.cat((xs, actions), dim=1)
#        x = F.relu(self.fc2(x))
#        return self.fc3(x)
        
        
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fcs1(xs))
        if self.use_BatchNorm:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)



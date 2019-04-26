# based on my solution of assignment 2 and Udacity's implementation of ddpg-pendulum/ddpg_agent.py


import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from model import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, 
                 num_agents,
                 state_size,
                 action_size, 
                 random_seed = 2, 
                 buffer_size = int(1e5),
                 batch_size = 256,
                 gamma = 0.99,
                 update_every = 1, 
                 update_m_times = 1,
                 add_noise = False,
                 noise_factor = 1.0,
                 noise_reduction = 0.95):
        """Initialize an Agent object.
        
        Params
        ======
            num_agents (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            update_every (int): how often to update the network
            update_m_times (int): how many times to update the network consecutively
            add_noise (bool): if True, noise is being added
            noise_factor (float): multiplicative noise factor, 1 means neutral 
            noise_reduction (float): multiplicative noise reduction, 1 means no decay
        """
        
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.update_m_times = update_m_times
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        self.noise_reduction = noise_reduction

        # Most of the code generalizes to more than 2 agents,
        # but certain parts of DDPG_Agent.learn() assume 2 agents only.
        assert( num_agents==2 )

        # Create 'num_agents' DDPG Agents
        self.agents = [DDPG_Agent(i, self.num_agents, self.state_size, self.action_size, random_seed) for i in range(num_agents)]
        
        # Replay buffer is shared amongst the different agents,
        #    and therefore needs to be set up on the level of MADDPG
        self.memory = ReplayBuffer(buffer_size, self.batch_size, random_seed)

        # update_every and t_step based on dqn_agent.py of the lunar lander solution
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0


    # Stepping takes place on the level of MADDPG
    # Adds to the replay buffer and samples from it => calls learn
    # This function is being called with all states, actions, rewards, next states and dones of ALL / BOTH agents - 
    #    Re-shape / flatten before adding to Replay Buffer
    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # all_states: (2, 24)
        # all_actions: (1, 4)
        # all_rewards: 2
        # all_next_states: (2, 24)
        # all_dones: 2

        # Flatten all_states and all_next_states: (2, 24) -> (1, 48) before adding to Replay Buffer
        all_states = all_states.reshape(1, -1)
        all_actions = all_actions.reshape(1, -1)
        all_next_states = all_next_states.reshape(1, -1)
                
        # Save experience
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        self.t_step = self.t_step + 1

        # Learn every update_every time steps.
        if self.t_step % self.update_every == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                # After every N steps update M times continuously                
                # as discussed in: https://knowledge.udacity.com/questions/29983
                for i in range(self.update_m_times):
#                    experiences = self.memory.sample()  
                    experiences = [ self.memory.sample() for k in range(self.num_agents) ]
                    # Provide a list of experience samples, one per each agent
                    self.learn(experiences, self.gamma)




    # The MADDPG agent calls on the individual agents to act 
    def act(self, all_states):
        """Iterate through individual agents and states.
           Call the individual agent's act function, 
           calculate the action, then append it back to the list of all_actions."""
           
        all_actions = [ agent.act(state, self.noise_factor, self.add_noise) for agent, state in zip( self.agents, all_states ) ]
        self.noise_factor = self.noise_factor * self.noise_reduction
        
        return np.array(all_actions).reshape(1, -1)   # (2, 2) -> (1, 4)




    # The MADDPG agent calls on the individual agents to reset
    def reset(self):
        for agent in self.agents:
            agent.reset()



        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        
           Calls on the individual agents to learn and provides input:
           Determining all_actions and all_next_actions needs to be done here on the top level
           where we have straightforward access to actor_local and actor_target of both agents.
           
           As pointed out in:
               Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,
               Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel and Igor Mordatch
           the main motivation behind this approach is that, if one knows the actions taken by all agents, 
           then the environment is stationary even if the policies change.
        """
        
        all_actions = []
        all_next_actions = []
        
        for i, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = experiences[i]
            agent_id = i

            # Index range in state and next_state for the agent under consideration:
            agent_index_range  = [ index for index in range(agent_id * self.state_size, (agent_id + 1) * self.state_size) ]
            agent_index_range  = torch.tensor( agent_index_range ).to(device)    
            
            # states (stacked / flattened) -> state -> action -> all_actions
            state = states.index_select(1, agent_index_range)
            action = agent.actor_local(state)   # get action from actor_local.forward(state)
            all_actions.append(action)   # used for pred_action, for updating actor in DDPG_Agent.learn()

            # next states (stacked / flattened) -> next state -> next action -> all next actions
            next_state = next_states.index_select(1, agent_index_range)
            next_action = agent.actor_target(next_state)   # get action from actor_target.forward(next_state)
            all_next_actions.append(next_action)   # for updating the critic
            
        # Flatten it: Bring into shape required subsequently in DDPG_Agent.learn()
        #    Concatenate all_next_actions list consisting of sequence of next_action 
        all_next_actions = torch.cat(all_next_actions, dim=1).to(device)
                       
        # MADDPG agent calls on its individual agents to learn given a sample of the replay buffer -
        #    Note that each sample contains the (flattened) states, actions, etc. of both agents
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_actions, all_next_actions)



        
    def save_model_weights(self, case_name):
        # save model weights 
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  "checkpoint_" + case_name + "_agent_" + str(i) + "_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_" + case_name + "_agent_" + str(i) + "_critic.pth")






# =============================================================================
#    Below the DDPG Agent - 
#       based on my solution of assignment 2 and Udacity's implementation of ddpg-pendulum/ddpg_agent.py
#       with adjustments for multi-agent framework
# =============================================================================

class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    
    def __init__(self, 
                 agent_id, 
                 num_agents,
                 state_size,
                 action_size, 
                 random_seed, 
                 lr_actor = 5e-4,
                 lr_critic = 5e-4,
                 tau = 1e-3, 
                 weight_decay = 0.0,
                 use_clipping=False
                 ):
        """Initialize an Agent object.
        
        Params
        ======
            agent_id (int): id of agent
            num_agents (int): number of agents
        
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            
            lr_actor (float): learning rate of the actor 
            lr_critic (float): learning rate of the critic

            tau (float): for soft update of target parameters
            weight_decay (float): L2 weight decay
            
            use_clipping (bool): if 'True', then use clipping
        """
        self.agent_id = agent_id
        self.tau = tau
        self.use_clipping = use_clipping
        
        # Most of the code generalizes to more than 2 agents,
        # but certain parts of DDPG_Agent.learn() assume 2 agents only.
        assert( num_agents==2 )
        
        #%https://knowledge.udacity.com/questions/32892
        #   Actor input layer size = state_size
        #   Critic input layer size = num_agents*(state_size+action_size)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed=0).to(device)
        self.actor_target = Actor(state_size, action_size, seed=0).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(num_agents * state_size, num_agents * action_size, seed=0).to(device)
        self.critic_target = Critic(num_agents * state_size, num_agents * action_size, seed=0).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)




    def act(self, state, noise_factor=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * noise_factor
        return np.clip(action, -1, 1)




    def reset(self):
        self.noise.reset()




    def learn(self, agent_id, experiences, gamma, all_actions, actions_next):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        agent_id = torch.tensor([agent_id]).to(device)

        #actions_next = self.actor_target(next_states)
        #
        # The critic uses 'actions_next' from ALL individual DDPG agents.
        #    Since actions_next is being calculated using each agent's actor_target,
        #    this needs to be done on the level of MADDPG.learn()
        #    and is then passed into this function as an argument.
        
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Select rewards and dones for the 'agent_id' under consideration
        rewards = rewards.index_select(1, agent_id)
        dones = dones.index_select(1, agent_id)
            
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets = Q_targets.detach()
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Use gradient clipping when training the critic network
        # as discussed in Udacity's Benchmark Implementation (Assignment 2, Subsection 6.)        
        if self.use_clipping:
            torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        
        #actions_pred = self.actor_local(states)
        #
        # The actor uses 'all_actions' from ALL individual DDPG agents.
        # Since all_actions is being calculated using each agent's actor_local,
        # this needs to be done on the level of MADDPG.learn()
        #    and is then passed into this function as argument 'all_actions'.
        
        # Caution: Below implementation specific to two agents !!!
        
        my_actions = all_actions[self.agent_id]
        other_actions = all_actions[(self.agent_id+1)%2].detach()

        # Arrange according to ID of the DDPG agent
        if self.agent_id == 0:
            actions_pred = torch.cat((my_actions, other_actions), dim=1).to(device)
        else:
            actions_pred = torch.cat((other_actions, my_actions), dim=1).to(device)
        
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     




    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)






class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        
        x = self.state
#        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])

        #maddpg-lab/OUNoise.py samples from standard normal distribution
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))

        self.state = x + dx
        return self.state






class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)




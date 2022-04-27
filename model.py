from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class PGNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, layer=2):
        """
        Initialize the parameter for the policy network
        """
        super(PGNetwork, self).__init__()

        layers = []
        if layer == 0:
            layers.append(nn.Linear(in_dim, 1, bias=False))
            layers.append(nn.Sigmoid())
        else:
            # Input layer
            layers.append(nn.Linear(in_dim, hidden_dim))

            # Intermediary layers
            for _ in range(layer - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            # Final layer
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            layers.append(nn.Softmax(dim=-1))

        self.n_layers = layer
        self.seq = nn.Sequential(*layers)

    def forward(self, observation):
        """
        This function takes in a batch of observations and a batch of actions, and 
        computes a probability distribution (Categorical) over all (discrete) actions
        
        observation: shape (batch_size, observation_size) torch Tensor
        
        return: a categorical distribution over all possible actions. You may find torch.distributions.Categorical useful
        """
        dist = self.seq(observation)
        if self.n_layers > 0:
            return torch.distributions.Categorical(dist)
        else:
            return torch.distributions.bernoulli.Bernoulli(dist.squeeze(-1))


class ValueNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        """
        Initialize the parameter for the value function
        """
        super(ValueNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        """
        This function takes in a batch of observations, and 
        computes the corresponding batch of values V(s)
        
        observation: shape (batch_size, observation_size) torch Tensor
        
        return: shape (batch_size,) values, i.e. V(observation)
        """
        return self.seq(observation).squeeze(-1)


class Model(pl.LightningModule):

    def __init__(self, temporal: bool = False, baseline: bool = False):
        self.temporal = temporal
        self.baseline = baseline

    def rollout(self, vmodel=None, device=None, MAX_T=10000):
        actions = torch.zeros(MAX_T, device=device, dtype=torch.int)
        rewards = torch.zeros(MAX_T, device=device)
        log_probs = torch.zeros(MAX_T, device=device)
        values = torch.zeros(MAX_T, device=device)
        obs = env.reset()
        T = 0
        ep_reward = 0
        while T < MAX_T:
            x = torch.tensor(obs, device=device, dtype=torch.float).unsqueeze(dim=0)
            dist = self(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if vmodel:
                value = vmodel(x)
                values[T] = value

            next_obs, reward, done, _ = env.step(int(action.item()))
            ep_reward += reward
            rewards[T] = reward
            actions[T] = action
            log_probs[T] = log_prob
            obs = next_obs

            T += 1

            if done:
                break
        return actions[:T], rewards[:T], log_probs[:T], values[:T], ep_reward

    def discounted_returns(self, rewards, gamma):
        returns = torch.zeros_like(rewards)
        #### Your code here
        for i in range(len(rewards)):
            returns[i] = sum(rewards[j] * gamma ** j for j in range(i, len(rewards)))

        return returns

    def training_step(self, *args, **kwargs):
        actions, rewards, log_probs, values, ep_reward = self.rollout()

        # compute policy losses
        policy_loss = []
        returns = self.discounted_returns(rewards, gamma, device)
        eps = np.finfo(np.float32).eps.item()

        if values != None:
            #### Your code here: compute value loss by fitting values
            #### to the returns with F.smooth_l1_loss
            value_loss = F.smooth_l1_loss(values, returns)

        if values != None:  # use the value function as the baseline
            returns = (
                returns
                - torch.tensor([self.gamma ** i for i in range(len(values))])
                * values.detach()
            )  # this is the "advantage"

        #### Your code here: compute policy loss based on different objectives
        if self.temporal:
            policy_loss = -(log_probs * returns).sum()
        else:
            policy_loss = -log_probs.sum() * returns[0]

        if values != None:
            loss = policy_loss + value_loss
        else:
            loss = policy_loss

        return loss


def train():
    #### Train the model
    model = PGNetwork(env.observation_space.shape[0], env.action_space.n, 64, layer).to(device)

    running_reward = 0
    history_reward = [] # store moving average of empirical rewards

    for step in trange(MAX_EPISODES):

        
        if step % LOG_INTERVAL == 0:
            print('Episode {}\tLast reward: {:.2f} \tAverage reward: {:.2f}'.format(
                  step, ep_reward, running_reward))

    return history_reward
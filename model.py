import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from environment import FPLEnvironment
import os
import time
import numpy as np

SAVE_DIR = "checkpoints"
TEMPORAL = True
USE_VALUE_NETWORK = False
MAX_EPISODES = 1000
LOG_INTERVAL = 1
STEP_SAVE = 100
LR = 1e-4
GAMMA = 0.999
MAX_T = 5
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        num_layers: int = 2,
        dropout: float = 0,
    ):
        """
        Initialize the parameter for the policy network
        """
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # TODO: check this
        )

    def forward(self, data):
        # data is of shape 1 x N x D
        embedding = self.embedding(data)
        # encodings is of shape 1 x N x H
        encodings = self.transformer(embedding)
        # output is of shape 1 x N
        output = self.fc(encodings).squeeze(-1)
        return output


class ValueNetwork(nn.Module):
    def __init__(
        self, in_dim, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0,
    ):
        """
        Initialize the parameter for the value function
        """
        super(ValueNetwork, self).__init__()
        module_list = []
        for _ in range(num_layers):
            module_list.extend(
                [nn.Linear(in_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()]
            )
            in_dim = hidden_dim
        module_list.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*module_list)

    def forward(self, observation):
        """
        This function takes in a batch of observations, and 
        computes the corresponding batch of values V(s)

        observation: shape (batch_size, observation_size) torch Tensor

        return: shape (batch_size,) values, i.e. V(observation)
        """
        return self.model(observation).squeeze(-1)


def save_checkpoint(net, optimizer, save_dir="checkpoints"):
    step = time.strftime("%m-%d-%Y-%H:%M:%S", time.localtime())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(step) + ".pth")

    torch.save(dict(net=net.state_dict(), optimizer=optimizer.state_dict()), save_path)
    print("Saved checkpoint %s" % save_path)


def add_noise(priority):
    priority_noise = {}
    for k, v in priority.items():
        if np.random.rand() < 0.05:
            new_v = np.random.uniform(0, 1)
        else:
            new_v = v
        priority_noise[k] = new_v
    return priority_noise


def rollout(env, model, vmodel=None, device=DEVICE):
    env.reset()
    rewards = torch.zeros(MAX_T, device=device)
    priorities = torch.zeros(MAX_T, len(env.players), device=device)
    masks = torch.zeros(MAX_T, len(env.players), device=device)
    values = torch.zeros(MAX_T, device=device)

    ep_reward = 0
    for T in tqdm(range(MAX_T), total=MAX_T):
        obs = env.features(device)
        x = obs.unsqueeze(dim=0)
        priority = model(x)[0]
        if vmodel:
            value = vmodel(x)
            values[T] = value

        priority_dict = {p: priority[i].cpu().item() for i, p in enumerate(env.players)}
        if np.random.rand() < 0.1:
            priority_dict = add_noise(priority_dict)

        action = env.sample_action(priority_dict)
        reward, mask = env.update(action)

        ep_reward += reward
        rewards[T] = reward
        priorities[T] = priority
        masks[T] = mask

    # return priorities[:T], rewards[:T], values[:T], ep_reward, masks[:T]
    return priorities, rewards, values, ep_reward, masks


def discounted_returns(rewards, gamma):
    """Returns sum of discounted future rewards for each time step (temporal structure)"""
    returns = torch.zeros_like(rewards)
    for i in range(len(rewards)):
        returns[i] = sum(rewards[j] * gamma ** j for j in range(i, len(rewards)))

    return returns


def update_parameters(optimizer, priorities, rewards, values, mask):

    # compute policy losses
    policy_loss = []
    returns = discounted_returns(rewards, GAMMA)

    # compute value loss by fitting values to observed returns
    if values is not None:
        value_loss = F.smooth_l1_loss(values, returns)

    # use the value function as the baseline
    if values is not None:
        returns = (
            returns
            - torch.tensor([GAMMA ** i for i in range(len(values))]).to(returns)
            * values.detach()
        )  # this is the "advantage"

    # compute policy loss based on different objectives
    if TEMPORAL:
        policy_loss = -((priorities * mask).mean(dim=1) * returns).sum()
    else:
        policy_loss = -(priorities * mask).mean(dim=1).sum() * returns[0]

    if values is not None:
        loss = policy_loss + value_loss
    else:
        loss = policy_loss

    # parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Loss: ", loss)


def train():
    env = FPLEnvironment()
    model = PolicyNetwork(env.feature_dim).to(DEVICE)

    if USE_VALUE_NETWORK:
        vmodel = ValueNetwork(env.feature_dim).to(DEVICE)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(vmodel.parameters()), lr=LR
        )
    else:
        vmodel = None
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    running_reward = 0
    history_reward = []  # store moving average of empirical rewards

    for step in tqdm(range(MAX_EPISODES)):
        priorities, rewards, values, ep_reward, mask = rollout(
            env, model, vmodel=vmodel, device=DEVICE
        )
        update_parameters(optimizer, priorities, rewards, values, mask)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        history_reward.append(running_reward)

        if step % LOG_INTERVAL == 0:
            print(
                "Episode {}\tLast reward: {:.2f} \tAverage reward: {:.2f}".format(
                    step, ep_reward, running_reward
                )
            )

        if step % STEP_SAVE == 0:
            pass
            # Saves model checkpoint
            # save_checkpoint(model, optimizer, SAVE_DIR)

    # save_checkpoint(model, optimizer, SAVE_DIR)
    return history_reward


if __name__ == "__main__":
    train()

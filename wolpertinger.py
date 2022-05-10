import torch
import torch.nn as nn
from tqdm import tqdm
from collections import deque
from environment import FPLEnvironment
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import scipy.stats


SAVE_DIR = "checkpoints"
MAX_EPISODES = 1000
LOG_INTERVAL = 1
STEP_SAVE = 100
Q_LR = 1e-4
P_LR = 1e-5
GAMMA = 0.999
MAX_T = 38
PROB_OF_ADD_NOISE_FINE = 0.05
BUFFER_SIZE = 1000
BATCH_SIZE = 32
PROB_OF_ADD_NOISE = 0.3
# NOISE_ANNEAL = 100000
WARMUP = 100
TAU = 1e-3
EVAL_STEP = 20


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def add_noise(priority):
    priority_noise = {}
    for k, v in priority.items():
        if np.random.rand() < PROB_OF_ADD_NOISE_FINE:
            new_v = np.random.uniform(0, 1)
        else:
            new_v = v
        priority_noise[k] = new_v
    return priority_noise


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
        # layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
        # self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            # nn.ReLU()
            nn.Sigmoid(),  # TODO: check this
        )

    def forward(self, data):
        # data is of shape 1 x N x D
        embedding = self.embedding(data)
        # encodings is of shape 1 x N x H
        # encodings = self.transformer(embedding)
        # output is of shape 1 x N
        output = self.fc(embedding).squeeze(-1)
        # output = torch.softmax(output, dim=1)
        return output


class Critic(nn.Module):
    def __init__(
        self,
        feature_dim,
        player_dim,
        hidden_dim=256,
        dropout=0,
        nhead=4,
        dim_feedforward=1024,
        num_layers=2,
    ):
        super(Critic, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # hidden_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(player_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            # nn.ReLU(),
            nn.Sigmoid(),  # TODO: check this
        )

        # layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
        # self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1),
        #     # nn.ReLU(),
        #     nn.Sigmoid(),  # TODO: check this
        # )

    def forward(self, x, a):
        data = torch.cat([x, a.unsqueeze(-1)], dim=-1)
        # data is of shape 1 x N x D + 1
        embedding = self.embedding(data).squeeze(-1)
        # encodings is of shape 1 x N x H
        # encodings = self.transformer(embedding)
        # Average pooling
        # encodings = encodings.mean(dim=1)
        # output is of shape 1 x N
        output = self.fc(embedding).squeeze(-1)
        # output = torch.softmax(output, dim=1)
        return output


class WOLPAgent(object):
    def __init__(self, feature_dim, num_players, q_lr=Q_LR, a_lr=P_LR, tau=TAU):

        self.actor = PolicyNetwork(feature_dim)
        self.actor_target = PolicyNetwork(feature_dim)
        self.actor_target.eval()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=q_lr)

        self.critic = Critic(feature_dim, num_players)
        self.critic_target = Critic(feature_dim, num_players)
        self.critic_target.eval()
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=a_lr)

        hard_update(
            self.actor_target, self.actor
        )  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Hyper-parameters
        self.tau = tau

        # replay buffer
        self.memory = deque()

        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            self.cuda()

    def update_policy(self):
        (
            state,
            raw_action,
            action,
            reward,
            next_state,
            next_action,
            done,
        ) = self.sample_batch()

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(
                next_state.detach(), next_action.detach()
            )

        target_q_batch = reward + GAMMA * ((1 - done.float()) * next_q_values)

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic(state, action)
        value_loss = self.criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        policy_loss = -self.critic(state, raw_action.detach())
        policy_loss = policy_loss.mean()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss, policy_loss

    def sample_batch(self):
        indicies = np.random.choice(len(self.memory), BATCH_SIZE)
        state = torch.stack([self.memory[i]["state"] for i in indicies])
        raw_action = torch.stack([self.memory[i]["raw_action"] for i in indicies])
        action = torch.stack([self.memory[i]["action"] for i in indicies])
        reward = torch.tensor([self.memory[i]["reward"] for i in indicies])
        next_state = torch.stack([self.memory[i]["next_state"] for i in indicies])
        next_action = torch.stack([self.memory[i]["next_action"] for i in indicies])
        done = torch.tensor([self.memory[i]["done"] for i in indicies])

        return (
            state.to(DEVICE),
            raw_action.to(DEVICE),
            action.to(DEVICE),
            reward.to(DEVICE),
            next_state.to(DEVICE),
            next_action.to(DEVICE),
            done.to(DEVICE),
        )

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def load_weights(self, output):
        if output is None:
            return

        self.actor.load_state_dict(torch.load("{}/actor.pkl".format(output)))
        self.critic.load_state_dict(torch.load("{}/critic.pkl".format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))

    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)


def learn(env, model, episode, device=DEVICE):
    env.reset()
    rewards = torch.zeros(MAX_T, device=device)
    priorities = torch.zeros(MAX_T, len(env.players), device=device)
    masks = torch.zeros(MAX_T, len(env.players), device=device)
    values = torch.zeros(MAX_T, device=device)
    dist = scipy.stats.norm(200, 100)

    ep_reward = 0
    for T in tqdm(range(MAX_T), total=MAX_T):
        if (episode * MAX_T) + T < WARMUP:
            priority = torch.rand(len(env.players), device=device, requires_grad=False)
            obs = env.features(device)
            obs.requires_grad = False
        else:
            obs = env.features(device)
            x = obs.unsqueeze(dim=0)
            priority = model.actor(x)[0]

        priority_dict = {p: priority[i].cpu().item() for i, p in enumerate(env.players)}
        # if ((episode * MAX_T) + T) < NOISE_ANNEAL:
        #     prob = PROB_OF_ADD_NOISE * (NOISE_ANNEAL - ((episode * MAX_T) + T)) / NOISE_ANNEAL
        #     if np.random.rand() < prob:
        #         priority_dict = add_noise(priority_dict)
        if (
            ((episode % EVAL_STEP) != 0) or (episode == 0)
        ) and np.random.rand() < PROB_OF_ADD_NOISE:
            priority_dict = add_noise(priority_dict)

        action = env.sample_action(priority_dict)
        binary_action = torch.tensor(
            [float(p in action.players_all) for p in env.players], device=device
        )
        reward, mask = env.update(action)

        done = T == MAX_T - 1
        if done:
            next_priority = torch.zeros(
                len(env.players), device=device, requires_grad=False
            )
            next_state = torch.zeros_like(obs, device=device, requires_grad=False)
            next_action = torch.zeros_like(
                binary_action, device=device, requires_grad=False
            )
        else:
            with torch.no_grad():
                if (episode * MAX_T) + T < WARMUP:
                    next_priority = torch.rand(
                        len(env.players), device=device, requires_grad=False
                    )
                    next_state = env.features(device)
                else:
                    next_state = env.features(device)
                    next_priority = model.actor_target(next_state.unsqueeze(0))[0]

                next_priority_dict = {
                    p: next_priority[i].cpu().item() for i, p in enumerate(env.players)
                }
                next_action = env.sample_action(next_priority_dict)
                next_binary_action = torch.tensor(
                    [float(p in next_action.players_all) for p in env.players],
                    device=device,
                )

        model.memory.append(
            {
                "state": obs,
                "next_state": next_state,
                "raw_action": priority,
                "action": binary_action,
                "reward": float(dist.cdf(reward)),
                "next_action": next_binary_action,
                "done": T == done,
            }
        )

        while len(model.memory) > BUFFER_SIZE:
            model.memory.popleft()

        value_loss, policy_loss = torch.tensor(0), torch.tensor(0)
        if (((episode * MAX_T) + T) >= WARMUP) and ((episode % EVAL_STEP) != 0):
            value_loss, policy_loss = model.update_policy()

        ep_reward += reward
        rewards[T] = reward
        priorities[T] = priority
        masks[T] = mask

    return priorities, rewards, values, ep_reward, masks, value_loss, policy_loss


def train():
    writer = SummaryWriter()
    env = FPLEnvironment()
    model = WOLPAgent(env.feature_dim, len(env.players))
    if torch.cuda.is_available():
        model.cuda()

    running_reward = 0
    history_reward = []  # store moving average of empirical rewards

    for step in tqdm(range(MAX_EPISODES)):
        priorities, rewards, values, ep_reward, mask, value_loss, policy_loss = learn(
            env, model, step, device=DEVICE
        )
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        history_reward.append(running_reward)

        if step % EVAL_STEP == 0:
            print("EVAL")

        if step % LOG_INTERVAL == 0:

            print(
                "Episode {}\tLast reward: {:.2f} \tAverage reward: {:.2f}".format(
                    step, ep_reward, running_reward
                )
            )
            writer.add_scalar("Reward", ep_reward, step)
            writer.add_scalar("Average", running_reward, step)
            writer.add_scalar("value_loss", value_loss.cpu().item(), step)
            writer.add_scalar("policy_loss", policy_loss.cpu().item(), step)

        if step % STEP_SAVE == 0:
            pass
            # Saves model checkpoint
            # save_checkpoint(model, optimizer, SAVE_DIR)

    # save_checkpoint(model, optimizer, SAVE_DIR)
    return history_reward


if __name__ == "__main__":
    train()

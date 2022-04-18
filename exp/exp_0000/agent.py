import random
import numpy as np
import torch
from torch import nn
from model import MarioNet
from collections import deque


class Mario:
    def __init__(self, cfg, action_dim, save_dir):
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.state_dim = (cfg.state_channel, cfg.state_height, cfg.state_width)

        self.use_cuda = torch.cuda.is_available()

        self.net = MarioNet(cfg, self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

        self.exploration_rate = cfg.exploration_rate
        self.exploration_rate_decay = cfg.exploration_rate_decay
        self.exploration_rate_min = cfg.exploration_rate_min
        self.curr_step = 0

        self.save_every = cfg.save_interval

        self.memory = deque(maxlen=cfg.memory_length)
        self.batch_size = cfg.batch_size

        self.gamma = cfg.gamma
        if cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        if cfg.loss_fn == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss()

        self.burnin = cfg.burnin
        self.learn_every = cfg.learn_every
        self.sync_every = cfg.sync_every

    def action(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(
                action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])
        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (self.save_dir / f'mario_net.pth')
        torch.save(
            dict(model=self.net.state_dict(),
                 exploration_rate=self.exploration_rate), save_path
        )

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

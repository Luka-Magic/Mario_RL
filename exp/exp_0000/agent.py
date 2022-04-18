import random
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from model import MarioNet
from collections import deque
import pandas as pd
import pickle
import time
import datetime
import matplotlib.pyplot as plt
import wandb


class Mario:
    def __init__(self, cfg, action_dim, save_dir):
        # input
        self.action_dim = action_dim
        self.save_dir = save_dir

        # init
        self.init_learning = cfg.init_learning
        self.use_cuda = torch.cuda.is_available()
        self.curr_step = 0
        self.restart_steps = 0
        self.restart_episodes = 0
        self.save_every = cfg.save_interval

        # model
        self.state_dim = (cfg.state_channel, cfg.state_height, cfg.state_width)
        self.net = MarioNet(cfg, self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

        # exploration
        self.exploration_rate = cfg.exploration_rate
        self.exploration_rate_decay = cfg.exploration_rate_decay
        self.exploration_rate_min = cfg.exploration_rate_min

        # memory
        self.memory = deque(maxlen=cfg.memory_length)
        self.batch_size = cfg.batch_size

        # learn
        self.gamma = cfg.gamma
        self.scaler = GradScaler()
        if cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        if cfg.loss_fn == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss()
        self.burnin = cfg.burnin
        self.learn_every = cfg.learn_every
        self.sync_every = cfg.sync_every

        # log variable

        # self.ep_rewards = []
        # self.ep_lengths = []
        # self.ep_avg_losses = []
        # self.ep_avg_qs = []

        # self.log_df = pd.DataFrame(
        #     columns=['step', 'episode', 'epsilon', 'reward', 'loss', 'Q', 'Time delta', 'Datetime'])

        # self.record_time = time.time()

        self.init_episode()

    # exploration
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
            with autocast():
                action_values = self.net(state, model='online')
            action_idx = torch.argmax(
                action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    # memory
    def cache(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()

        # log
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

        # momory
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
        with autocast():
            next_state_Q = self.net(next_state, model='online')
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.net(next_state, model='target')[
                np.arange(0, self.batch_size), best_action
            ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        with autocast():
            loss = self.loss_fn(td_estimate, td_target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step < self.burnin + self.restart_steps:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += td_est.mean().item()
            self.curr_ep_loss_length += 1

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def log_episode(self, episode):
        self.episode = episode
        # self.ep_rewards.append(self.curr_ep_reward)
        # self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = self.curr_ep_loss / self.curr_ep_loss_length
            ep_avg_q = self.curr_ep_q / self.curr_ep_loss_length
        # self.ep_avg_losses.append(ep_avg_loss)
        # self.ep_avg_qs.append(ep_avg_q)
        wandb.log(dict(
            episode=episode,
            step=self.curr_step,
            reward=self.curr_ep_reward,
            length=self.curr_ep_length,
            average=ep_avg_loss,
            avg_qs=ep_avg_q
        ))
        self.init_episode()

    def save(self):
        save_path = (self.save_dir / f'mario_net.pth')
        # modelをsave
        torch.save(dict(
            model=self.net.state_dict(),
            exploration_rate=self.exploration_rate,
            step=self.curr_step,
            episode=self.episode
        ), save_path)
        # record
        # mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        # mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        # mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        # mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        # last_record_time = self.record_time
        # self.record_time = time.time()
        # time_since_last_record = np.round(
        #     self.record_time - last_record_time, 3)

        datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(
            f"Episode {self.episode} - "
            f"Step {self.curr_step} - "
            f"Epsilon {self.exploration_rate:.3f} - "
            # f"Mean Reward {mean_ep_reward} - "
            # f"Mean Length {mean_ep_length} - "
            # f"Mean Loss {mean_ep_loss} - "
            # f"Mean Q Value {mean_ep_q} - "
            # f"Time Delta {time_since_last_record} - "
            f"Time {datetime_now}"
        )

        # log_list = [self.curr_step, self.episode, self.exploration_rate,
        #             mean_ep_reward, mean_ep_loss, mean_ep_q, time_since_last_record, datetime_now]
        # self.log_df = self.log_df.append(
        #     {column: log for column, log in zip(self.log_df.columns, log_list)}, ignore_index=True)
        # self.log_df.to_csv(self.save_dir / 'log.csv', index=False)
        # # wandb
        # wandb.log({column: log for column, log in zip(
        #     self.log_df.columns, log_list)})

    def load(self):
        if self.init_learning:
            return 0
        if not (self.save_dir / 'mario_net.pth').exists():
            print('Zero Start')
            return 0
        load_data = torch.load(self.save_dir / 'mario_net.pth')
        # model
        self.net.load_state_dict(load_data['model'])
        # log
        # self.log_df = pd.read_csv(self.save_dir / 'log.csv')
        # logをparamに移行
        # self.exploration_rate = self.log_df['epsilon'].values[-1]
        self.exploration_rate = load_data['exploration_rate']
        # self.curr_step = self.log_df['step'].values[-1]
        self.curr_step = load_data['step']

        self.restart_steps = self.curr_step
        # self.restart_episodes = self.log_df['episode'].values[-1]
        self.restart_episodes = load_data['episode']
        print(f'Start from episode: {self.restart_episodes}')
        return self.restart_episodes

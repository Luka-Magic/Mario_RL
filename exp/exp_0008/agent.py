import random
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from model import MarioNet
from utils.SumTree import SumTree
from collections import deque, namedtuple
import pandas as pd
import pickle
import time
import datetime
import wandb
import zlib


class Mario:
    def __init__(self, cfg, action_dim, save_dir):
        # input
        self.action_dim = action_dim
        self.save_dir = save_dir

        # init
        self.wandb = cfg.wandb
        self.init_learning = cfg.init_learning
        self.curr_step = 0
        self.restart_steps = 0
        self.restart_episodes = 0  # 学習再開が何episode目から始めるか
        self.episode = self.restart_episodes  # 現在何episode目か
        self.episodes_num = cfg.episodes  # 学習するepisode合計数
        self.save_interval = cfg.save_interval
        self.save_model_interval = cfg.save_model_interval
        self.video_save_fps = cfg.video_save_fps
        self.Transition = namedtuple('Transition',
                                     ('state', 'next_state', 'action', 'reward', 'done'))

        # model
        self.state_dim = (cfg.state_channel, cfg.state_height, cfg.state_width)

        self.policy_net = MarioNet(
            cfg, self.state_dim, self.action_dim).float().to('cuda')
        self.target_net = MarioNet(
            cfg, self.state_dim, self.action_dim).float().to('cuda')
        self.sync_Q_target()
        self.target_net.eval()

        # exploration
        self.exploration_rate = cfg.exploration_rate
        self.exploration_rate_decay = cfg.exploration_rate_decay
        self.exploration_rate_min = cfg.exploration_rate_min

        # memory
        self.memory_length = cfg.memory_length
        self.batch_size = cfg.batch_size

        self.memory_compress = cfg.memory_compress
        self.priority_experience_reply = cfg.priority_experience_reply
        if self.priority_experience_reply:
            self.memory = SumTree(self.memory_length)
        else:
            self.memory = deque(maxlen=self.memory_length)
        self.priority_alpha = cfg.priority_alpha
        self.priority_epsilon = cfg.priority_epsilon
        self.priority_use_IS = cfg.priority_use_IS
        self.priority_beta = cfg.priority_beta

        # learn
        self.gamma = cfg.gamma
        self.scaler = GradScaler()
        if cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr=cfg.lr)
        if cfg.loss_fn == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss()
        self.burnin = cfg.burnin
        self.learn_every = cfg.learn_every
        self.sync_every = cfg.sync_every

        self.init_episode()
        self.load()

    # exploration
    def action(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state.__array__()
            state = torch.tensor(state).cuda()
            state = state.unsqueeze(0)
            with autocast():
                action_values = self.policy_net(state)
            action_idx = torch.argmax(
                action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    # memory
    def push(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()

        # log
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

        # momory
        state = torch.tensor(state).cuda()
        next_state = torch.tensor(next_state).cuda()
        action = torch.tensor([action]).cuda().squeeze()
        reward = torch.tensor([reward]).cuda().squeeze()
        done = torch.tensor([done]).cuda().squeeze()

        exp = self.Transition(state, next_state, action, reward, done)

        # memory compress
        if self.memory_compress:
            exp = zlib.compress(pickle.dumps(exp))

        # priority experience reply
        if self.priority_experience_reply:
            priority = self.memory.max()
            if priority <= 0:
                priority = 1
            self.memory.add(priority, exp)
        else:
            self.memory.append(exp)

    def sample(self):
        if self.priority_experience_reply:  # priority experience reply
            batch = []
            indices = []
            weights = np.empty(self.batch_size, dtype='float32')
            total = self.memory.total()
            beta = self.priority_beta + \
                (1 - self.priority_beta) * self.episode / self.episodes_num

            for i, rand in enumerate(np.random.uniform(0, self.memory.total(), self.batch_size)):
                idx, priority, memory = self.memory.get(rand)

                # weightを計算
                weights[i] = (self.memory_length * priority / total) ** (-beta)

                # decompress
                if self.memory_compress:
                    memory = pickle.loads(zlib.decompress(memory))
                batch.append(memory)
                indices.append(idx)
            weights /= weights.max()
        else:
            # decompress
            if self.memory_compress:
                indices = np.random.choice(
                    np.arange(len(self.memory)), replace=False, size=self.batch_size)
                batch = [pickle.loads(zlib.decompress(self.memory[idx]))
                         for idx in indices]
            else:
                batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = None

        transaction = self.Transition(*map(torch.stack, zip(*batch)))
        return (indices, transaction, weights)

    def td_estimate(self, state, action):
        current_Q = self.policy_net(state)[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        with autocast():
            next_state_Q = self.policy_net(next_state)
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.target_net(next_state)[
                np.arange(0, self.batch_size), best_action
            ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target, weights):
        if self.priority_use_IS:
            loss = torch.abs(td_estimate - td_target) * \
                torch.from_numpy(weights).mean().to('cuda')
        with autocast():
            loss = self.loss_fn(td_estimate, td_target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        return loss.item()

    def sync_Q_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        # check step num
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step < self.burnin + self.restart_steps:
            return
        if self.curr_step % self.learn_every != 0:
            return

        # sample
        indices, transaction, weights = self.sample()

        # learn
        td_est = self.td_estimate(transaction.state, transaction.action)
        td_tgt = self.td_target(
            transaction.reward, transaction.next_state, transaction.done)
        loss = self.update_Q_online(td_est, td_tgt, weights)

        # priority experience reply
        if self.priority_experience_reply:
            if (indices != None):
                for i, (td_est_i, td_tgt_i) in enumerate(zip(td_est, td_tgt)):
                    td_error = abs(td_est_i.item() - td_tgt_i.item())
                    priority = (
                        td_error + self.priority_epsilon) ** self.priority_alpha
                    self.memory.update(indices[i], priority)
        # log
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
        self.curr_ep_time = time.time()

    def log_episode(self, episode, info):
        self.episode = episode
        last_time = time.time()
        episode_time = last_time - self.curr_ep_time
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
            ep_step_per_second = 0
        else:
            ep_avg_loss = self.curr_ep_loss / self.curr_ep_loss_length
            ep_avg_q = self.curr_ep_q / self.curr_ep_loss_length
            ep_step_per_second = self.curr_ep_loss_length / episode_time
        wandb_dict = dict(
            episode=episode,
            step=self.curr_step,
            epsilon=self.exploration_rate,
            step_per_second=ep_step_per_second,
            reward=self.curr_ep_reward,
            length=self.curr_ep_length,
            average_loss=ep_avg_loss,
            average_q=ep_avg_q,
            dead_or_alive=int(info['flag_get']),
            x_pos=int(info['x_pos']),
            time=int(info['time'])
        )
        if info['video'] is not None:
            wandb_dict['video'] = wandb.Video(
                info['video'], fps=self.video_save_fps, format='mp4', caption=f'episode: {episode}')
        if self.wandb:
            wandb.log(wandb_dict)
        self.save(episode)
        self.init_episode()

    def save(self, episode):
        if episode != 0 and episode != self.restart_episodes:
            if episode % self.save_interval == 0:
                save_path = (self.save_dir / f'mario_net.pth')
                torch.save(dict(
                    model=self.policy_net.state_dict(),
                    exploration_rate=self.exploration_rate,
                    step=self.curr_step,
                    episode=self.episode
                ), save_path)
                datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                print(
                    f"Episode {self.episode} - "
                    f"Step {self.curr_step} - "
                    f"Epsilon {self.exploration_rate:.3f} - "
                    f"Time {datetime_now}"
                )
            if episode % self.save_model_interval == 0:
                save_path = (self.save_dir / f'mario_net_{episode}.pth')
                torch.save(dict(
                    model=self.policy_net.state_dict(),
                    exploration_rate=self.exploration_rate,
                    step=self.curr_step,
                    episode=self.episode
                ), save_path)

    def load(self):
        if self.init_learning:
            return 0
        if not (self.save_dir / 'mario_net.pth').exists():
            print('Zero Start')
            return 0
        load_data = torch.load(self.save_dir / 'mario_net.pth')
        # model
        self.policy_net.load_state_dict(load_data['model'])
        self.sync_Q_target()
        # log
        self.exploration_rate = load_data['exploration_rate']
        self.curr_step = load_data['step']
        self.restart_steps = self.curr_step
        self.restart_episodes = load_data['episode']
        print(f'Start from episode: {self.restart_episodes}')

from genericpath import exists
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random
import datetime
import os
import time
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
import hydra
from omegaconf import DictConfig

from IPython import display
from tqdm.notebook import tqdm

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

import wandb


class SkipFrame(gym.Wrapper):  # 数フレーム分スキップする (真隣同士のフレームはそこまで変化がないため) => rewardをスキップ間の合計のreward
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


# GrayScaleにする numpy(240, 256, 3) => tensor(1, 240, 256)
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Box(0, 255, (240, 256, 3), uint8)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):  # numpy(HWC) => tensor(CHW)
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = transforms.Grayscale()
        observation = transform(observation)
        return observation


# tensor(1, 240, 256) => tensor(1, 84, 84)
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # (shape[0], shape[1], チャンネル数)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms_ = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.Normalize(0, 255)
        ])
        observation = transforms_(observation).squeeze(0)
        return observation


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84:
            raise ValueError(f'Expecting input height: 84, got: {h}')
        if w != 84:
            raise ValueError(f'Expecting input width: 84, got: {w}')

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)  # ターゲット方策

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        # act
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # netを保存するまでの実験ステップの数

        # cache
        self.memory = deque(maxlen=50000)
        self.batch_size = 32

        # learn
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = nn.SmoothL1Loss()  # huberloss

        self.burnin = 1e4  # 経験を訓練させるために最低限必要なステップ数
        self.learn_every = 3  # 挙動方策を更新するタイミング
        self.sync_every = 1e4  # ターゲットと挙動の同期のタイミング

    def action(self, state):  # epsilon-greedyで行動を選択
        if np.random.rand() < self.exploration_rate:
            # 探索
            action_idx = np.random.randint(self.action_dim)

        else:
            # 活用
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')  # Q値を出力
            action_idx = torch.argmax(
                action_values, axis=1).item()  # Qが最大になるアクションを出力

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

    def sync_Q_target(self):  # ターゲット方策に挙動方策のパラメータを移す
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (self.save_dir /
                     f'mario_net_{int(self.curr_step // self.save_every)}.pth')
        torch.save(
            dict(model=self.net.state_dict(),
                 exploration_rate=self.exploration_rate), save_path
        )
        print(f'MarioNet saved to {save_path} at step {self.curr_step}')

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / 'log'
        with open(self.save_log, 'w') as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / 'reward_plot.jpg'
        self.ep_lengths_plot = save_dir / 'length_plot.jpg'
        self.ep_avg_losses_plot = save_dir / 'loss_plot.jpg'
        self.ep_avg_qs_plot = save_dir / 'q_plot.jpg'

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()

        self.record_time = time.time()

    def log_step(self, reward, loss, q):  # ステップ単位でカウントする (reward, loss, q)
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):  # episode毎に報酬の長さとloss, qの平均を記録
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss /
                                   self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):  # 初期化
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):  # 報酬や長さ、loss, qの移動平均を算出。
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(
            self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )

        with open(self.save_log, 'a') as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'):>20}\n"
            )

        for metric in ['ep_rewards', 'ep_lengths', 'ep_avg_losses', 'ep_avg_qs']:
            plt.plot(getattr(self, f'moving_avg_{metric}'))
            plt.savefig(getattr(self, f'{metric}_plot'))
            plt.clf()


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # 設定
    save_dir = Path('/'.join(os.getcwd().split('/')
                    [:-6])) / f"outputs/{os.getcwd().split('/')[-4]}"
    Path.touch(save_dir, exist_ok=True)
    print(save_dir)

    episodes = cfg.episodes
    use_cuda = torch.cuda.is_available()

    # 環境
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, [['right'], ['right', 'A']])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    # エージェント
    # load_state_dict
    mario = Mario(state_dim=(4, 84, 84),
                  action_dim=env.action_space.n, save_dir=save_dir)

    # ログ
    # loadできたら
    logger = MetricLogger(save_dir)

    # 学習
    # train_one_episodeを作る
    for episode in tqdm(range(episodes)):
        state = env.reset()
        while 1:
            action = mario.action(state)
            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)

            q, loss = mario.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if info['flag_get']:
                print(f'CLEAR!!!: Episode: {episode}')
                break

            if done:
                break

        logger.log_episode()

        if episode % 50 == 0:
            logger.record(episode=episode,
                          epsilon=mario.exploration_rate, step=mario.curr_step)


if __name__ == '__main__':
    main()

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


Transition = namedtuple(
    'Transition', ('state', 'next_state', 'action', 'reward', 'done'))


class Memory:
    def __init__(self, cfg):
        self.memory_size = cfg.memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.memory_compress = cfg.memory_compress

        self.batch_size = cfg.batch_size

    def _compress(self, exp):
        if self.memory_compress:
            exp = zlib.compress(pickle.dumps(exp))
        return exp

    def _decompress(self, exp):
        if self.memory_compress:
            exp = pickle.loads(zlib.decompress(exp))
        return exp

    def push(self, exp):
        exp = self._compress(exp)
        self.memory.append(exp)

    def sample(self, episode):
        sample_indices = np.random.choice(
            np.arange(len(self.memory)), replace=False, size=self.batch_size)
        batch = [self._decompress(self.memory[idx]) for idx in sample_indices]
        batch = Transition(*map(torch.stack, zip(*batch)))
        return (None, batch, None)

    def update(self, indices, td_error):
        pass

    def __len__(self):
        return len(self.memory)


class PERMemory(Memory):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.memory = SumTree(self.memory_size)

        self.n_episodes = cfg.n_episodes

        self.priority_alpha = cfg.priority_alpha
        self.priority_epsilon = cfg.priority_epsilon
        self.priority_use_IS = cfg.priority_use_IS
        self.priority_beta = cfg.priority_beta

    def push(self, exp):
        exp = self._compress(exp)
        priority = self.memory.max()
        if priority <= 0:
            priority = 1
        self.memory.add(priority, exp)

    def sample(self, episode):
        batch = []
        indices = []
        weights = np.empty(self.batch_size, dtype='float32')
        total = self.memory.total()
        beta = self.priority_beta + \
            (1 - self.priority_beta) * episode / self.n_episodes

        for i, rand in enumerate(np.random.uniform(0, total, self.batch_size)):
            idx, priority, exp = self.memory.get(rand)

            weights[i] = (self.memory_size * priority / total) ** (-beta)

            exp = self._decompress(exp)
            batch.append(exp)
            indices.append(idx)
        weights /= weights.max()

        batch = Transition(*map(torch.stack, zip(*batch)))
        return (indices, batch, weights)

    def update(self, indices, td_error):
        # if not self.priority_use_IS:
        #     return
        if indices != None:
            return

        for i in range(len(indices)):
            priority = (
                td_error[i] + self.priority_epsilon) ** self.priority_alpha
            self.memory.update(indices[i], priority)


class Brain:
    def __init__(self, cfg, n_actions, save_dir):
        # input
        self.n_actions = n_actions
        self.save_dir = save_dir

        # init
        self.cfg = cfg
        self.wandb = cfg.wandb
        self.restart_episode = 0  # 学習再開が何episode目から始めるか
        self.episode = 0  # 現在何episode目か
        self.n_episodes = cfg.n_episodes  # 学習するepisode合計数
        self.gamma = cfg.gamma

        # model
        self.policy_net, self.target_net = self._create_model(cfg)
        self.synchronize_model()
        self.target_net.eval()

        self.scaler = GradScaler()
        if cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.policy_net.parameters(), lr=cfg.lr)
        if cfg.loss_fn == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss()

        # exploration
        self.exploration_rate = cfg.exploration_rate
        self.exploration_rate_decay = cfg.exploration_rate_decay
        self.exploration_rate_min = cfg.exploration_rate_min

        # memory
        if cfg.PER.use_PER:
            self.memory = PERMemory(cfg)
        else:
            self.memory = Memory(cfg)

        self.multi_step_learning = cfg.multi_step_learning
        if self.multi_step_learning:
            self.n_multi_steps = cfg.n_multi_steps
            self.multi_step_trainsitions = deque(maxlen=self.n_multi_steps)

        self.noisy = cfg.noisy

        self.categorical = cfg.categorical
        if self.categorical:
            self.n_atoms = cfg.n_atoms
            self.V_min = cfg.V_min
            self.V_max = cfg.V_max
            if self.n_atoms > 1:
                self.delta_z = (self.V_max - self.V_min) / (self.n_atoms - 1)
                self.support = torch.linspace(
                    self.V_min, self.V_max, self.n_atoms).to('cuda')
        else:
            self.n_atoms = 1

    def synchronize_model(self):
        # モデルの同期
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _create_model(self, cfg):
        # modelを選べるように改変
        policy_net = MarioNet(
            self.cfg, self.n_actions).float().to('cuda')
        target_net = MarioNet(
            self.cfg, self.n_actions).float().to('cuda')
        return policy_net, target_net

    def select_action(self, state):
        # noisy
        epsilon = 0. if self.noisy else self.exploration_rate

        if np.random.rand() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = torch.tensor(state).cuda().unsqueeze(0)
            with torch.no_grad():
                if self.categorical:
                    Q = self._get_Q_categorical(self.policy_net, state)
                else:
                    Q = self._get_Q(self.policy_net, state)
            action = torch.argmax(
                Q, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)
        return action

    def send_memory(self, exp):
        if self.multi_step_learning:
            self.multi_step_trainsitions.append(exp)
            if len(self.multi_step_trainsitions) != self.n_multi_steps:
                return
            multi_step_reward = 0
            multi_step_done = False
            for i in range(self.n_multi_steps):
                _, _, _, reward, done = self.multi_step_trainsitions[i]
                multi_step_reward += reward * self.gamma ** i
                if done:
                    multi_step_done = True
                    break
            state = self.multi_step_trainsitions[0].state
            next_state = self.multi_step_trainsitions[-1].next_state
            action = self.multi_step_trainsitions[0].action
            exp = Transition(state, next_state, action,
                             multi_step_reward, multi_step_done)
        self.memory.push(exp)

    def update(self, episode):
        # sample
        indices, batch, weights = self.memory.sample(episode)

        if self.categorical:
            loss, td_error, q = self._loss_categorical(batch, weights)
        else:
            loss, td_error, q = self._loss(batch, weights)

        self.memory.update(indices, td_error)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        return loss.detach().cpu(), q

    def _get_Q(self, model, x):
        return model(x)

    def _get_Q_categorical(self, model, x):
        # self.policy_model.reset_noise()
        x = model(x, softmax='normal')
        return torch.sum(x * self.support, dim=2)

    def _loss(self, batch, weights):
        # state = torch.tensor(batch.state).to('cuda')
        # next_state = torch.tensor(batch.next_state).to('cuda')
        # action = torch.tensor(batch.action).to('cuda')
        # reward = torch.tensor(batch.reward).to('cuda')
        # done = torch.tensor(batch.done).to('cuda')
        state, next_state, action, reward, done = map(
            lambda x: torch.tensor(x).to('cuda'))

        # non_final_mask = torch.tensor(~.done).to('cuda')
        # non_final_next_state = torch.stack([next_state for not])
        with autocast():  # 大丈夫？ + no_gradも?
            Q = self.policy_net(state)
            td_estimate = Q[np.arange(0, self.batch_size), action.squeeze()]
            if self.double:
                next_state_Q = self.policy_net(next_state)
                best_action = torch.argmax(next_state_Q, axis=1)
                with torch.no_grad():
                    next_Q = self.target_net(next_state)[
                        np.arange(0, self.batch_size), best_action
                    ]
            else:
                with torch.no_grad():
                    next_Q = torch.max(self.target_net(next_state), axis=1)

            td_target = (reward + (1. - done.float())
                         * self.gamma * next_Q).float()

            td_error = torch.abs(td_target - td_estimate)

            if self.priority_use_IS:
                loss = (td_error * torch.from_numpy(weights)).mean()
            else:
                loss = self.loss_fn(td_estimate, td_target)
        return loss, td_error.detach().cpu(), td_estimate.detach().cpu()

    def _loss_categorical(self, batch, weights):
        # state = torch.tensor(batch.state).to('cuda')
        # next_state = torch.tensor(batch.next_state).to('cuda')
        # action = torch.tensor(batch.action).to('cuda')
        # reward = torch.tensor(batch.reward).to('cuda')
        # done = torch.tensor(batch.done).to('cuda')
        state, next_state, action, reward, done = map(
            lambda x: torch.tensor(x).to('cuda'))

        non_final_mask = torch.tensor(~done).to('cuda')
        non_final_next_state = torch.stack([one_next_state for not_done, one_next_state in zip(
            non_final_mask, next_state) if not_done])
        non_final_action = torch.stack([one_action for not_done, one_action in zip(
            non_final_mask, action) if not_done])

        with torch.no_grad():
            # terminal stateだけ取り除く処理
            Q = self._get_Q_categorical(
                self.policy_net, non_final_next_state)
            td_estimate = Q[np.arange(0, len(Q)), non_final_action.squeeze()]
            best_actions = Q.argmax(dim=1)
            # self.target_model.reset_model()
            p_next = self.target_net(non_final_next_state, softmax='normal')

            p_next_best = torch.zeros(0).to('cuda', dtype=torch.float32).new_full(
                (self.batch_size, self.n_atoms), 1.0 / self.n_atoms)
            p_next_best[non_final_mask] = p_next[range(
                len(non_final_next_state)), best_actions]

            gamma = torch.zeros(self.batch_size, self.n_atoms).to('cuda')
            gamma[non_final_mask] = self.gamma

            Tz = reward.unsqueeze(
                1) + gamma * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.V_min, self.V_max)
            b = (Tz - self.V_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()

            l[(l == u) * (0 < l)] -= 1
            u[(l == u) * (u < self.n_atoms - 1)] += 1

            m = torch.zeros(self.batch_size, self.n_atoms).to(
                'cuda', dtype=torch.float32)
            offset = torch.linspace(0, ((self.batch_size-1) * self.n_atoms),
                                    self.batch_size).unsqueeze(1).expand(self.batch_size, self.n_atoms).to(l)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (p_next_best * (u.float() - b)).float().view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (p_next_best * (b - l.float())).float().view(-1))
        # self.model.reset_noise()
        with autocast():
            log_z = self.policy_net(state, softmax='log')
            log_z_a = log_z[range(self.batch_size), action.squeeze()]
            losses = -torch.sum(m * log_z_a, dim=1)
            if self.priority_use_IS:
                loss = (losses * torch.from_numpy(weights)).mean()
            else:
                loss = losses.mean()
        return loss, losses.detach.cpu(), td_estimate.detach().cpu()


class Mario:
    def __init__(self, cfg, n_actions, save_dir):
        self.cfg = cfg
        self.step = 0
        self.episode = 0
        self.restart_step = 0
        self.restart_episode = 0

        self.sync_every = cfg.sync_every
        self.burnin = cfg.burnin
        self.learn_every = cfg.learn_every

        self.brain = Brain(cfg, n_actions, save_dir)
        self.wandb = cfg.wandb
        if self.wandb:
            self.logger = Logger(cfg, self.restart_episode)

    def action(self, state):
        self.step += 1
        action = self.brain.select_action(state.__array__())
        return action

    def observe(self, state, next_state, action, reward, done):
        exp = Transition(state.__array__(),
                         next_state.__array__(),
                         action,
                         reward,
                         done
                         )
        self.brain.send_memory(exp)

    def learn(self):
        # check step num
        if self.step % self.sync_every == 0:
            self.brain.synchronize_model()
        if self.step < self.burnin + self.restart_step:
            return
        if self.step % self.learn_every != 0:
            return
        loss, q = self.brain.update(self.episode)

        if self.wandb:
            self.logger.step(self.brain.exploration_rate, loss, q)

    def restart_learning(self, checkpoint_path):
        self._reset_episode_log()

        checkpoint = torch.load(checkpoint_path)
        self.brain.policy_net.load_state_dict(checkpoint['model'])
        self.brain.synchronize_model()

        self.brain.exploration_rate = checkpoint['exploration_rate']
        self.restart_step = checkpoint['step']
        self.restart_episode = checkpoint['episode']
        print(f'Restart learning from episode {self.restart_episode}')
        self.step = self.restart_step
        self.episode = self.restart_episode
        return self.restart_episode

    def log_episode(self, episode, info):
        if self.wandb == False:
            return
        self.episode = episode
        self.logger.log_episode(episode, info)

        if episode != 0 and episode != self.restart_episode:
            if episode % self.save_checkpoint_interval == 0:
                self._save_checkpoint(episode)
            if episode % self.save_model_interval == 0:
                self._save(episode)

    def _save_checkpoint(self, episode):
        checkpoint_path = (self.save_dir / f'mario_net.ckpt')
        torch.save(dict(
            model=self.brain.policy_net.state_dict(),
            exploration_rate=self.exploration_rate,
            step=self.step,
            episode=episode
        ), checkpoint_path)
        datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(
            f"Episode {episode} - "
            f"Step {self.step} - "
            f"Epsilon {self.exploration_rate:.3f} - "
            f"Time {datetime_now}"
        )

    def _save(self, episode):
        checkpoint_path = (self.save_dir / f'mario_net_{episode}.ckpt')
        torch.save(dict(
            model=self.brain.policy_net.state_dict(),
            exploration_rate=self.brain.exploration_rate,
            step=self.step,
            episode=episode
        ), checkpoint_path)


class Logger:
    def __init__(self):
        self.episode_last_time = time.time()
        self.step = 0
        self._reset_episode_log()

    def _reset_episode_log(self):
        # 変数名どうしよう、logとかつけたらわかりやすそう
        # reward_during_one_episode? -> episode_reward
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_loss = 0.0
        self.episode_q = 0.0
        self.episode_loss_length = 0
        self.episode_start_time = self.episode_last_time

    def step(self, exploration_rate, loss, q):
        self.step += 1
        self.episode_loss += loss
        self.episode_loss_length += 1
        self.episode_q += q
        self.exploration_rate = exploration_rate

    def log_episode(self, episode, info):
        self.episode_last_time = time.time()
        episode_time = self.episode_last_time - self.episode_start_time
        if self.episode_loss_length == 0:
            episode_average_loss = 0
            episode_average_q = 0
            episode_step_per_second = 0
        else:
            episode_average_loss = self.episode_loss / self.episode_loss_length
            episode_average_q = self.episode_q / self.episode_loss_length
            episode_step_per_second = self.episode_loss_length / episode_time

        wandb_dict = dict(
            episode=episode,
            step=self.step,
            epsilon=self.exploration_rate,
            step_per_second=episode_step_per_second,
            reward=self.episode_reward,
            length=self.episode_length,
            average_loss=episode_average_loss,
            average_q=episode_average_q,
            dead_or_alive=int(info['flag_get']),
            x_pos=int(info['x_pos']),
            time=int(info['time'])
        )
        if info['video'] is not None:
            wandb_dict['video'] = wandb.Video(
                info['video'], fps=self.video_save_fps, format='mp4', caption=f'episode: {episode}')
            wandb.log(wandb_dict)

        self._reset_episode_log()

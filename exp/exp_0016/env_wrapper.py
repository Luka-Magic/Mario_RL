import numpy as np
import torch
from torchvision import transforms
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from utils.customwrappers import CustomRecordVideo


class SkipFrame(gym.Wrapper):
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

class RunForYourLifeEnv(gym.Wrapper):
    def __init__(self, env, init_episode=0, threshold=80):
        super().__init__(env)
        self.last_x_pos = 0
        self.count = 0
        self.episode = init_episode
        self.init_threshold = threshold
        self.threshold = max(1, self.init_threshold - 10*(self.episode//50))

    def reset(self, **kwargs):
        self.last_x_pos = 0
        self.count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        x_pos = info['x_pos']
        if x_pos <= self.last_x_pos:
            self.count += 1
        else:
            self.count = 0
            self.last_x_pos = x_pos

        if x_pos > 50 and self.count >= self.threshold:
            reward = -15
            done = True
        
        if done: # episodeをカウントする。
            self.episode += 1
            print(f'thresold: {self.thresold}')
        
        self.threshold = max(1, self.init_threshold - 10*(self.episode//50))
        
        return state, reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
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


def all_wrapper(env, cfg, init_episode):
    env = RunForYourLifeEnv(env, init_episode=init_episode)
    env = CustomRecordVideo(
        env, cfg, init_episode=init_episode)
    env = SkipFrame(env, skip=cfg.state_skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(cfg.state_height, cfg.state_width))
    env = FrameStack(env, num_stack=cfg.state_channel)
    return env

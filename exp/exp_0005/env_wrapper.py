import numpy as np
import torch
from torchvision import transforms
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack, RecordVideo
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


def all_wrapper(env, cfg, video_folder, init_episode):
    env = CustomRecordVideo(env, video_folder=video_folder,
                            init_episode=init_episode)
    env = SkipFrame(env, skip=cfg.state_skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(cfg.state_height, cfg.state_width))
    env = FrameStack(env, num_stack=cfg.state_channel)
    return env

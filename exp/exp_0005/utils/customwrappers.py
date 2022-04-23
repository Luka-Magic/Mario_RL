from gym import logger
import numpy as np
import os.path
import os
from typing import Callable
import gc

import gym
from gym import logger


def capped_cubic_video_schedule(episode_id, video_save_interval):  # 条件
    return episode_id % video_save_interval == 0


class CustomRecordVideo(gym.Wrapper):
    def __init__(
        self,
        env,
        cfg,
        init_episode=0,
        episode_trigger: Callable[[int], bool] = None,
    ):
        super().__init__(env)
        # 条件設定
        if episode_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        self.video_save_interval = cfg.video_save_interval
        self.episode_trigger = episode_trigger

        self.recording = False
        self.episode_id = init_episode
        self.frames = []

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
            del self.frames
            gc.collect()
            self.frames = []
            frame = observations.copy()
            self.frames.append(frame)
        return observations

    def start_video_recorder(self):
        self.recording = True

    def _video_enabled(self):
        return self.episode_trigger(self.episode_id, self.video_save_interval)

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        
        dones = bool(dones)
        video = None
        if infos['flag_get']:
            dones = True

        if dones:
            self.episode_id += 1

        if self.recording:
            frame = observations.copy()
            self.frames.append(frame)
            if dones:
                video = self.close_video_recorder()
        
        infos['video'] = video
        return observations, rewards, dones, infos

    def close_video_recorder(self) -> None:
        self.recording = False
        if len(self.frames):
            video = np.stack(self.frames).transpose(0, 3, 1, 2)
            return video
        else:
            return None
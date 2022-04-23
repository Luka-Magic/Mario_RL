from gym import logger
import numpy as np
import os.path
import os
from typing import Callable

import gym
from gym import logger


def capped_cubic_video_schedule(episode_id):  # 条件
    return episode_id % 100 == 0


class CustomRecordVideo(gym.Wrapper):
    def __init__(
        self,
        env,
        video_folder: str,
        init_episode=0,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "mario_rl",
    ):
        super().__init__(env)
        # 条件設定
        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule
        trigger_count = sum(x is not None for x in [
                            episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = None
        # video_folderを作成
        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = init_episode
        self.frames = []

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
            self.frames = []
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    # actionが入力されたときの反応
    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        video = None
        if infos['flag_get']:
            dones = True

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            frame = observations.copy()
            self.frames.append(frame)
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    video = self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        video = self.close_video_recorder()
                elif dones[0]:
                    video = self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()
        infos['video'] = video
        return observations, rewards, dones, infos

    def close_video_recorder(self) -> None:
        self.recording = False
        self.recorded_frames = 1
        if len(self.frames):
            video = np.stack(self.frames).transpose(0, 3, 1, 2)
            return video
        else:
            return None

import os
import hydra
import wandb
from pathlib import Path
from omegaconf import DictConfig
from tqdm.notebook import tqdm

import torch

# 環境
import gym_super_mario_bros

from env_wrapper import SkipFrame, GrayScaleObservation, ResizeObservation
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

# エージェント
from model import MarioNet
from agent import Mario

# ログ
from logger import MetricLogger


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # 設定
    save_dir = Path('/'.join(os.getcwd().split('/')
                    [:-6])) / f"outputs/{os.getcwd().split('/')[-4]}"
    Path.touch(save_dir, exist_ok=True)

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

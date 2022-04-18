import os
import hydra
import wandb
from pathlib import Path
from omegaconf import DictConfig
from tqdm.notebook import tqdm
import torch

# 環境
import gym_super_mario_bros

from env_wrapper import all_wrapper

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
    save_dir.mkdir(exist_ok=True)

    # 環境
    env = gym_super_mario_bros.make(cfg.environment)
    env = all_wrapper(env, cfg)

    # エージェント
    # load_state_dict
    mario = Mario(cfg, action_dim=env.action_space.n, save_dir=save_dir)
    if not cfg.init_learning:
        pass

    # ログ
    # loadできたら
    logger = MetricLogger(save_dir)

    # 学習
    # train_one_episodeを作る
    for episode in tqdm(range(cfg.episodes)):
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

        if episode % cfg.save_interval == 0:
            mario.save()
            logger.record(episode=episode,
                          epsilon=mario.exploration_rate, step=mario.curr_step)


if __name__ == '__main__':
    main()

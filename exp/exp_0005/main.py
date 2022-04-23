import os
import hydra
import wandb
from pathlib import Path
from omegaconf import DictConfig
from tqdm.notebook import tqdm
# 環境
import gym_super_mario_bros
from env_wrapper import all_wrapper
# エージェント
from agent import Mario
import warnings


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # 設定
    save_dir = Path('/'.join(os.getcwd().split('/')
                    [:-6])) / f"outputs/{os.getcwd().split('/')[-4]}"
    save_dir.mkdir(exist_ok=True)
    warnings.simplefilter('ignore')

    # wandb
    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.wandb_project, entity='luka-magic',
                   name=os.getcwd().split('/')[-4], config=cfg)

    # 環境
    env = gym_super_mario_bros.make(cfg.environment)
    env = all_wrapper(env, cfg, save_dir)

    # エージェント
    mario = Mario(cfg, action_dim=env.action_space.n, save_dir=save_dir)
    init_episode = mario.restart_episodes
    env.episode_id = init_episode
    env.name_prefix = 'mario_rl'
    print(env.episode_id)
    print(env.name_prefix)

    # 学習
    for episode in tqdm(range(init_episode, cfg.episodes)):
        state = env.reset()
        while True:
            action = mario.action(state)
            next_state, reward, done, info = env.step(action)
            mario.push(state, next_state, action, reward, done)
            mario.learn()
            state = next_state

            if done or info['flag_get']:
                break
        mario.log_episode(episode, info)


if __name__ == '__main__':
    main()

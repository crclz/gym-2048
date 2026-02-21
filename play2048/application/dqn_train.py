import sys


sys.path.append(".")

from gym_2048.envs.game2048_env import Game2048Env
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from stable_baselines3 import DQN

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

class MaxTileCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MaxTileCallback, self).__init__(verbose)
        self.max_tiles = []

    def _on_step(self) -> bool:
        # 检查是否有 episode 结束
        # 'dones' 是一个布尔数组，因为可能存在向量化环境
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                # 尝试从 info 中获取最大值，这里取决于你的环境实现
                # 如果环境 info 里没有 'max_tile'，可能需要从 obs 提取
                highest = info["highest"]
                
                self.logger.record_mean("rollout/highest_block", highest)

        self.logger.record("rollout/group_size", len(self.locals["dones"]))
        return True

def make_env_maker(rank, seed=0):
    """
    实用工具函数，用于多进程环境创建
    """
    def _init():
        env = gym.make("2048-v0", render_mode="rgb_array", illegal_move_reward=-2.0)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    worker_count = 8

    envs = []
    for i in range(worker_count):
        envs.append(make_env_maker(i))

    the_game_env = SubprocVecEnv(envs)
    # the_game_env = envs[0]
    del envs


    # random model
    use_random = False

    if use_random:
        model = DQN(
            "MlpPolicy", 
            the_game_env, 
            verbose=1, 
            tensorboard_log="./tensorboard/dqn-rand",
            learning_starts=10000000, # 设一个极大的值，让它永远不开始学习优化网络
            exploration_fraction=1.0, # 整个训练过程都在探索
            exploration_initial_eps=1.0, # 初始探索率为 1
            exploration_final_eps=1.0    # 最终探索率也为 1
        )

    else:
        model = DQN("MlpPolicy", the_game_env, verbose=1, tensorboard_log="./tensorboard/dqn-1")

    model.learn(total_timesteps=200_0000, callback=MaxTileCallback())

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

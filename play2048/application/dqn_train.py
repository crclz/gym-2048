import sys


sys.path.append(".")

from gym_2048.envs.game2048_env import Game2048Env

import gymnasium as gym

from stable_baselines3 import DQN

from stable_baselines3.common.callbacks import BaseCallback

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
                
                self.logger.record("rollout/highest_block", highest)
        return True

if __name__ == "__main__":
    env: Game2048Env = gym.make("2048-v0", render_mode="rgb_array", illegal_move_reward=-1.0)

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000_0000, callback=MaxTileCallback())

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

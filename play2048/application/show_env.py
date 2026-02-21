import gymnasium as gym
import sys
from PIL import Image
import numpy as np


sys.path.append(".")

from gym_2048.envs.game2048_env import Game2048Env

import gym_2048


if __name__ == "__main__":
    env: Game2048Env = gym.make("2048-v0", render_mode="human")
    a = env.reset()

    print("env reset output:", a)

    # board = env.get_board()
    # print("env board:", board)

    for i in range(30):
        action = env.action_space.sample()
        print("env action:", action)

        (obs, reward, terminated, truncated, info) = env.step(action)
        print("env step output. obs:", obs, "reward:", reward, "terminated:", terminated, "truncated:", truncated, "info:", info)

        if terminated or truncated:
            print("terminated or truncated")
            break

        env.render()

    # frame = env.render()
    # img = Image.fromarray(frame)
    # img.save("show_env.tmp.png")

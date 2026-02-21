import gym
import sys
from PIL import Image
import numpy as np


sys.path.append(".")

from gym_2048.envs.game2048_env import Game2048Env

import gym_2048


if __name__ == "__main__":
    env: Game2048Env = gym.make("2048-v0")
    a = env.reset()

    print("env reset output:", a)

    board = env.get_board()
    print("env board:", board)

    action = env.action_space.sample()
    print("env action:", action)
    img_array = env.render(mode="rgb_array")

    ob, reward, done, info = env.step(action)
    print("env step output. ob:", ob, "reward:", reward, "done:", done, "info:", info)

    # img = Image.fromarray(img_array.astype(np.uint8)).transpose(Image.TRANSPOSE)
    # img.save("./2048_manual_save.png")

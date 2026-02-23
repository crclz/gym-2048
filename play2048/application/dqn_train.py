import sys
import time
from typing import SupportsFloat

from play2048.domain.models.cnn_extractor import CnnExtractor
from play2048.infra.better_eval import BetterEvalCallback
from sb3_contrib import QRDQN

sys.path.append(".")

from gym_2048.envs.game2048_env import Game2048Env  # noqa: F401
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium.wrappers import NormalizeReward, TimeLimit

from stable_baselines3 import DQN
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

TIME_STEP_LIMIT = 5000  # 理论上2000步能完成2048，给5000步，然后不惩罚任何无效移动


class MaxTileCallback(BaseCallback):
    def __init__(self, verbose=0, metrics_prefix="rollout"):
        """
        metrics_prefix: rollout, eval
        """
        super(MaxTileCallback, self).__init__(verbose)
        self.metrics_prefix = metrics_prefix
        # self.max_tiles = []

    def _on_step(self) -> bool:
        # 检查是否有 episode 结束
        # 'dones' 是一个布尔数组，因为可能存在向量化环境
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                # 尝试从 info 中获取最大值，这里取决于你的环境实现
                # 如果环境 info 里没有 'max_tile'，可能需要从 obs 提取
                highest = info["highest"]

                self.logger.record_mean(f"{self.metrics_prefix}/highest_block", highest)

        self.logger.record(
            f"{self.metrics_prefix}/group_size", len(self.locals["dones"])
        )
        return True


class Log2RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward: SupportsFloat): 
        # 2048 中 reward 通常是合并产生的分数
        # 如果 reward > 0（发生了合并），取 log2
        # 如果 reward <= 0（比如无效移动或没合并），保持原样或设为 0
        reward = float(reward)

        if reward > 0:
            return float(np.log2(reward))

        return reward


def make_env_maker(rank, seed=0):
    """
    实用工具函数，用于多进程环境创建
    """

    def _init():
        env = gym.make("2048-v0", render_mode="rgb_array", illegal_move_reward=0)
        env = Log2RewardWrapper(env)
        env = NormalizeReward(env)
        env = TimeLimit(env, TIME_STEP_LIMIT)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def make_sub_process_env(count: int, eval=False):

    rank_add = 0
    if eval:
        rank_add = 5555

    envs = []
    for i in range(count):
        envs.append(make_env_maker(rank_add + i))

    the_game_env = SubprocVecEnv(envs)
    return the_game_env


def make_model(env, name: str):
    if name == "random":
        return DQN(
            "MlpPolicy",
            make_sub_process_env(8, eval=False),
            verbose=1,
            tensorboard_log="./tensorboard/dqn-rand",
            learning_starts=10000000,  # 设一个极大的值，让它永远不开始学习优化网络
            exploration_fraction=1.0,  # 整个训练过程都在探索
            exploration_initial_eps=1.0,  # 初始探索率为 1
            exploration_final_eps=1.0,  # 最终探索率也为 1
        )

    if name == "dqn":
        policy_kwargs = dict(net_arch=[256, 256, 256, 256])
        return DQN(
            "MlpPolicy",
            env,
            learning_starts=50000,
            # exploration_fraction=0.01,
            verbose=1,
            tensorboard_log="./tensorboard/dqn-1",
            policy_kwargs=policy_kwargs,
        )

    if name == "qrdqn":
        policy_kwargs = dict(net_arch=[256, 256, 256], n_quantiles=50)
        return QRDQN(
            "MlpPolicy",
            env,
            learning_starts=50000,
            # exploration_fraction=0.1,
            verbose=1,
            tensorboard_log="./tensorboard/qrdqn",
            policy_kwargs=policy_kwargs,
        )

    if name == "dqn-conv":
        policy_kwargs = dict(
            features_extractor_class=CnnExtractor,
            features_extractor_kwargs=dict(),
            net_arch=[],
        )
        return DQN(
            "MlpPolicy",
            env,
            learning_starts=50000,
            # exploration_fraction=0.01,
            verbose=1,
            tensorboard_log="./tensorboard/dqn-conv",
            policy_kwargs=policy_kwargs,
        )

    if name == "qrdqn-conv":
        policy_kwargs = dict(
            features_extractor_class=CnnExtractor,
            features_extractor_kwargs=dict(),
            net_arch=[],
            n_quantiles=50,
        )
        return QRDQN(
            "MlpPolicy",
            env,
            learning_starts=50000,
            # exploration_fraction=0.1,
            verbose=1,
            tensorboard_log="./tensorboard/qrdqn-cnn",
            policy_kwargs=policy_kwargs,
        )

    raise ValueError(f"Unknown model name: {name}")


def main():
    worker_count = 64
    model_name = "qrdqn-conv"

    eval_env = make_sub_process_env(worker_count, eval=True)
    train_env = make_sub_process_env(worker_count, eval=False)

    eval_callback = BetterEvalCallback(
        eval_env,
        n_eval_episodes=20,
        best_model_save_path="./checkpoints/best_model",  # 自动保存得分最高的模型
        log_path="./logs/eval_results",  # 记录评估结果
        # 这个step不是训练step，而是callback step，要等待并行的才算step1次。建议积极尝试寻找合理的。
        eval_freq=int(3e4),
        deterministic=True,  # 评估时使用确定性动作（DQN 必选）
        render=False,  # 评估时是否渲染（建议关闭以加速）
        info_metrics={
            "avg": ["highest"],
            "max": ["highest"],
            "min": ["highest"],
        },
    )

    model = make_model(train_env, model_name)

    print(model.policy)
    time.sleep(2)

    # exit(0)

    callbacks = [eval_callback, MaxTileCallback()]

    model.learn(total_timesteps=2000_0000, callback=callbacks, log_interval=60)

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

if __name__ == "__main__":
    main()

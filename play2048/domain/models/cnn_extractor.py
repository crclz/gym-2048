from collections import OrderedDict
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim=128):
        # features_dim 是最后输出给 Policy 的维度，这里设为你的 512 FC
        super().__init__(observation_space, features_dim)

        assert observation_space.shape is not None

        assert observation_space.shape == (4, 4, 16), f"observation_space.shape[0] is {observation_space.shape}"

        input_channels = observation_space.shape[2]

        self.sequential = nn.Sequential(
            OrderedDict(
                [
                    # 第一层：2x2, 256 filters, 无 padding
                    # Input: (N, 16, 4, 4) -> Output: (N, 256, 3, 3)
                    ("conv1", nn.Conv2d(input_channels, 256, kernel_size=2, stride=1, padding=0)),
                    ("relu1", nn.ReLU()),
                    # 第二层：2x2, 256 filters, 无 padding
                    # Input: (N, 256, 3, 3) -> Output: (N, 256, 2, 2)
                    ("conv2", nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)),
                    ("relu2", nn.ReLU()),
                    # 展平: (N, 256, 2, 2) -> (N, 1024)
                    ("flatten", nn.Flatten()),
                    # 第一层全连接 (FC)
                    ("fc1", nn.Linear(256 * 2 * 2, 512)),
                    ("relu3", nn.ReLU()),
                    # 第二层全连接 (FC)
                    # 注意：这里的输入是 512，输出是 features_dim
                    ("fc2", nn.Linear(512, features_dim)),
                    ("relu4", nn.ReLU()),
                ]
            )
        )

    def forward(self, x):
        # first permute N*4*4*16 => N*16*4*4
        assert x.shape == (x.shape[0], 4, 4, 16), f"x.shape is {x.shape}"
        x = x.permute(0, 3, 1, 2)
        x = self.sequential(x)
        return x

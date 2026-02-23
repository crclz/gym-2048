from collections import OrderedDict
import torch
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
                    ("conv1", nn.Conv2d(input_channels, 128, kernel_size=2, stride=1, padding=0)),
                    # ("ln1", nn.LayerNorm([256, 3, 3])),
                    ("relu1", nn.ReLU()),
                    # 第二层：2x2, 256 filters, 无 padding
                    # Input: (N, 256, 3, 3) -> Output: (N, 256, 2, 2)
                    ("conv2", nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)),
                    # ("ln2", nn.LayerNorm([256, 2, 2])),
                    ("relu2", nn.ReLU()),
                    # 展平: (N, 256, 2, 2) -> (N, 1024)
                    ("flatten", nn.Flatten()),
                    # 第一层全连接 (FC)
                    ("fc1", nn.Linear(128 * 2 * 2, 256)),
                    ("relu3", nn.ReLU()),
                    # 第二层全连接 (FC)
                    # 注意：这里的输入是 512，输出是 features_dim
                    ("fc2", nn.Linear(256, features_dim)),
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

class MultiBranchCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim=128):
        super().__init__(observation_space, features_dim)

        assert observation_space.shape

        input_channels = observation_space.shape[2] # 16

        # --- Layer 1 分支 ---
        # c1: 1x2 卷积捕捉水平特征
        self.c1 = nn.Conv2d(input_channels, 128, kernel_size=(1, 2))
        # c2: 2x1 卷积捕捉垂直特征
        self.c2 = nn.Conv2d(input_channels, 128, kernel_size=(2, 1))

        # --- Layer 2 分支 ---
        # 注意：TF 代码中 c3, c4 是共享权重的名空间，但在结构上是并行的
        # 从 relu1 (水平特征) 进一步提取
        self.c3_on_relu1 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.c4_on_relu1 = nn.Conv2d(128, 128, kernel_size=(2, 1))
        # 从 relu2 (垂直特征) 进一步提取
        self.c3_on_relu2 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.c4_on_relu2 = nn.Conv2d(128, 128, kernel_size=(2, 1))

        # --- 全连接层 ---
        # 维度对齐计算 (Valid Padding):
        # relu1: 4x3x128 = 1536 | relu2: 3x4x128 = 1536
        # relu11: 4x2x128 = 1024 | relu12: 3x3x128 = 1152
        # relu21: 3x3x128 = 1152 | relu22: 2x4x128 = 1024
        # Total Concat Size = 1536 + 1536 + 1024 + 1152 + 1152 + 1024 = 7424
        concat_size = 7424

        self.fc1 = nn.Linear(concat_size, 512)
        self.fc2 = nn.Linear(512, features_dim)
        self.relu = nn.ReLU()

        # 初始化对齐: init_scale=np.sqrt(2) 对应 kaiming_normal
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N, 4, 4, 16 -> N, 16, 4, 4
        x = x.permute(0, 3, 1, 2).float()

        # Layer 1
        relu1 = self.relu(self.c1(x)) # Output: (N, 128, 4, 3)
        relu2 = self.relu(self.c2(x)) # Output: (N, 128, 3, 4)

        # Layer 2
        relu11 = self.relu(self.c3_on_relu1(relu1)) # (N, 128, 4, 2)
        relu12 = self.relu(self.c4_on_relu1(relu1)) # (N, 128, 3, 3)
        relu21 = self.relu(self.c3_on_relu2(relu2)) # (N, 128, 3, 3)
        relu22 = self.relu(self.c4_on_relu2(relu2)) # (N, 128, 2, 4)

        # Flatten & Concatenate
        h = torch.cat([
            torch.flatten(relu1, 1),
            torch.flatten(relu2, 1),
            torch.flatten(relu11, 1),
            torch.flatten(relu12, 1),
            torch.flatten(relu21, 1),
            torch.flatten(relu22, 1)
        ], dim=1)

        # FC Layers
        linear_1 = self.relu(self.fc1(h))
        linear_2 = self.relu(self.fc2(linear_1))

        return linear_2
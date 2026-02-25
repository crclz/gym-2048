from __future__ import print_function

import gymnasium as gym  # Modified: Replace gym with gymnasium
from gymnasium import spaces  # Modified: Use gymnasium spaces
from gymnasium.utils import seeding  # Modified: Use gymnasium seeding

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import itertools
import logging
from six import StringIO
import sys

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

def stack(flat, layers=16):
  """Convert an [4, 4] representation into [4, 4, layers] with one layers for each value."""
  # representation is what each layer represents
  representation = 2 ** (np.arange(layers, dtype=int) + 1)

  # layered is the flat board repeated layers times
  layered = np.repeat(flat[:,:,np.newaxis], layers, axis=-1)

  # Now set the values in the board to 1 or zero depending whether they match representation.
  # Representation is broadcast across a number of axes
  layered = np.where(layered == representation, 1, 0)

  return layered

class Game2048Env(gym.Env):  # gymnasium.Env (aliased as gym)
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 6,
    }

    # ========== 新增：定义角落奖励的权重矩阵 ==========
    WEIGHT_MATRIX = np.array([
        [2**0, 2**1, 2**2, 2**3],
        [2**4, 2**5, 2**6, 2**7],
        [2**8, 2**9, 2**10, 2**11],
        [2**12, 2**13, 2**14, 2**15]
    ])

    WEIGHT_MATRIX_SUM = np.sum(WEIGHT_MATRIX)

    def __init__(self, render_mode=None, illegal_move_reward:float=0, illegal_move_truncate:int=-1):  # 添加 render_mode 参数
        # 验证 render_mode 合法性
        assert render_mode is None or render_mode in self.metadata["render_modes"], \
            f"Invalid render_mode {render_mode}, only support {self.metadata['render_modes']}"
        self.render_mode = render_mode

        self.set_illegal_move_reward(illegal_move_reward)
        self.illegal_move_truncate = illegal_move_truncate

        self.illegal_move_count = 0

        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, layers), dtype=int)
        # self.set_illegal_move_reward(0.)
        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70

        # 渲染相关资源初始化
        self._font = None  # 字体资源，延迟初始化

        # 新增：Pygame 相关变量
        self.window = None
        self.clock = None

    # ========== 新增：计算角落奖励的方法 ==========
    def _calculate_corner_reward(self):
        """计算基于权重矩阵的角落奖励"""
        # 逐元素相乘后求和
        corner_reward = np.sum(self.Matrix * self.WEIGHT_MATRIX)
        corner_reward = corner_reward / self.WEIGHT_MATRIX_SUM
        return float(corner_reward)

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        # self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        info = {
            'illegal_move': False,
            'corner_reward': 0.0,  # ========== 新增：初始化角落奖励 ==========
        }
        try:
            score = float(self.move(action))
            self.score += score
            assert score <= 2**(self.w*self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)
            
            # ========== 新增：有效移动时计算并赋值角落奖励 ==========
            info['corner_reward'] = self._calculate_corner_reward()
            
            # 成功移动，重置无效计数
            self.illegal_move_count = 0
            
        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            # done = True # 为了让episode不陷入死循环，这里直接kill。后续可以连续10个illegal move才kill

            
            # 修改逻辑：增加计数，仅在达到5次时结束
            self.illegal_move_count += 1
            if self.illegal_move_count >= self.illegal_move_truncate and self.illegal_move_truncate != -1:
                done = True
            else:
                done = False
                
            reward = self.illegal_move_reward
            # ========== 新增：非法移动时角落奖励设为0 ==========
            info['corner_reward'] = 0.0

        #print("Am I done? {}".format(done))
        info['highest'] = self.highest()

        # Modified: Gymnasium step returns (obs, reward, terminated, truncated, info)
        # Truncated = False (no time limit/truncation in 2048)
        return stack(self.Matrix), reward, done, False, info

    def reset(self, seed=None, options=None):  # Modified: Add seed/options for gymnasium
        """Reset the environment to initial state (required for gymnasium)"""
        # Modified: Call parent class reset to handle seed properly
        super().reset(seed=seed)
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0
        # 重置计数器
        self.illegal_move_count = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        # Modified: Gymnasium reset returns (obs, info)
        return stack(self.Matrix), {}

    def render(self):
        """渲染环境，符合 Gymnasium 最新规范"""
        if self.render_mode is None:
            # Gymnasium 规范：未指定 render_mode 时调用 render 应返回 None 或报警
            return None

        assert self.render_mode in ("rgb_array", "human")

        # --- 渲染逻辑 (rgb_array) ---
        grey = (128, 128, 128)
        white = (255, 255, 255)
        tile_colour_map = {
            2: (255, 0, 0),
            4: (224, 32, 0),
            8: (192, 64, 0),
            16: (160, 96, 0),
            32: (128, 128, 0),
            64: (96, 160, 0),
            128: (64, 192, 0),
            256: (32, 224, 0),
            512: (0, 255, 0),
            1024: (0, 224, 32),
            2048: (0, 192, 64),
            4096: (0, 160, 96),
        }
        grid_size = self.grid_size

        if self._font is None:
            self._font = ImageFont.load_default(size=30)

        # 创建画布
        canvas = Image.new("RGB", (grid_size * 4, grid_size * 4), color=grey)
        draw = ImageDraw.Draw(canvas)

        for y in range(4):
            for x in range(4):
                val = self.get(y, x)
                if val:
                    color = tile_colour_map.get(val, (0, 128, 128))
                    rect = [x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size]
                    draw.rectangle(rect, fill=color)
                    
                    # --- Pillow 新版迁移：使用 textbbox 代替 textsize ---
                    text = str(val)
                    # 获取文字的边界框 [left, top, right, bottom]
                    bbox = draw.textbbox((0, 0), text, font=self._font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    
                    # 计算居中坐标
                    text_x = x * grid_size + (grid_size - text_w) // 2
                    text_y = y * grid_size + (grid_size - text_h) // 2
                    draw.text((text_x, text_y), text, font=self._font, fill=white)

        # 转换为 numpy 数组 (H, W, C)
        rgb_array = np.array(canvas, dtype=np.uint8)

        if self.render_mode == "rgb_array":
            return rgb_array

        assert self.render_mode == "human"

        import pygame # 延迟导入

        if self.window is None:
            pygame.init()
            pygame.display.init()
            # 窗口大小对应画布大小 (grid_size * 4)
            window_size = self.grid_size * 4
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("2048 Game")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        self.window.blit(surface, (0, 0))
        
        pygame.event.pump()
        pygame.display.update()

        # 控制帧率
        self.clock.tick(self.metadata["render_fps"])
        

    def close(self):
        self._font = None
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board

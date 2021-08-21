import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete

import agent
import babyai_main
from agent import NNBase
from gpt_agent import build_gpt
from utils import init


class Agent(agent.Agent):
    def __init__(self, *args, observation_space, **kwargs):
        spaces = babyai_main.Spaces(*observation_space.spaces)
        super().__init__(
            *args,
            num_directions=spaces.direction.n,
            obs_shape=spaces.image.shape,
            observation_space=observation_space,
            **kwargs
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


def get_size(space: Space):
    if isinstance(space, (Box, MultiDiscrete)):
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return 1
    raise TypeError()


class Base(NNBase):
    def __init__(
        self,
        gpt_size: str,
        hidden_size: int,
        num_directions: int,
        observation_space: Dict,
        randomize_parameters: bool,
        train_ln: bool,
        train_wpe: bool,
    ):
        super().__init__(False, hidden_size, hidden_size)
        self.observation_spaces = babyai_main.Spaces(*observation_space.spaces)
        self.num_directions = num_directions
        self.gpt = build_gpt(gpt_size, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)
        self.embedding_size = self.gpt.config.n_embd

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        h, w, d = self.observation_spaces.image.shape

        self.conv = nn.Sequential(
            init_(nn.Conv2d(d, 32, 4, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.merge = nn.Sequential(
            init_(
                nn.Linear(
                    32 * 2 * 2 + num_directions + self.embedding_size, hidden_size
                )
            ),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        inputs = babyai_main.Spaces(
            *torch.split(
                inputs,
                [get_size(space) for space in self.observation_spaces],
                dim=-1,
            )
        )

        image = inputs.image.reshape(-1, *self.observation_spaces.image.shape).permute(
            0, 3, 1, 2
        )
        image = self.conv(image)
        directions = inputs.direction.squeeze().long()
        directions = F.one_hot(directions, num_classes=self.num_directions)
        mission = self.gpt.forward(inputs.mission.long()).last_hidden_state[:, -1]
        x = torch.cat([image, directions, mission], dim=-1)
        x = self.merge(x)
        return self.critic_linear(x), x, rnn_hxs


class DataParallel(nn.DataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

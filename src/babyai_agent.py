import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from transformers import GPT2Config
from dataclasses import astuple

import agent
from agent import NNBase
from babyai_env import Spaces
from utils import get_gpt_size, init


def get_size(space: Space):
    if isinstance(space, (Box, MultiDiscrete)):
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return 1
    raise TypeError()


class Agent(agent.Agent):
    def __init__(self, *args, observation_space, **kwargs):
        spaces = Spaces(*observation_space.spaces)
        super().__init__(
            *args,
            obs_shape=spaces.image.shape,
            observation_space=observation_space,
            **kwargs
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class Base(NNBase):
    def __init__(
        self,
        embedding_size: str,
        hidden_size: int,
        observation_space: Dict,
    ):
        super().__init__(False, hidden_size, hidden_size)
        self.observation_spaces = Spaces(*observation_space.spaces)
        self.num_directions = self.observation_spaces.direction.n

        self.embedding_size = GPT2Config.from_pretrained(
            get_gpt_size(embedding_size),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        ).n_embd

        self.embeddings = self.build_embeddings()

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
                nn.Linear(32 + self.num_directions + self.embedding_size, hidden_size)
            ),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def build_embeddings(self):
        num_embeddings = int(self.observation_spaces.mission.nvec[0])
        return nn.EmbeddingBag(num_embeddings, self.embedding_size)

    def forward(self, inputs, rnn_hxs, masks):
        inputs = Spaces(
            *torch.split(
                inputs,
                [get_size(space) for space in astuple(self.observation_spaces)],
                dim=-1,
            )
        )

        image = inputs.image.reshape(-1, *self.observation_spaces.image.shape).permute(
            0, 3, 1, 2
        )
        image = self.conv(image)
        directions = inputs.direction.long()
        directions = F.one_hot(directions, num_classes=self.num_directions).squeeze(1)
        mission = self.embed(inputs.mission.long())
        x = torch.cat([image, directions, mission], dim=-1)
        x = self.merge(x)
        return self.critic_linear(x), x, rnn_hxs

    def embed(self, inputs):
        return self.embeddings.forward(inputs)
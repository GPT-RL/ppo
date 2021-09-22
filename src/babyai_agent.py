from dataclasses import astuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from transformers import GPT2Config

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
    def __init__(self, observation_space, **kwargs):
        spaces = Spaces(*observation_space.spaces)
        super().__init__(
            obs_shape=spaces.image.shape, observation_space=observation_space, **kwargs
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class Base(NNBase):
    def __init__(
        self,
        embedding_size: str,
        hidden_size: int,
        observation_space: Dict,
        recurrent: bool,
        second_layer: bool,
    ):
        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.observation_spaces = Spaces(*observation_space.spaces)
        self.num_directions = self.observation_spaces.direction.n
        self.num_actions = self.observation_spaces.action.n

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
        dummy_input = torch.zeros(1, d, h, w)

        self.conv = nn.Sequential(
            init_(nn.Conv2d(d, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        try:
            output = self.conv(dummy_input)
            assert not second_layer
        except RuntimeError:
            self.conv = (
                nn.Sequential(
                    init_(nn.Conv2d(d, 32, 3, stride=2)),
                    nn.ReLU(),
                    init_(nn.Conv2d(32, 32, 3, stride=1)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                if second_layer
                else nn.Sequential(
                    init_(nn.Conv2d(d, 32, 3, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            )
            output = self.conv(dummy_input)

        self.merge = nn.Sequential(
            init_(
                nn.Linear(
                    output.size(-1)
                    + self.num_directions
                    + self.num_actions
                    + self.embedding_size,
                    hidden_size,
                )
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
        return nn.Sequential(
            nn.Embedding(num_embeddings, self.embedding_size),
            nn.GRU(self.embedding_size, self.embedding_size, batch_first=True),
        )

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
        action = inputs.action.long()
        action = F.one_hot(action, num_classes=self.num_actions).squeeze(1)

        mission = self.embed(inputs.mission.long())
        x = torch.cat([image, directions, action, mission], dim=-1)
        x = self.merge(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs

    def embed(self, inputs):
        return self.embeddings.forward(inputs)[1].squeeze(0)

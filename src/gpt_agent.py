from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GPT2Config, GPT2Model

import agent
from agent import NNBase
from distributions import Bernoulli, Categorical, DiagGaussian
from utils import init


class Agent(agent.Agent):
    def __init__(
        self, obs_shape, action_space, save_interval, save_dir, data_parallel, **kwargs
    ):
        nn.Module.__init__(self)

        self.step = 0
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_path = Path(save_dir, "linguistic-analysis.pkl")
        self.save_interval = save_interval
        self.base = Base(obs_shape[0], **kwargs)
        if data_parallel:
            self.base = DataParallel(self.base)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        if (
            self.save_interval is not None
            and self.step % self.save_interval == 0
            and self.save_path is not None
        ):
            torch.save(
                dict(
                    inputs=inputs,
                    perception=self.base.perception(inputs),
                    probs=dist.probs,
                ),
                self.save_path,
            )
        self.step += 1
        return value, action, action_log_probs, rnn_hxs


class GPTCell(nn.Module):
    def __init__(self, context_size, gpt: GPT2Model):
        super().__init__()
        self.gpt = gpt
        self.context_size = context_size

    def forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:  # noqa: F811
        n_embd = self.gpt.config.n_embd
        hx = hx.reshape(-1, self.context_size, (n_embd + 1))
        hx = torch.cat([hx, input.reshape(1, -1, 1)])
        hx = hx[:, 1:]
        inputs_embeds, attention_mask = torch.split(hx, [n_embd, 1], dim=-1)
        output = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return output, hx


class Base(NNBase):
    def __init__(
        self,
        num_inputs,
        gpt_size: str,
        randomize_parameters: bool,
        hidden_size: int,
        action_hidden_size: Optional[int],
        transpose: bool,
        one_layer: bool,
        recurrent=False,
    ):
        super().__init__(recurrent, hidden_size, hidden_size)

        self.transpose = transpose
        gpt_size = "" if gpt_size == "small" else f"-{gpt_size}"
        gpt_size = f"gpt2{gpt_size}"
        self.gpt = (
            GPT2Model(
                GPT2Config.from_pretrained(
                    gpt_size,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            )
            if randomize_parameters
            else GPT2Model.from_pretrained(
                gpt_size,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        )
        # self.rnn = GPTCell(gpt=self.gpt, context_size=num_embeddings)
        # Freeze GPT parameters
        for p in self.gpt.parameters():
            p.requires_grad_(False)
        self.embedding_size = embedding_size = self.gpt.config.n_embd

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.perception = nn.Sequential(
            *(
                [init_(nn.Conv2d(num_inputs, embedding_size, 12, stride=12))]
                if one_layer
                else [
                    init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
                    nn.ReLU(),
                    init_(nn.Conv2d(32, embedding_size, 16, stride=2)),
                ]
            )
        )
        self.action = (
            None
            if action_hidden_size is None
            else nn.Sequential(
                init_(nn.Linear(embedding_size, action_hidden_size)),
                nn.ReLU(),
            )
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        self._output_size = (
            embedding_size if action_hidden_size is None else action_hidden_size
        )
        self.critic_linear = init_(nn.Linear(self._output_size, 1))

        self.train()

    @property
    def recurrent_hidden_state_size(self):
        return self.context_size * (self.embedding_size + 1)  # +1 for mask

    @property
    def output_size(self):
        return self._output_size

    @property
    def initial_hxs(self):
        hx = torch.zeros(1, self.recurrent_hidden_state_size)
        hx = hx.reshape(-1, self.context_size, (self.embedding_size + 1))
        hx[:, :-1, -1] = 1
        return hx

    def forward(self, inputs, rnn_hxs, masks):
        inputs = inputs / 255.0
        # inputs = torch.nn.functional.layer_norm(inputs, normalized_shape=inputs.shape)
        breakpoint()
        perception = self.perception(inputs)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(perception, rnn_hxs, masks)
        else:
            inputs_embeds = (
                perception.reshape(inputs.size(0), self.embedding_size, -1).transpose(
                    2, 1
                )
                if self.transpose
                else perception.reshape(inputs.size(0), -1, self.embedding_size)
            )
            x = self.gpt(inputs_embeds=inputs_embeds).last_hidden_state[:, -1]
            if self.action is not None:
                x = self.action(x)

        return self.critic_linear(x), x, rnn_hxs


class DataParallel(nn.DataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

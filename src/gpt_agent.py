from torch import nn
from transformers import GPT2Config, GPT2Model

import babyai_agent
from utils import get_gpt_size


def build_gpt(gpt_size, randomize_parameters):
    gpt_size = get_gpt_size(gpt_size)
    return (
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


class Agent(babyai_agent.Agent):
    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class GPTEmbed(nn.Module):
    def __init__(
        self,
        embedding_size: str,
        randomize_parameters: bool,
        train_wpe: bool,
        train_ln: bool,
    ):
        super().__init__()
        self.gpt = build_gpt(embedding_size, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

    def forward(self, x, **_):
        return self.gpt.forward(x).last_hidden_state[:, -1]


class Base(babyai_agent.Base):
    def __init__(
        self,
        *args,
        embedding_size: str,
        randomize_parameters: bool,
        train_ln: bool,
        train_wpe: bool,
        **kwargs,
    ):
        self._embedding_size = embedding_size
        self.randomize_parameters = randomize_parameters
        self.train_wpe = train_wpe
        self.train_ln = train_ln
        super().__init__(*args, embedding_size=embedding_size, **kwargs)

    def build_embeddings(self):
        return GPTEmbed(
            embedding_size=self._embedding_size,
            randomize_parameters=self.randomize_parameters,
            train_wpe=self.train_wpe,
            train_ln=self.train_ln,
        )

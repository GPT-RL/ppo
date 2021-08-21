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
        gpt = build_gpt(self._embedding_size, self.randomize_parameters)
        for name, p in gpt.named_parameters():
            requires_grad = (self.train_wpe and "wpe" in name) or (
                self.train_ln and "ln" in name
            )
            p.requires_grad_(requires_grad)
        return gpt

    def embed(self, inputs):
        return self.embeddings.forward(inputs).last_hidden_state[:, -1]

from typing import Literal

import main
from gpt_agent import Agent


class Args(main.Args):
    gpt_size: Literal[
        "small", "medium", "large", "xl"
    ]  # what size of pretrained GPT to use
    num_embeddings: int = (
        1  # How many embeddings should the perception module generate as input for GPT?
    )


class Trainer(main.Trainer):
    @staticmethod
    def make_agent(obs_shape, action_space, args) -> Agent:
        return Agent(
            obs_shape=obs_shape,
            action_space=action_space,
            recurrent=args.recurrent_policy,
            gpt_size=args.gpt_size,
            num_embeddings=args.num_embeddings,
        )


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

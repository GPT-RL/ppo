from typing import Literal, Optional

import main
from gpt_agent import Agent


class Args(main.Args):
    gpt_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    linguistic_analysis_save_interval: Optional[
        str
    ] = None  # path to save linguistic analysis data
    num_embeddings: int = (
        1  # How many embeddings should the perception module generate as input for GPT?
    )
    randomize_parameters: bool = False


class Trainer(main.Trainer):
    @staticmethod
    def make_agent(obs_shape, action_space, args: Args) -> Agent:
        return Agent(
            obs_shape=obs_shape,
            action_space=action_space,
            recurrent=args.recurrent_policy,
            gpt_size=args.gpt_size,
            num_embeddings=args.num_embeddings,
            randomize_parameters=args.randomize_parameters,
            save_dir=args.save_dir,
            save_interval=args.linguistic_analysis_save_interval,
        )


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

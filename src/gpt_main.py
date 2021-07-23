from typing import Literal, Optional
import logging
from pprint import pformat

import torch
from transformers import GPT2Model

import main
import utils
from gpt_agent import Agent


class Args(main.Args):
    gpt_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    num_embeddings: int = (
        1  # How many embeddings should the perception module generate as input for GPT?
    )
    linguistic_analysis_path: Optional[
        str
    ] = None  # path to save linguistic analysis data
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
            save_interval=args.save_interval,
            save_path=args.linguistic_analysis_path,
        )

    @staticmethod
    def save(agent: Agent, args, envs):
        gpt: GPT2Model = agent.base.gpt
        gpt_params = {f"base.gpt.{k}": v for k, v in gpt.named_parameters()}
        non_gpt_params = {
            k: v for k, v in agent.named_parameters() if k not in gpt_params
        }
        logging.info("Saving parameters:")
        logging.info(pformat([*non_gpt_params]))
        torch.save(
            dict(
                **non_gpt_params,
                obs_rms=getattr(utils.get_vec_normalize(envs), "obs_rms", None),
            ),
            args.save_path,
        )
        logging.info(f"Saved to {args.save_path}")


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

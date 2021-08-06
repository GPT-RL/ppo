import logging
from pathlib import Path
from pprint import pformat
from typing import Literal, Optional

import torch
from transformers import GPT2Model

import main
import utils
from gpt_agent import Agent


class Args(main.Args):
    action_hidden_size: Optional[int] = None
    data_parallel: bool = True
    gpt_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    kernel: int = 16
    linguistic_analysis_save_interval: Optional[
        str
    ] = None  # path to save linguistic analysis data
    randomize_parameters: bool = False
    stride: int = 8
    transpose: bool = True
    one_layer: bool = False


class Trainer(main.Trainer):
    @staticmethod
    def make_agent(obs_shape, action_space, args: Args) -> Agent:
        return Agent(
            action_hidden_size=args.action_hidden_size,
            action_space=action_space,
            data_parallel=args.data_parallel,
            gpt_size=args.gpt_size,
            hidden_size=args.hidden_size,
            obs_shape=obs_shape,
            one_layer=args.one_layer,
            randomize_parameters=args.randomize_parameters,
            recurrent=args.recurrent_policy,
            save_interval=args.linguistic_analysis_save_interval,
            save_dir=args.save_dir,
            transpose=args.transpose,
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
        save_path = Path(args.save_dir, f"checkpoint.pkl")
        torch.save(
            dict(
                **non_gpt_params,
                obs_rms=getattr(utils.get_vec_normalize(envs), "obs_rms", None),
            ),
            save_path,
        )
        logging.info(f"Saved to {save_path}")


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

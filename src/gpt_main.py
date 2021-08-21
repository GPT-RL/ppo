import logging
from pathlib import Path
from pprint import pformat
from typing import Optional

import torch
from transformers import GPT2Model

import babyai_main
from envs import VecPyTorch
from gpt_agent import Agent


class Args(babyai_main.Args):
    data_parallel: bool = True
    linguistic_analysis_save_interval: Optional[
        str
    ] = None  # path to save linguistic analysis data
    randomize_parameters: bool = False
    train_ln: bool = True
    train_wpe: bool = False


class Trainer(babyai_main.Trainer):
    @staticmethod
    def make_agent(envs: VecPyTorch, args: Args) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=action_space,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            randomize_parameters=args.randomize_parameters,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
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
        torch.save(non_gpt_params, save_path)
        logging.info(f"Saved to {save_path}")


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

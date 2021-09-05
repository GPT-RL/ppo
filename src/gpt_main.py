import logging
from typing import Optional

import torch

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
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: Args) -> Agent:
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
            recurrent=cls.recurrent(args),
        )

    @staticmethod
    def save(
        agent: Agent,
        save_path,
        args,
    ):
        trainable = {k for k, v in agent.named_parameters() if v.requires_grad}
        trainable_params = {
            k: v for k, v in agent.state_dict().items() if k in trainable
        }
        torch.save(trainable_params, save_path)
        logging.info(f"Saved the following modules to {save_path}:")
        for p in trainable_params:
            logging.info(p)

    @staticmethod
    def load(agent, load_path):
        loaded = torch.load(load_path)
        for name, module in agent.named_modules():
            name_ = f"{name}."
            parameters = {}
            for k, v in loaded.items():
                if k.startswith(name_):
                    parameters[k[len(name_) :]] = v
            if parameters:
                try:
                    module.load_state_dict(parameters)
                    logging.info(f"Loaded parameters into {name}.")
                except RuntimeError:
                    pass


if __name__ == "__main__":
    Trainer.main(Args().parse_args())

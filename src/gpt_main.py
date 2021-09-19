import logging
from typing import Literal, Optional, cast

import torch
from tap import Tap

import babyai_main
from envs import VecPyTorch
from gpt_agent import Agent
from main import RUN_OR_SWEEP, configure_logger_args


class GptArgs(Tap):
    randomize_parameters: bool = False
    train_ln: bool = True
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="second_args")
        configure_logger_args(self)


class Args(babyai_main.Args):
    def configure(self) -> None:
        self.add_subparsers(dest="first_args")
        self.add_subparser("gpt", GptArgs)
        configure_logger_args(self)


class ArgsType(babyai_main.Args, GptArgs):
    first_args: Optional[Literal[RUN_OR_SWEEP, "gpt"]]
    second_args: Optional[RUN_OR_SWEEP]


class Trainer(babyai_main.Trainer):
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
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

    @classmethod
    def load_config(cls, args):
        return args


def main():
    args = Args().parse_args()
    if args.config:
        args = babyai_main.Trainer.load_config(args)
    args = cast(ArgsType, args)
    logging.getLogger().setLevel(args.log_level)
    if args.first_args == "gpt":
        logging.info(f"Using {args.embedding_size} GPT architecture.")
        args.logger_args = args.second_args
        Trainer().main(args)
    else:
        logging.info(f"Using baseline architecture.")
        args.logger_args = args.first_args
        babyai_main.Trainer().main(args)


if __name__ == "__main__":
    main()

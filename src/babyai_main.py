from typing import Literal

from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import (
    Env,
    ObsSpaceWrapper,
    ZeroOneRewardWrapper,
    RolloutsWrapper,
    TokenizerWrapper,
    get_train_and_test_objects,
)
from envs import VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "GoToLocal"  # env ID for gym
    room_size: int = 5
    num_dists: int = 2
    strict: bool = True


class Trainer(main.Trainer):
    @staticmethod
    def make_agent(envs: VecPyTorch, args: Args) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=action_space,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
        )

    @staticmethod
    def make_env(env_id, seed, rank, allow_early_resets, *args, **kwargs):
        def _thunk():
            tokenizer = kwargs.pop("tokenizer")
            env = Env(*args, seed=seed + rank, **kwargs)
            env = ZeroOneRewardWrapper(env)
            env = ObsSpaceWrapper(env)
            env = TokenizerWrapper(
                env, tokenizer=tokenizer, longest_mission="pick up a blue ball"
            )
            env = RolloutsWrapper(env)

            env = Monitor(env, allow_early_resets=allow_early_resets)

            return env

        return _thunk

    @classmethod
    def make_vec_envs(cls, args, device, **kwargs):
        test_objects, train_objects = get_train_and_test_objects()
        assert len(test_objects) >= 3
        test = kwargs.pop("test")
        goal_objects = test_objects if test else train_objects

        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))
        return super().make_vec_envs(
            args,
            device,
            room_size=args.room_size,
            num_dists=args.num_dists,
            goal_objects=goal_objects,
            strict=args.strict,
            tokenizer=tokenizer,
            **kwargs,
        )


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

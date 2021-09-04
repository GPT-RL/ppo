from typing import Literal

from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import (
    FullyObsWrapper,
    PickupEnv,
    PickupEnvRoomObjects,
    PickupRedEnv,
    PlantAnimalWrapper,
    RolloutsWrapper,
    SequenceEnv,
    SequenceSynonymWrapper,
    SynonymWrapper,
    TokenizerWrapper,
    ZeroOneRewardWrapper,
)
from envs import RenderWrapper, VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "GoToLocal"  # env ID for gym
    room_size: int = 5
    strict: bool = True
    record_interval: int = 200


class InvalidEnvIdError(RuntimeError):
    pass


class Trainer(main.Trainer):
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: Args) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=action_space,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
        )

    @staticmethod
    def recurrent(args: Args):
        return "sequence" in args.env

    @staticmethod
    def make_env(
        env_id, seed, rank, allow_early_resets, render: bool = False, *args, **kwargs
    ):
        def _thunk():
            tokenizer = kwargs.pop("tokenizer")
            if env_id == "pickup":
                env = PickupEnv(*args, seed=seed + rank, num_dists=1, **kwargs)
                longest_mission = "pick up the red ball"
            elif env_id == "pickup-synonyms":
                env = PickupRedEnv(*args, seed=seed + rank, **kwargs)
                env = SynonymWrapper(env)
                longest_mission = "pick-up the crimson phone"
            elif env_id == "plant-animal":
                objects = {*PlantAnimalWrapper.replacements.keys()}
                test_objects = {
                    PlantAnimalWrapper.purple_animal,
                    PlantAnimalWrapper.black_plant,
                }
                room_objects = test_objects if test else objects - test_objects
                room_objects = [o.split() for o in room_objects]
                room_objects = [(t, c) for (c, t) in room_objects]
                kwargs.update(room_objects=room_objects)
                env = PickupEnvRoomObjects(*args, seed=seed + rank, **kwargs)
                env = PlantAnimalWrapper(env)
                longest_mission = "pick up the grasshopper"
            else:
                del kwargs["goal_objects"]
                if env_id == "sequence-paraphrases":
                    env = SequenceEnv(*args, seed=seed + rank, **kwargs)
                    env = SequenceSynonymWrapper(env, test=kwargs["test"])
                    longest_mission = (
                        "pick up the red ball, having already picked up the red key"
                    )
                elif env_id == "sequence":
                    env = SequenceEnv(*args, seed=seed + rank, **kwargs)
                    longest_mission = "pick up the red ball, then pick up the red key"
                else:
                    raise InvalidEnvIdError()

            env = FullyObsWrapper(env)
            env = ZeroOneRewardWrapper(env)
            env = TokenizerWrapper(
                env,
                tokenizer=tokenizer,
                longest_mission=longest_mission,
            )
            env = RolloutsWrapper(env)

            env = Monitor(env, allow_early_resets=allow_early_resets)
            if render:
                env = RenderWrapper(env)

            return env

        return _thunk

    @classmethod
    def make_vec_envs(cls, args, device, **kwargs):
        # assert len(test_objects) >= 3
        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))
        return super().make_vec_envs(
            args,
            device,
            room_size=args.room_size,
            tokenizer=tokenizer,
            strict=args.strict,
            **kwargs,
        )


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

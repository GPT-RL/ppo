from collections import defaultdict
from typing import Literal

import gym
import numpy as np
from gym.spaces import Box, Tuple
from gym.spaces import Dict, MultiDiscrete
from gym.wrappers import TimeLimit
from gym_minigrid.minigrid import COLOR_NAMES
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import GoToLocalEnv, ObsSpaceWrapper, Spaces
from babyai_env import all_object_types
from envs import VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "GoToLocal"  # env ID for gym
    room_size: int = 8
    num_dists: int = 8
    max_episode_steps: int = 20


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.original_observation_space = Tuple(Spaces(**self.observation_space.spaces))
        image_space = spaces["image"]
        # direction_space = spaces["direction"]
        mission_space = spaces["mission"]
        self.observation_space = Box(
            shape=[np.prod(image_space.shape) + 1 + np.prod(mission_space.shape)],
            low=-np.inf,
            high=np.inf,
        )

    def observation(self, observation):
        return np.concatenate(
            Spaces(
                image=observation["image"].flatten(),
                direction=np.array([observation["direction"]]),
                mission=observation["mission"],
            )
        )


class TokenizerWrapper(gym.ObservationWrapper):
    def __init__(self, env, tokenizer: GPT2Tokenizer, longest_mission: str):
        self.tokenizer: GPT2Tokenizer = tokenizer
        encoded = tokenizer.encode(longest_mission)
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.observation_space = Dict(
            spaces=dict(**spaces, mission=MultiDiscrete([50257 for _ in encoded]))
        )

    def observation(self, observation):
        mission = self.tokenizer.encode(observation["mission"])
        observation.update(mission=mission)
        return observation


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
            max_episode_steps = kwargs.pop("max_episode_steps")
            env = GoToLocalEnv(*args, seed=seed + rank, **kwargs)
            env = ObsSpaceWrapper(env)
            env = TokenizerWrapper(
                env, tokenizer=tokenizer, longest_mission="go to a blue ball"
            )
            env = RolloutsWrapper(env)

            # if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
            #
            env = Monitor(env, allow_early_resets=allow_early_resets)

            return env

        return _thunk

    @classmethod
    def make_vec_envs(cls, args, device, **kwargs):
        object_types = [*all_object_types()]
        np.random.shuffle(object_types)

        def pairs():
            colors = [*COLOR_NAMES]
            types = ["key", "ball", "box"]

            np.random.shuffle(colors)
            np.random.shuffle(types)

            colors_to_types = {color: [*types] for color in colors}

            # yield all colors
            for color, types in colors_to_types.items():
                yield types.pop(), color

            # reverse colors_to_types
            types_to_colors = defaultdict(set)
            for color, types in colors_to_types.items():
                for type in types:
                    types_to_colors[type].add(color)

            # yield all types
            for type, [*colors] in types_to_colors.items():
                yield type, colors.pop()

            # yield remaining
            remaining = [
                (type, color)
                for type, colors in types_to_colors.items()
                for color in colors
            ]
            np.random.shuffle(remaining)
            yield from remaining

        pairs = [*pairs()]
        three_quarters = round(3 / 4 * len(pairs))
        train_objects = pairs[:three_quarters]
        test_objects = pairs[three_quarters:]
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
            max_episode_steps=args.max_episode_steps,
            tokenizer=tokenizer,
            **kwargs,
        )


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

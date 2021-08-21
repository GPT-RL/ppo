from collections import defaultdict
from typing import NamedTuple, TypeVar

import gym
import numpy as np
from gym.spaces import Box, Tuple
from gym_minigrid.minigrid import COLOR_NAMES

import main
from babyai_env import all_object_types


class Args(main.Args):
    env: str = "GoToLocal"  # env ID for gym
    room_size: int = 8
    num_dists: int = 8
    max_episode_steps: int = 20


T = TypeVar("T")  # Declare type variable


class Spaces(NamedTuple):
    image: T
    direction: T
    mission: T


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


class Trainer(main.Trainer):
    @staticmethod
    def make_env(env_id, seed, rank, allow_early_resets, *args, **kwargs):
        def _thunk():
            raise NotImplementedError

            # if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimit(env, max_episode_steps=20)
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
        return super().make_vec_envs(
            args,
            device,
            room_size=args.room_size,
            num_dists=args.num_dists,
            goal_objects=goal_objects,
            max_episode_steps=args.max_episode_steps,
            **kwargs,
        )


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())

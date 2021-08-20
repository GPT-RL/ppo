import numpy as np
from gym_minigrid.minigrid import COLOR_NAMES

import main
from babyai_env import all_object_types


class Args(main.Args):
    env: str = "GoToLocal"  # env ID for gym


class Trainer(main.Trainer):
    @classmethod
    def make_vec_envs(cls, args, device, **kwargs):
        object_types = set(all_object_types())
        np.random.shuffle(object_types)

        def train_objects():
            seen_types = set()
            seen_colors = set()
            for i, (object_type, color) in enumerate(object_types):
                seen_types.add(object_type)
                seen_colors.add(color)
                yield object_type, color
                if all(
                    [
                        seen_types == {"key", "ball", "box"},
                        seen_colors == set(COLOR_NAMES),
                        i > 3 / 4 * len(object_types),
                    ]
                ):
                    return

        train_object = set(train_objects())
        test_objects = object_types - train_object
        assert len(test_objects) >= 3
        test = kwargs.pop("test")
        goal_objects = test_objects if test else train_objects
        return super().make_vec_envs(args, device, goal_objects=goal_objects, **kwargs)

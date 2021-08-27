from collections import defaultdict
from dataclasses import astuple, dataclass
from typing import TypeVar

import babyai
import gym
import gym_minigrid
import numpy as np
from babyai.levels.verifier import ObjDesc, PickupInstr
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import COLOR_NAMES, WorldObj
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from transformers import GPT2Tokenizer
from gym_minigrid.window import Window


def get_train_and_test_objects():
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
        # remaining = [
        #     (type, color)
        #     for type, colors in types_to_colors.items()
        #     for color in colors
        # ]
        # np.random.shuffle(remaining)
        # yield from remaining

    # train_objects = [*pairs()][:1]
    train_objects = [("ball", "red")]
    test_objects = [x for x in all_object_types() if x not in set(train_objects)]
    return test_objects, train_objects


def all_object_types():
    for color in COLOR_NAMES:
        for object_type in ["key", "ball", "box"]:
            yield object_type, color


class Env(babyai.levels.iclr19_levels.Level_GoToLocal):
    def __init__(
        self,
        goal_objects,
        room_size: int,
        num_dists: int,
        seed: int,
        strict: bool,
    ):
        self.strict = strict
        self.goal_objects = goal_objects
        super().__init__(seed=seed, room_size=room_size, num_dists=num_dists)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj, *_ = objs
        assert (obj.type, obj.color) in self.goal_objects, (
            obj.type,
            obj.color,
            self.goal_objects,
        )
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color), strict=self.strict)

    def can_be_goal(self, obj: WorldObj):
        return (obj.type, obj.color) in self.goal_objects

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True):
        """
        Add random objects that can potentially distract/confuse the agent.
        """

        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            if not dists:
                obj = self._rand_elem(self.goal_objects)
            else:
                color = self._rand_elem(COLOR_NAMES)
                type = self._rand_elem(["key", "ball", "box"])
                obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i is None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j is None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists


T = TypeVar("T")  # Declare type variable


@dataclass
class Spaces:
    image: T
    direction: T
    mission: T


class ZeroOneRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return int(bool(reward > 0))


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.original_observation_space = Tuple(
            astuple(Spaces(**self.observation_space.spaces))
        )
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
            astuple(
                Spaces(
                    image=observation["image"].flatten(),
                    direction=np.array([observation["direction"]]),
                    mission=observation["mission"],
                )
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


class FullyObsWrapper(gym_minigrid.wrappers.FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Dict(
            spaces=dict(
                **self.observation_space.spaces,
                direction=Discrete(4),
            )
        )

    def observation(self, obs):
        direction = obs["direction"]
        obs = super().observation(obs)
        obs["direction"] = direction
        return obs


def main(args: "Args"):
    def redraw(img):
        if not args.agent_view:
            img = env.render("rgb_array", tile_size=args.tile_size)
        window.show_img(img)

    def reset():
        obs = env.reset()

        if hasattr(env, "mission"):
            print("Mission: %s" % env.mission)
            window.set_caption(env.mission)

        redraw(obs)

    def step(action):
        obs, reward, done, info = env.step(action)
        print("step=%s, reward=%.2f" % (env.step_count, reward))

        if done:
            print("done!")
            reset()
        else:
            redraw(obs)

    def key_handler(event):
        print("pressed", event.key)

        if event.key == "escape":
            window.close()
            return

        if event.key == "backspace":
            reset()
            return

        if event.key == "left":
            step(env.actions.left)
            return
        if event.key == "right":
            step(env.actions.right)
            return
        if event.key == "up":
            step(env.actions.forward)
            return

        # Spacebar
        if event.key == " ":
            step(env.actions.toggle)
            return
        if event.key == "pageup":
            step(env.actions.pickup)
            return
        if event.key == "pagedown":
            step(env.actions.drop)
            return

        if event.key == "enter":
            step(env.actions.done)
            return

    train, test = get_train_and_test_objects()
    goal_objects = test if args.test else train
    env = Env(
        goal_objects=goal_objects,
        room_size=args.room_size,
        num_dists=args.num_dists,
        seed=args.seed,
        strict=args.strict,
    )
    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
    window = Window("gym_minigrid")
    window.reg_key_handler(key_handler)
    reset()
    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    import babyai_main

    class Args(babyai_main.Args):
        tile_size: int = 32
        agent_view: bool = False
        test: bool = False

    main(Args().parse_args())

from collections import defaultdict
from dataclasses import astuple, dataclass
from typing import Generator, Optional, TypeVar

import colors
import gym
import gym_minigrid
import numpy as np
from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import ObjDesc, PickupInstr
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import COLOR_NAMES, OBJECT_TO_IDX, WorldObj
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from transformers import GPT2Tokenizer


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
    # test_objects = [x for x in all_object_types() if x not in set(train_objects)]
    test_objects = [("key", "yellow")]
    return test_objects, train_objects


def all_object_types():
    for color in COLOR_NAMES:
        for object_type in ["key", "ball", "box"]:
            yield object_type, color


class Agent(WorldObj):
    def render(self, r):
        pass


class Env(RoomGridLevel):
    def __init__(
        self,
        goal_objects,
        room_size: int,
        num_dists: int,
        seed: int,
        strict: bool,
    ):
        self.strict = strict
        self.goal_object, *_ = self.goal_objects = goal_objects
        self.num_dists = num_dists
        self.__reward = None
        self.__done = None
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed,
        )

    def step(self, action):
        s, self.__reward, self.__done, i = super().step(action)
        return s, self.__reward, self.__done, i

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        goal_object = self._rand_elem(self.goal_objects)
        self.add_object(0, 0, *goal_object)
        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(*goal_object))

    def row_objs(self, i: int) -> Generator[Optional[WorldObj], None, None]:
        for j in range(self.width):
            if np.all(self.agent_pos == (i, j)):
                yield Agent(color="blue", type="agent")
            else:
                yield self.grid.get(i, j)

    def row_strings(self, i: int) -> Generator[str, None, None]:
        for obj in self.row_objs(i):
            if obj is None:
                string = ""
            elif isinstance(obj, Agent):
                if self.agent_dir == 0:
                    string = "v"
                elif self.agent_dir == 1:
                    string = ">"
                elif self.agent_dir == 2:
                    string = "^"
                elif self.agent_dir == 3:
                    string = "<"
                else:
                    raise RuntimeError(f"invalid agent dir: {self.agent_dir}")
            else:
                string = obj.type

            string = f"{string:<{self.max_string_length}}"
            if obj is not None:
                string = colors.color(string, obj.color)
            yield string + "\033[0m"

    @property
    def max_string_length(self):
        return max(map(len, OBJECT_TO_IDX)) + 1

    def row_string(self, i: int):
        return "|".join(self.row_strings(i))

    def horizontal_separator_string(self):
        return "-" * ((self.max_string_length + 1) * self.grid.width - 1)

    def render_string(self):
        yield self.row_string(0)
        for i in range(1, self.grid.height):
            yield self.horizontal_separator_string()
            yield self.row_string(i)

    def render(self, *args, **kwargs):
        for string in self.render_string():
            print(string)
        print(self.mission)
        print("Reward:", self.__reward)
        print("Done:", self.__done)
        input("Press enter to coninue.")


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

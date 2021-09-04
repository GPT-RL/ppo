import re
import typing
from abc import ABC
from dataclasses import astuple, dataclass
from itertools import chain, cycle, islice
from typing import Callable, Generator, List, Optional, TypeVar

import babyai.levels.verifier
import gym
import gym_minigrid
import numpy as np
from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import (
    ActionInstr,
    AfterInstr,
    GoToInstr,
    ObjDesc,
    PickupInstr,
)
from colors import color as ansi_color
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import OBJECT_TO_IDX, WorldObj
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from transformers import GPT2Tokenizer


@dataclass
class TrainTest:
    train: list
    test: list


TYPES = ["key", "ball", "box"]


class Agent(WorldObj):
    def render(self, r):
        pass


class RenderEnv(RoomGridLevel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__reward = None
        self.__done = None

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
                    breakpoint()
                    raise RuntimeError
            else:
                string = obj.type

            string = f"{string:<{self.max_string_length}}"
            if obj is not None:
                string = ansi_color(string, obj.color)
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

    def render(self, mode="human", pause=True, **kwargs):
        if mode == "human":
            for string in self.render_string():
                print(string)
            print(self.mission)
            print("Reward:", self.__reward)
            print("Done:", self.__done)
            if pause:
                input("Press enter to coninue.")
        else:
            return super().render(mode=mode, **kwargs)

    def step(self, action):
        s, self.__reward, self.__done, i = super().step(action)
        return s, self.__reward, self.__done, i


class ToggleInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, obj_desc, strict=False):
        super().__init__()
        self.desc = obj_desc
        self.strict = strict

    def surface(self, env):
        return "toggle " + self.desc.surface(env)

    def reset_verifier(self, env):
        super().reset_verifier(env)

        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # Only verify when the pickup action is performed
        if action != self.env.actions.toggle:
            return "continue"

        # For each object position
        for pos in self.desc.obj_poss:
            # If the agent is next to (and facing) the object
            if np.array_equal(pos, self.env.front_pos):
                return "success"

        if self.strict:
            return "failure"  # not allowed to toggle except in front of correct object

        return "continue"


class ToggleEnv(RenderEnv):
    def __init__(
        self,
        goal_objects: typing.Iterable[typing.Tuple[str, str]],
        room_size: int,
        seed: int,
        strict: bool,
        num_dists: int = 1,
    ):
        self.strict = strict
        self.goal_object, *_ = self.goal_objects = list(goal_objects)
        self.num_dists = num_dists
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed,
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        goal_object = self._rand_elem(self.goal_objects)
        self.add_object(0, 0, *goal_object)
        self.check_objs_reachable()
        self.instrs = ToggleInstr(ObjDesc(*goal_object), strict=self.strict)


class PickupEnv(RenderEnv):
    def __init__(
        self,
        goal_objects: typing.Iterable[typing.Tuple[str, str]],
        room_size: int,
        seed: int,
        strict: bool,
        num_dists: int = 1,
    ):
        self.strict = strict
        self.goal_object, *_ = self.goal_objects = list(goal_objects)
        self.num_dists = num_dists
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed,
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        goal_object = self._rand_elem(self.goal_objects)
        self.add_object(0, 0, *goal_object)
        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(*goal_object), strict=self.strict)


class PickupEnvRoomObjects(PickupEnv):
    def __init__(
        self,
        *args,
        room_objects: typing.Iterable[typing.Tuple[str, str]],
        **kwargs,
    ):
        self.room_objects = room_objects
        kwargs.update(goal_objects=room_objects)
        super().__init__(*args, **kwargs)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        goal_object = self._rand_elem(self.goal_objects)
        self.add_object(0, 0, *goal_object)
        objects = {*self.room_objects} - {goal_object}
        for _ in range(self.num_dists):
            obj = self._rand_elem(objects)
            self.add_object(0, 0, *obj)
        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(*goal_object), strict=self.strict)


COLOR = "red"


class PickupRedEnv(PickupEnv):
    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        for kind in TYPES:
            self.add_object(0, 0, kind=kind, color=COLOR)
        goal_type = self._rand_elem(TYPES)
        self.check_objs_reachable()
        self.instrs = PickupInstr(
            ObjDesc(type=goal_type, color=COLOR), strict=self.strict
        )


class BeforeInstr(babyai.levels.verifier.BeforeInstr):
    def verify(self, action):
        if self.a_done == "success":
            self.b_done = self.instr_b.verify(action)

            if self.b_done == "failure":
                return "failure"

            if self.b_done == "success":
                return "success"
        else:
            self.a_done = self.instr_a.verify(action)
            if self.a_done == "failure":
                return "failure"

            if self.a_done == "success":
                return "continue"

            # In strict mode, completing b first means failure
            if self.strict:
                if self.instr_b.verify(action) == "success":
                    return "failure"

        return "continue"


class SequenceEnv(RenderEnv):
    def __init__(self, *args, strict: bool, **kwargs):
        self.strict = strict
        super().__init__(*args, **kwargs)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        color = "red"
        goal1 = self._rand_elem(TYPES)
        goal2 = self._rand_elem(set(TYPES) - {goal1})

        for kind in [goal1, goal2]:
            self.add_object(0, 0, kind=kind, color=color)

        instr1 = ToggleInstr(ObjDesc(type=goal1, color=color), strict=self.strict)
        instr2 = ToggleInstr(ObjDesc(type=goal2, color=color), strict=self.strict)
        self.check_objs_reachable()
        self.instrs = (
            BeforeInstr(instr1, instr2, strict=True)
            # if self.np_random.choice(2)
            # else AfterInstr(instr2, instr1, strict=True)
        )


T = TypeVar("T")  # Declare type variable


@dataclass
class Spaces:
    image: T
    direction: T
    mission: T


class PlantAnimalWrapper(gym.ObservationWrapper):
    green_animal = "green box"
    orange_animal = "yellow box"
    green_plant = "green ball"
    orange_plant = "yellow ball"
    white_animal = "grey box"
    white_plant = "grey ball"
    purple_animal = "purple box"
    purple_plant = "purple ball"
    # pink_animal = "pink box"
    # pink_plant = "pink ball"
    black_animal = "blue box"
    black_plant = "blue ball"
    red_animal = "red box"
    red_plant = "red ball"
    replacements = {
        red_animal: [
            "rooster",
            "lobster",
            "crab",
            "ladybug",
            "cardinal",
        ],
        red_plant: [
            "cherry",
            "tomato",
            "chili",
            "apple",
            "raspberry",
            "cranberry",
            "strawberry",
            "pomegranate",
            "radish",
            "beet",
            "rose",
        ],
        black_animal: [
            "gorilla",
            "crow",
            "panther",
            "raven",
            "bat",
        ],
        black_plant: ["black plant"],
        # pink_animal: ["flamingo", "pig"],
        # pink_plant: ["lychee", "dragonfruit"],
        purple_animal: ["purple animal"],
        purple_plant: [
            "grape",
            "eggplant",
            "plum",
            "shallot",
            "lilac",
        ],
        white_animal: [
            "polar bear",
            "swan",
            "ermine",
            "sheep",
            "seagull",
        ],
        white_plant: [
            "coconut",
            "cauliflower",
            "onion",
            "garlic",
        ],
        green_animal: [
            "iguana",
            "frog",
            "grasshopper",
            "turtle",
            "mantis",
            "lizard",
            "caterpillar",
        ],
        green_plant: [
            "lime",
            "kiwi",
            "broccoli",
            "lettuce",
            "kale",
            "spinach",
            "avocado",
            "cucumber",
            "basil",
            "pea",
            "arugula",
            "celery",
        ],
        orange_animal: [
            "tiger",
            "lion",
            "orangutan",
            "goldfish",
            "clownfish",
            "fox",
        ],
        orange_plant: [
            "peach",
            "yam",
            "tangerine",
            "carrot",
            "papaya",
            "clementine",
            "kumquat",
            "pumpkin",
            "marigold",
        ],
    }

    def render(self, mode="human", pause=True, **kwargs):
        self.env.render(pause=False)
        print(self._mission)
        if pause:
            input("Press enter to coninue.")

    def observation(self, observation):
        self._mission: str = observation["mission"]
        for k, v in self.replacements.items():
            if k in self._mission:
                replacement = self.np_random.choice(v)
                self._mission = self._mission.replace(k, replacement)

        observation["mission"] = self._mission
        return observation


class SynonymWrapper(gym.ObservationWrapper):
    synonyms = {
        "pick-up": ["take", "grab", "get"],
        "red": ["crimson"],
        "box": ["carton", "package"],
        "ball": ["sphere", "globe"],
        "phone": ["cell", "mobile"],
    }

    def observation(self, observation):
        mission: str = observation["mission"]
        mission = mission.replace("key", "phone")
        mission = mission.replace("pick up", "pick-up")

        def new_mission():
            for word in mission.split():
                choices = [word, *self.synonyms.get(word, [])]
                yield self.np_random.choice(choices)

        mission = " ".join(new_mission())
        observation["mission"] = mission
        return observation


class InvalidInstructionError(RuntimeError):
    pass


class SequenceSynonymWrapper(gym.ObservationWrapper):
    def __init__(self, env, test: bool):
        super().__init__(env)
        self.test = test

    def observation(self, observation):
        mission = observation["mission"]

        def after(instr1: str, instr2: str):
            return f"{instr2} after you {instr1}"

        def after_reverse(instr1: str, instr2: str):
            return f"After you {instr1}, {instr2}"

        def before(instr1: str, instr2: str):
            return f"{instr1} before you {instr2}"

        def before_reverse(instr1: str, instr2: str):
            return f"Before you {instr2}, {instr1}"

        def then(instr1: str, instr2: str):
            return f"{instr1}, then {instr2}"

        def _next(instr1: str, instr2: str):
            return f"{instr1}. Next, {instr2}"

        def having(instr1: str, instr2: str):
            return f"{instr2}, having already {past(instr1)}"

        wordings = [after, after_reverse, before, before_reverse, then, _next, having]

        def past(instr: str):
            return instr.replace("pick", "picked")

        match = re.match(r"(.*), then (.*)", mission)
        if not match:
            match = re.match(r"(.*) after you (.*)", mission)
            if not match:
                breakpoint()

        wording: Callable[[str, str], str] = (
            before if self.test else self.np_random.choice(wordings)
        )
        mission = wording(*match.group(1, 2))
        observation["mission"] = mission

        return observation


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
        length = len(self.observation_space.spaces["mission"].nvec)
        eos = self.tokenizer.eos_token_id
        mission = [*islice(chain(mission, cycle([eos])), length)]
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

    objects = {*PlantAnimalWrapper.replacements.keys()}
    test_objects = {
        PlantAnimalWrapper.purple_animal,
        PlantAnimalWrapper.black_plant,
    }
    env = SequenceEnv(
        seed=args.seed, num_cols=1, num_rows=1, room_size=args.room_size, strict=True
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

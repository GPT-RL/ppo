import abc
import itertools
import re
import typing
from abc import ABC
from dataclasses import astuple, dataclass
from itertools import chain, cycle, islice
from typing import Callable, Generator, Optional, TypeVar

import gym
import gym_minigrid
import numpy as np
from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import (
    ActionInstr,
    BeforeInstr,
    ObjDesc,
    PickupInstr,
    GoToInstr,
)
from colors import color as ansi_color
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import COLOR_NAMES, MiniGridEnv, OBJECT_TO_IDX, WorldObj
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from transformers import GPT2Tokenizer

T = TypeVar("T")  # Declare type variable


@dataclass
class Spaces:
    image: T
    direction: T
    mission: T
    action: T


@dataclass
class TrainTest:
    train: list
    test: list


COLORS = [*COLOR_NAMES][:3]
TYPES = ["key", "ball", "box"]


class Agent(WorldObj):
    def render(self, r):
        pass


class RenderEnv(RoomGridLevel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__reward = None
        self.__done = None
        self.__action = None

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
            print(
                "Action:",
                None
                if self.__action is None
                else MiniGridEnv.Actions(self.__action).name,
            )
            if pause:
                input("Press enter to continue.")
        else:
            return super().render(mode=mode, **kwargs)

    def step(self, action):
        self.__action = action
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


class LocDesc:
    """
    Description of a set of objects in an environment
    """

    def __init__(self, grid: RoomGrid, i: int, j: int):
        assert 1 <= i < grid.height - 1
        assert 1 <= j < grid.width - 1
        self.i = i
        self.j = j

    def __repr__(self):
        return f"({self.i}, {self.j})"

    def surface(self):
        """
        Generate a natural language representation of the object description
        """
        return repr(self)

    @property
    def array(self):
        return np.array([self.i, self.j])

    @staticmethod
    def find_matching_objs(*args, **kwargs):
        return [], []


class GoToLoc(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, loc_desc: LocDesc):
        self.desc = loc_desc
        super().__init__()

    def surface(self, env):
        return "go to " + self.desc.surface()

    def reset_verifier(self, env: MiniGridEnv):
        super().reset_verifier(env)

    def verify_action(self, *args, **kwargs):
        # Only verify when the pickup action is performed
        if np.array_equal(self.env.agent_pos, self.desc.array):
            return "success"

        return "continue"


class GoToObjEnv(RenderEnv):
    def __init__(
        self,
        goal_objects,
        room_size: int,
        seed: int,
        strict: bool,
        num_dists: int = 1,
    ):
        self.strict = strict
        self.goal_object, *_ = self.goal_objects = goal_objects
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
        self.instrs = GoToInstr(ObjDesc(*goal_object))


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


class GoToLocEnv(RenderEnv):
    def __init__(
        self,
        room_size: int,
        seed: int,
        num_dists: int = 0,
    ):
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
        self.check_objs_reachable()
        locs = itertools.product(
            range(1, self.grid.height - 1),
            range(1, self.grid.width - 1),
        )
        self.instrs = GoToLoc(LocDesc(self.grid, *self._rand_elem(locs)))


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


class PickupEnvRoomObjects(RenderEnv):
    def __init__(
        self,
        room_objects: typing.Iterable[typing.Tuple[str, str]],
        room_size: int,
        seed: int,
        strict: bool,
        num_dists: int = 1,
    ):
        self.strict = strict
        self.room_objects = list(room_objects)
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
        goal_object = self._rand_elem(self.room_objects)
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


class SequenceEnv(RenderEnv):
    def __init__(self, *args, strict: bool, **kwargs):
        self.strict = strict
        super().__init__(*args, **kwargs)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = {(ty, COLOR) for ty in TYPES}
        goal1 = self._rand_elem(objs)
        goal2 = self._rand_elem(objs - {goal1})
        for obj in [goal1, goal2]:
            self.add_object(0, 0, *obj)
        instr1 = GoToInstr(ObjDesc(*goal1))
        instr2 = GoToInstr(ObjDesc(*goal2))
        self.check_objs_reachable()
        self.instrs = BeforeInstr(instr1, instr2, strict=True)


class MissionWrapper(gym.Wrapper, abc.ABC):
    def __init__(self, env):
        self._mission = None
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._mission = self.change_mission(observation["mission"])
        observation["mission"] = self._mission
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation["mission"] = self._mission
        return observation, reward, done, info

    def render(self, mode="human", pause=True, **kwargs):
        self.env.render(pause=False)
        print(self._mission)
        if pause:
            input("Press enter to coninue.")

    def change_mission(self, mission):
        raise NotImplementedError


class PlantAnimalWrapper(MissionWrapper):
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

    def change_mission(self, mission):
        for k, v in self.replacements.items():
            if k in mission:
                replacement = self.np_random.choice(v)
                mission = mission.replace(k, replacement)

        return mission


class SynonymWrapper(MissionWrapper):
    synonyms = {
        "pick-up": ["take", "grab", "get"],
        "red": ["crimson"],
        "box": ["carton", "package"],
        "ball": ["sphere", "globe"],
        "phone": ["cell", "mobile"],
    }

    def __init__(self, env):
        super().__init__(env)
        self._mission

    def change_mission(self, mission):
        mission = mission.replace("key", "phone")
        mission = mission.replace("pick up", "pick-up")

        def new_mission():
            for word in mission.split():
                choices = [word, *self.synonyms.get(word, [])]
                yield self.np_random.choice(choices)

        mission = " ".join(new_mission())
        return mission


class InvalidInstructionError(RuntimeError):
    pass


class SequenceSynonymWrapper(MissionWrapper):
    def __init__(self, env, test: bool):
        super().__init__(env)
        self.test = test

    def change_mission(self, mission):
        def after_reverse(instr1: str, instr2: str):
            return f"After you {instr1}, {instr2}"

        def before(instr1: str, instr2: str):
            return f"{instr1} before you {instr2}"

        def then(instr1: str, instr2: str):
            return f"{instr1}, then {instr2}"

        def _next(instr1: str, instr2: str):
            return f"{instr1}. Next, {instr2}"

        def having(instr1: str, instr2: str):
            return f"{instr2}, having already {past(instr1)}"

        def after(instr1: str, instr2: str):
            return f"{instr2} after you {instr1}"

        def before_reverse(instr1: str, instr2: str):
            return f"Before you {instr2}, {instr1}"

        wordings = [after, after_reverse, before_reverse, then, _next, having]

        def past(instr: str):
            return instr.replace(" go ", " gone ")

        match = re.match(r"(.*), then (.*)", mission)
        if not match:
            match = re.match(r"(.*) after you (.*)", mission)
            if not match:
                breakpoint()

        wording: Callable[[str, str], str] = (
            before if self.test else self.np_random.choice(wordings)
        )
        mission = wording(*match.group(1, 2))
        return mission


def get_train_and_test_objects():
    all_objects = set(itertools.product(TYPES, COLORS))

    def pairs():

        remaining = set(all_objects)

        for _type, color in zip(TYPES, itertools.cycle(COLORS)):
            remaining.remove((_type, color))
            yield _type, color
        # yield from remaining

    train_objects = [*pairs()]
    return TrainTest(train=train_objects, test=list(all_objects))


class ActionInObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.spaces = dict(
            **self.observation_space.spaces, action=Discrete(self.action_space.n + 1)
        )
        self.observation_space = Dict(spaces=self.spaces)

    def reset(self, **kwargs):
        s = super().reset(**kwargs)
        s["action"] = self.spaces["action"].n - 1
        return s

    def step(self, action):
        s, r, t, i = super().step(action)
        s["action"] = action
        return s, r, t, i


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
        mission_space = spaces["mission"]
        self.observation_space = Box(
            shape=[np.prod(image_space.shape) + 2 + np.prod(mission_space.shape)],
            low=-np.inf,
            # direction_space = spaces["direction"]
            high=np.inf,
        )

    def observation(self, observation):
        return np.concatenate(
            astuple(
                Spaces(
                    image=observation["image"].flatten(),
                    direction=np.array([observation["direction"]]),
                    action=np.array([int(observation["action"])]),
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

    env = SequenceEnv(
        seed=args.seed,
        strict=args.strict,
        room_size=args.room_size,
        num_rows=1,
        num_cols=1,
    )
    env = SequenceEnv(
        seed=args.seed,
        strict=args.strict,
        room_size=args.room_size,
        num_rows=1,
        num_cols=1,
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

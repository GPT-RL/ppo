import abc
import itertools
import re
import typing
from abc import ABC
from dataclasses import astuple, dataclass
from itertools import chain, cycle, islice
from typing import Callable, Generator, Optional, TypeVar
from typing import Set, Union

import gym
import gym_minigrid
import numpy as np
from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import (
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    PickupInstr,
)
from colors import color as ansi_color
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import COLOR_NAMES, MiniGridEnv, OBJECT_TO_IDX, WorldObj
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from transformers import GPT2Tokenizer

from descs import (
    CardinalDirection,
    CornerDesc,
    FaceDesc,
    LocDesc,
    OrdinalDirection,
    RoomDesc,
    WallDesc,
)
from instrs import (
    FaceInstr,
    GoToCornerInstr,
    GoToLoc,
    GoToRoomInstr,
    GoToWallInstr,
    MultiAndInstr,
    ToggleInstr,
)

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


class ReproducibleEnv(RoomGridLevel, ABC):
    def _rand_elem(self, iterable):
        return super()._rand_elem(sorted(iterable))


class RenderEnv(RoomGridLevel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__reward = None
        self.__done = None
        self.__action = None

    def row_objs(self, y: int) -> Generator[Optional[WorldObj], None, None]:
        for x in range(self.width):
            if np.all(self.agent_pos == (x, y)):
                yield Agent(color="grey", type="agent")
            else:
                yield self.grid.get(x, y)

    def row_strings(self, i: int) -> Generator[str, None, None]:
        for obj in self.row_objs(i):
            if obj is None:
                string = ""
            elif isinstance(obj, Agent):
                if self.agent_dir == 0:
                    string = ">"
                elif self.agent_dir == 1:
                    string = "v"
                elif self.agent_dir == 2:
                    string = "<"
                elif self.agent_dir == 3:
                    string = "^"
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
            self.pause(pause)
        else:
            return super().render(mode=mode, **kwargs)

    @staticmethod
    def pause(pause):
        if pause:
            input("Press enter to continue.")

    def step(self, action):
        self.__action = action
        s, self.__reward, self.__done, i = super().step(action)
        return s, self.__reward, self.__done, i


class GoToObjEnv(RenderEnv, ReproducibleEnv):
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


class PickupEnv(RenderEnv, ReproducibleEnv):
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


class GoToLocEnv(RenderEnv, ReproducibleEnv):
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


class InvalidDirectionError(RuntimeError):
    pass


class DirectionsEnv(GoToLocEnv):
    def __init__(
        self,
        room_size: int,
        seed: int,
        strict: bool,
        directions: Set[Union[CardinalDirection, OrdinalDirection]],
    ):
        self.directions = list(directions)
        self.strict = strict
        super().__init__(room_size, seed)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        direction = self.np_random.choice(self.directions)
        if isinstance(direction, CardinalDirection):
            self.instrs = GoToWallInstr(
                desc=WallDesc(direction=direction, random=None), strict=self.strict
            )
        elif isinstance(direction, OrdinalDirection):
            self.instrs = GoToCornerInstr(
                desc=CornerDesc(direction=direction, random=None), strict=self.strict
            )
        else:
            raise InvalidDirectionError


class GoAndFaceDirections(typing.NamedTuple):
    room_direction: OrdinalDirection
    wall_direction: Union[CardinalDirection, OrdinalDirection]
    face_direction: CardinalDirection


def key(directions: GoAndFaceDirections):
    return (
        directions.room_direction.value,
        directions.wall_direction.value,
        directions.face_direction.value,
    )


class GoAndFaceEnv(RenderEnv, ReproducibleEnv):
    def __init__(
        self,
        room_size: int,
        seed: int,
        directions: Set[GoAndFaceDirections],
        synonyms: bool,
    ):
        self.synonyms = synonyms
        self.directions = sorted(directions, key=key)
        super().__init__(
            room_size=room_size,
            num_rows=2,
            num_cols=2,
            seed=seed,
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.check_objs_reachable()

        idx = self._rand_int(0, len(self.directions))
        d = self.directions[idx]
        random = self.np_random if self.synonyms else None
        if isinstance(d.wall_direction, CardinalDirection):
            wall_instr = GoToWallInstr(
                desc=WallDesc(
                    direction=d.wall_direction,
                    random=random,
                ),
                strict=False,
            )
        elif isinstance(d.wall_direction, OrdinalDirection):
            wall_instr = GoToCornerInstr(
                desc=CornerDesc(
                    direction=d.wall_direction,
                    random=random,
                ),
                strict=False,
            )
        else:
            raise InvalidDirectionError
        self.instrs = MultiAndInstr(
            GoToRoomInstr(
                RoomDesc(
                    direction=d.room_direction,
                    random=random,
                )
            ),
            wall_instr,
            FaceInstr(
                desc=FaceDesc(
                    direction=d.face_direction,
                    random=random,
                )
            ),
        )


class ToggleEnv(RenderEnv, ReproducibleEnv):
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


class PickupEnvRoomObjects(RenderEnv, ReproducibleEnv):
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


class SequenceEnv(RenderEnv, ReproducibleEnv):
    def __init__(self, *args, strict: bool, **kwargs):
        self.strict = strict
        super().__init__(*args, **kwargs)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        locs = list(
            itertools.product(
                range(1, self.grid.height - 1),
                range(1, self.grid.width - 1),
            )
        )

        instr1 = GoToLoc(LocDesc(self.grid, *self._rand_elem(locs)))
        instr2 = GoToLoc(LocDesc(self.grid, *self._rand_elem(locs)))
        self.check_objs_reachable()
        self.instrs = BeforeInstr(instr1, instr2, strict=True)

    def render(self, mode="human", pause=True, **kwargs):
        super().render(mode=mode, pause=False, **kwargs)
        print("Done:", self.instrs.a_done, self.instrs.b_done)
        self.pause(pause)


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
        self.env.pause(pause)

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


class PickupSynonymWrapper(MissionWrapper):
    synonyms = {
        "pick-up": ["take", "grab", "get"],
        "red": ["crimson"],
        "box": ["carton", "package"],
        "ball": ["sphere", "globe"],
        "phone": ["cell", "mobile"],
    }

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


class SequenceParaphrasesWrapper(MissionWrapper):
    def __init__(
        self,
        env,
        test: bool,
        train_wordings: typing.List[str],
        test_wordings: typing.List[str],
    ):
        super().__init__(env)

        def before(instr1: str, instr2: str):
            return f"{instr1} before you {instr2}"

        def before_reverse(instr1: str, instr2: str):
            return f"Before you {instr2}, {instr1}"

        def after(instr1: str, instr2: str):
            return f"{instr2} after you {instr1}"

        def after_reverse(instr1: str, instr2: str):
            return f"After you {instr1}, {instr2}"

        def once(instr1: str, instr2: str):
            return f"{instr2} once you {instr1}"

        def once_reverse(instr1: str, instr2: str):
            return f"Once you {instr1}, {instr2}"

        def having_reverse(instr1: str, instr2: str):
            return f"Having already {past(instr1)}, {instr2}"

        def having(instr1: str, instr2: str):
            return f"{instr2}, having already {past(instr1)}"

        def then(instr1: str, instr2: str):
            return f"{instr1}, then {instr2}"

        def _next(instr1: str, instr2: str):
            return f"{instr1}. Next, {instr2}"

        wordings = dict(
            before=before,
            before_reverse=before_reverse,
            after=after,
            after_reverse=after_reverse,
            once=once,
            once_reverse=once_reverse,
            having_reverse=having_reverse,
            having=having,
            then=then,
            _next=_next,
        )

        self.wordings = [wordings[w] for w in train_wordings]
        self.test_wordings = [wordings[w] for w in test_wordings]

        def past(instr: str):
            return instr.replace(" go ", " gone ")

        self.test = test

    def change_mission(self, mission):
        match = re.match(r"(.*), then (.*)", mission)
        if not match:
            match = re.match(r"(.*) after you (.*)", mission)
            if not match:
                breakpoint()

        wording: Callable[[str, str], str] = self.np_random.choice(
            self.test_wordings if self.test else self.wordings
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

    train_objects = sorted(pairs())
    return TrainTest(train=train_objects, test=sorted(all_objects))


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

    env = GoAndFaceEnv(
        seed=args.seed,
        room_size=args.room_size,
        directions={
            GoAndFaceDirections(*x)
            for x in itertools.product(
                OrdinalDirection,
                [*CardinalDirection, *OrdinalDirection],
                CardinalDirection,
            )
        },
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

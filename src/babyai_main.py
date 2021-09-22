import functools
from typing import Generator, Literal, Union, cast

from gym_minigrid.minigrid import COLOR_NAMES
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    DirectionWrapper,
    DirectionsEnv,
    FullyObsWrapper,
    GoAndFaceDirections,
    GoAndFaceEnv,
    GoToLocEnv,
    GoToObjEnv,
    NegationEnv,
    NegationObject,
    PickupEnv,
    PickupEnvRoomObjects,
    PickupRedEnv,
    PickupSynonymWrapper,
    PlantAnimalWrapper,
    RGBImgObsWithDirectionWrapper,
    RolloutsWrapper,
    SequenceEnv,
    SequenceParaphrasesWrapper,
    ToggleEnv,
    TokenizerWrapper,
    ZeroOneRewardWrapper,
)
from descs import CardinalDirection, OrdinalDirection, TYPES
from envs import RenderWrapper, VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "GoToLocal"  # env ID for gym
    go_and_face_synonyms: str = ""
    num_dists: int = 1
    room_size: int = 5
    second_layer: bool = False
    strict: bool = True
    test_colors: str = None
    test_descriptors: str = None
    test_wordings: str = None
    test_walls: str = "south,southeast"
    train_wordings: str = None

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


class Trainer(main.Trainer):
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=action_space,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
            second_layer=args.second_layer,
        )

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @staticmethod
    def make_env(env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            env_id: str,
            go_and_face_synonyms: str,
            num_dists: int,
            room_size: int,
            seed: int,
            strict: bool,
            test: bool,
            test_colors: str,
            test_descriptors: str,
            test_walls: str,
            test_wordings: str,
            tokenizer: GPT2Tokenizer,
            train_wordings: str,
            **_,
        ):
            goal_objects = (
                [("ball", "green")]
                if test
                else [
                    ("box", "green"),
                    ("box", "yellow"),
                    ("ball", "yellow"),
                ]
            )
            _kwargs = dict(
                room_size=room_size, strict=strict, num_dists=num_dists, seed=seed
            )

            if env_id == "go-to-obj":
                _env = GoToObjEnv(*args, goal_objects=goal_objects, **_kwargs)
                longest_mission = "go to the red ball"
            elif env_id == "go-to-loc":
                _env = GoToLocEnv(*args, **_kwargs)
                longest_mission = "go to (0, 0)"
            elif env_id == "toggle":
                _env = ToggleEnv(*args, **_kwargs)
                longest_mission = "toggle the red ball"
            elif env_id == "pickup":
                _env = PickupEnv(
                    *args,
                    num_dists=1,
                    goal_objects=goal_objects,
                    **_kwargs,
                )
                longest_mission = "pick up the red ball"
            elif env_id == "pickup-synonyms":
                _env = PickupRedEnv(*args, goal_objects=goal_objects, **_kwargs)
                if test:
                    _env = PickupSynonymWrapper(_env)
                longest_mission = "pick-up the crimson phone"
            elif env_id == "plant-animal":
                objects = {*PlantAnimalWrapper.replacements.keys()}
                test_objects = {
                    PlantAnimalWrapper.purple_animal,
                    PlantAnimalWrapper.black_plant,
                }
                train_objects = test_objects if test else objects - test_objects
                train_objects = [o.split() for o in train_objects]
                train_objects = [(t, c) for (c, t) in train_objects]
                _kwargs.update(room_objects=sorted(train_objects))
                _env = PickupEnvRoomObjects(*args, **_kwargs)
                _env = PlantAnimalWrapper(_env)
                longest_mission = "pick up the grasshopper"
            elif env_id == "directions":
                test_directions = {CardinalDirection.north}
                directions = (
                    test_directions
                    if test
                    else {*OrdinalDirection, *CardinalDirection} - test_directions
                )
                _env = DirectionsEnv(*args, directions=directions, **_kwargs)
                longest_mission = "go to northwest corner"
            elif env_id == "go-and-face":
                del _kwargs["strict"]

                def parse_test_walls() -> Generator[
                    Union[CardinalDirection, OrdinalDirection], None, None
                ]:
                    for wall in test_walls.split(","):
                        try:
                            yield CardinalDirection[wall]
                        except KeyError:
                            yield OrdinalDirection[wall]

                test_directions = {
                    GoAndFaceDirections(
                        room_direction=OrdinalDirection.southeast,
                        wall_direction=d1,
                        face_direction=d2,
                    )
                    for d1 in parse_test_walls()
                    for d2 in CardinalDirection
                }

                def get_directions():
                    for d1 in OrdinalDirection:
                        for d2 in [*OrdinalDirection, *CardinalDirection]:
                            for d3 in CardinalDirection:
                                yield GoAndFaceDirections(
                                    room_direction=d1,
                                    wall_direction=d2,
                                    face_direction=d3,
                                )

                directions = set(get_directions())
                directions = test_directions if test else directions - test_directions
                _kwargs.update({k: True for k in go_and_face_synonyms.split(",")})
                _env = GoAndFaceEnv(
                    *args,
                    directions=directions,
                    **_kwargs,
                )
                longest_mission = (
                    "go to the southeast room, go to the west wall, and face east"
                )
            elif env_id == "sequence-paraphrases":
                _env = SequenceEnv(*args, num_rows=1, num_cols=1, **_kwargs)
                _env = SequenceParaphrasesWrapper(
                    _env,
                    test=test,
                    train_wordings=train_wordings.split(","),
                    test_wordings=test_wordings.split(","),
                )
                longest_mission = "go to (0, 0), having already gone to (0, 0)"
            elif env_id == "sequence":
                _env = SequenceEnv(*args, num_rows=1, num_cols=1, **_kwargs)
                longest_mission = "go to (0, 0), then go to (0, 0)"
            elif env_id == "negation":

                COLORS = ["red", "green", "blue"]

                objects = {(ty, col) for ty in TYPES for col in COLORS}
                goal_objects = (
                    {
                        NegationObject(positive=True, type=ty, color=col)
                        for (ty, col) in objects
                    }
                    | {NegationObject(positive=False, type=ty) for ty in TYPES}
                    | {NegationObject(positive=False, color=col) for col in COLORS}
                )

                def get_test_goals():
                    assert isinstance(test_descriptors, str)
                    for s in test_descriptors.split(","):
                        if s in TYPES:
                            yield NegationObject(type=s, positive=False)
                        elif s in COLORS:
                            yield NegationObject(color=s, positive=False)
                        else:
                            raise RuntimeError(f"{s} is not a valid test_descriptor.")

                test_goals = set(get_test_goals())
                goal_objects = test_goals if test else goal_objects - test_goals
                _env = NegationEnv(
                    *args, goal_objects=goal_objects, room_objects=objects, **_kwargs
                )
                longest_mission = "pick up an object that is not a ball."
            elif env_id == "colors":
                test_colors = test_colors.split(",")
                train_colors = set(COLOR_NAMES) - set(test_colors)
                train_objects = sorted(
                    {(ty, col) for ty in TYPES for col in train_colors}
                )
                test_type = "ball"
                test_objects = sorted({(test_type, col) for col in test_colors})
                goal_objects = test_objects if test else train_objects
                room_objects = (
                    [(test_type, col) for col in train_colors]
                    if test
                    else train_objects
                )
                _env = PickupEnv(
                    *args,
                    room_objects=room_objects,
                    goal_objects=goal_objects,
                    **_kwargs,
                )
                longest_mission = "pick up the forest green ball."
            else:
                raise RuntimeError(f"{env_id} is not a valid env_id")

            _env = DirectionWrapper(_env)
            if env_id == "colors":
                _env = RGBImgObsWithDirectionWrapper(_env)
            else:
                _env = FullyObsWrapper(_env)

            _env = ActionInObsWrapper(_env)
            _env = ZeroOneRewardWrapper(_env)
            _env = TokenizerWrapper(
                _env,
                tokenizer=tokenizer,
                longest_mission=longest_mission,
            )
            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    @classmethod
    def make_vec_envs(cls, num_frame_stack=None, **kwargs):
        tokenizer = GPT2Tokenizer.from_pretrained(
            get_gpt_size(kwargs["embedding_size"])
        )
        return super().make_vec_envs(tokenizer=tokenizer, **kwargs)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))

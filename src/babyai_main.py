from typing import Generator, Literal, Union, cast

from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    DirectionsEnv,
    FullyObsWrapper,
    GoAndFaceDirections,
    GoAndFaceEnv,
    GoToLocEnv,
    GoToObjEnv,
    PickupEnv,
    PickupEnvRoomObjects,
    PickupRedEnv,
    PickupSynonymWrapper,
    PlantAnimalWrapper,
    RolloutsWrapper,
    SequenceEnv,
    SequenceParaphrasesWrapper,
    ToggleEnv,
    TokenizerWrapper,
    ZeroOneRewardWrapper,
)
from descs import CardinalDirection, OrdinalDirection
from envs import RenderWrapper, VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "GoToLocal"  # env ID for gym
    room_size: int = 5
    strict: bool = True
    train_wordings: str = None
    test_wordings: str = None
    test_walls: str = "south,southeast"
    go_and_face_synonyms: str = ""

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
        )

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @staticmethod
    def make_env(
        env_id, seed, rank, allow_early_resets, render: bool = False, *args, **kwargs
    ):
        def _thunk():
            tokenizer = kwargs.pop("tokenizer")
            test = kwargs.pop("test")
            train_wordings = kwargs.pop("train_wordings")
            test_wordings = kwargs.pop("test_wordings")
            test_walls = kwargs.pop("test_walls")
            go_and_face_synonyms = kwargs.pop("go_and_face_synonyms")
            goal_objects = (
                [("ball", "green")]
                if test
                else [
                    ("box", "green"),
                    ("box", "yellow"),
                    ("ball", "yellow"),
                ]
            )
            if env_id == "go-to-obj":
                env = GoToObjEnv(
                    *args, seed=seed + rank, goal_objects=goal_objects, **kwargs
                )
                longest_mission = "go to the red ball"
            elif env_id == "go-to-loc":
                env = GoToLocEnv(*args, seed=seed + rank, **kwargs)
                longest_mission = "go to (0, 0)"
            elif env_id == "toggle":
                env = ToggleEnv(*args, seed=seed + rank, **kwargs)
                longest_mission = "toggle the red ball"
            elif env_id == "pickup":
                env = PickupEnv(
                    *args,
                    seed=seed + rank,
                    num_dists=1,
                    goal_objects=goal_objects,
                    **kwargs,
                )
                longest_mission = "pick up the red ball"
            elif env_id == "pickup-synonyms":
                env = PickupRedEnv(
                    *args, seed=seed + rank, goal_objects=goal_objects, **kwargs
                )
                if test:
                    env = PickupSynonymWrapper(env)
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
                kwargs.update(room_objects=sorted(room_objects))
                env = PickupEnvRoomObjects(*args, seed=seed + rank, **kwargs)
                env = PlantAnimalWrapper(env)
                longest_mission = "pick up the grasshopper"
            elif env_id == "directions":
                test_directions = {CardinalDirection.north}
                directions = (
                    test_directions
                    if test
                    else {*OrdinalDirection, *CardinalDirection} - test_directions
                )
                env = DirectionsEnv(
                    *args, seed=seed + rank, directions=directions, **kwargs
                )
                longest_mission = "go to northwest corner"
            elif env_id == "go-and-face":
                del kwargs["strict"]

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
                kwargs.update({k: True for k in go_and_face_synonyms.split(",")})
                env = GoAndFaceEnv(
                    *args,
                    seed=seed + rank,
                    directions=directions,
                    **kwargs,
                )
                longest_mission = (
                    "go to the southeast room, go to the west wall, and face east"
                )
            else:
                if env_id == "sequence-paraphrases":
                    env = SequenceEnv(
                        *args, seed=seed + rank, num_rows=1, num_cols=1, **kwargs
                    )
                    env = SequenceParaphrasesWrapper(
                        env,
                        test=test,
                        train_wordings=train_wordings.split(","),
                        test_wordings=test_wordings.split(","),
                    )
                    longest_mission = "go to (0, 0), having already gone to (0, 0)"
                elif env_id == "sequence":
                    env = SequenceEnv(
                        *args, seed=seed + rank, num_rows=1, num_cols=1, **kwargs
                    )
                    longest_mission = "go to (0, 0), then go to (0, 0)"
                else:
                    raise RuntimeError(f"{env_id} is not a valid env_id")

            env = FullyObsWrapper(env)
            env = ActionInObsWrapper(env)
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
    def make_vec_envs(cls, args: ArgsType, device, **kwargs):
        # assert len(test_objects) >= 3
        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))
        return super().make_vec_envs(
            args,
            device,
            room_size=args.room_size,
            tokenizer=tokenizer,
            strict=args.strict,
            train_wordings=args.train_wordings,
            test_wordings=args.test_wordings,
            test_walls=args.test_walls,
            go_and_face_synonyms=args.go_and_face_synonyms,
            **kwargs,
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))

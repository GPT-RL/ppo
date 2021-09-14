import abc
import typing
from enum import Enum, auto
from typing import List

import numpy as np
from gym_minigrid.minigrid import WorldObj
from gym_minigrid.roomgrid import RoomGrid


class CardinalDirection(Enum):
    east = auto()
    south = auto()
    west = auto()
    north = auto()


class OrdinalDirection(Enum):
    northeast = auto()
    northwest = auto()
    southeast = auto()
    southwest = auto()


direction_synonym = {
    OrdinalDirection.northeast: ["upper right", "top right"],
    OrdinalDirection.northwest: ["upper left", "top left"],
    OrdinalDirection.southeast: ["lower right", "bottom right"],
    OrdinalDirection.southwest: ["lower left", "bottom left"],
    CardinalDirection.east: ["right"],
    CardinalDirection.south: ["lower", "bottom"],
    CardinalDirection.west: ["left"],
    CardinalDirection.north: ["upper", "top"],
}

opposites = {
    OrdinalDirection.northeast: OrdinalDirection.southwest,
    OrdinalDirection.northwest: OrdinalDirection.southeast,
    CardinalDirection.east: CardinalDirection.west,
    CardinalDirection.south: CardinalDirection.north,
}


opposites.update({v: k for k, v in opposites.items()})


class Desc:
    def surface(self):
        """
        Generate a natural language representation of the object description
        """
        return repr(self)

    @staticmethod
    def find_matching_objs(
        *args, **kwargs
    ) -> typing.Tuple[List[WorldObj], List[WorldObj]]:
        return [], []


class RandomDesc(Desc):
    def __init__(self, random: np.random.RandomState):
        self.choices_list = list(self.choices())
        self.random = random
        self.repr = self.sample_repr()

    @abc.abstractmethod
    def choices(self) -> typing.Generator[str, None, None]:
        raise NotImplementedError

    def __repr__(self):
        return self.repr

    def sample_repr(self):
        return self.choices_list[self.random.choice(len(self.choices_list))]


adjacencies = {
    OrdinalDirection.northeast: [
        "to the right of the northwest",
        "above the southeast",
    ],
    OrdinalDirection.northwest: ["to the left of the northeast", "above the southwest"],
    OrdinalDirection.southeast: [
        "to the right of the southwest",
        "below the northeast",
    ],
    OrdinalDirection.southwest: [
        "to the left of the southeast",
        "below the northwest",
    ],
}


class RoomDesc(RandomDesc):
    def __init__(self, *args, direction: OrdinalDirection, synonyms: bool, **kwargs):
        self.synonyms = synonyms
        self.direction = direction
        super().__init__(*args, **kwargs)

    def choices(self):
        yield f"the {self.direction.name} room"
        if self.synonyms:
            for adjacent_to in adjacencies[self.direction]:
                yield f"the room {adjacent_to} room"
            for synonym in direction_synonym[self.direction]:
                yield f"the room in the {synonym}"
            opposite = opposites[self.direction]
            yield f"the room opposite from the {opposite.name}"
            for synonym in direction_synonym[opposite]:
                yield f"the room opposite from the {synonym}"
                yield f"the room furthest from the {synonym}"


class WallDesc(RandomDesc):
    def __init__(self, *args, direction: CardinalDirection, synonyms: bool, **kwargs):
        self.synonyms = synonyms
        self.direction = direction
        super().__init__(*args, **kwargs)

    def choices(self):
        yield f"the {self.direction.name} wall"
        if self.synonyms:
            for synonym in direction_synonym[self.direction]:
                yield f"the {synonym} wall"
            opposite = opposites[self.direction]
            yield f"the wall opposite from the {opposite.name}"
            for synonym in direction_synonym[opposite]:
                yield f"the wall opposite from the {synonym} wall"
                yield f"the wall furthest from the {synonym} wall"


class CornerDesc(RandomDesc):
    def __init__(self, *args, direction: OrdinalDirection, synonyms: bool, **kwargs):
        self.synonyms = synonyms
        self.direction = direction
        super().__init__(*args, **kwargs)

    def choices(self):
        yield f"the {self.direction.name} corner"
        if self.synonyms:
            for synonym in direction_synonym[self.direction]:
                yield f"the {synonym} corner"
            for adjacent_to in adjacencies[self.direction]:
                yield f"the corner {adjacent_to} corner"
            opposite = opposites[self.direction]
            yield f"the corner opposite from the {opposite.name}"
            for synonym in direction_synonym[opposite]:
                yield f"the corner opposite from the {synonym}"
                yield f"the corner furthest from the {synonym}"


class FaceDesc(RandomDesc):
    direction_synonym = {
        CardinalDirection.east: "right",
        CardinalDirection.south: "down",
        CardinalDirection.west: "left",
        CardinalDirection.north: "up",
    }

    def __init__(self, *args, direction: CardinalDirection, synonyms: bool, **kwargs):
        self.synonyms = synonyms
        self.direction = direction
        super().__init__(*args, **kwargs)

    def choices(self):
        yield f"face {self.direction.name}"
        if self.synonyms:
            opposite = opposites[self.direction]
            yield f"face away from the {opposite.name}"
            yield f"look {self.direction_synonym[self.direction]}"


class LocDesc(Desc):
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

    @property
    def array(self):
        return np.array([self.i, self.j])

    @staticmethod
    def find_matching_objs(*args, **kwargs):
        return [], []

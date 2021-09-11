import typing
from enum import Enum, auto
from typing import List

import numpy as np
from gym_minigrid.minigrid import WorldObj
from gym_minigrid.roomgrid import RoomGrid


class CardinalDirection(Enum):
    north = auto()
    south = auto()
    east = auto()
    west = auto()


class OrdinalDirection(Enum):
    northeast = auto()
    northwest = auto()
    southeast = auto()
    southwest = auto()


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


class WallDesc(Desc):
    """
    Description of a set of objects in an environment
    """

    def __init__(self, direction: CardinalDirection):
        self.direction = direction

    def __repr__(self):
        return f"the {self.direction.name} wall"


class CornerDesc(Desc):
    """
    Description of a set of objects in an environment
    """

    def __init__(self, direction: OrdinalDirection):
        self.direction = direction

    def __repr__(self):
        return f"the {self.direction.name} corner"


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

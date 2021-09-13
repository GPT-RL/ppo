import numpy as np
from babyai.levels.verifier import ActionInstr, AndInstr
from gym_minigrid.minigrid import MiniGridEnv

from descs import CardinalDirection, CornerDesc, LocDesc, OrdinalDirection, WallDesc


class GoToCornerInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, desc: CornerDesc, strict=False):
        super().__init__()
        self.desc = desc
        self.strict = strict

    def surface(self, env):
        return "Go to " + self.desc.surface()

    def verify_action(self, action):
        x, y = self.env.agent_pos
        direction = self.desc.direction

        def validate_direction(positional_direction: OrdinalDirection):
            if direction == positional_direction:
                return "success"
            elif self.strict:
                return "failure"
            else:
                return "continue"

        self.env: MiniGridEnv
        height = self.env.height
        width = self.env.width

        if (x, y) == (1, 1):
            return validate_direction(OrdinalDirection.northwest)
        if (x, y) == (width - 2, 1):
            return validate_direction(OrdinalDirection.northeast)
        if (x, y) == (1, height - 2):
            return validate_direction(OrdinalDirection.southwest)
        if (x, y) == (width - 2, height - 2):
            return validate_direction(OrdinalDirection.southeast)

        return "continue"


class GoToWallInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, desc: WallDesc, strict=False):
        super().__init__()
        self.desc = desc
        self.strict = strict

    def surface(self, env):
        return "Go to " + self.desc.surface()

    def verify_action(self, action):
        x, y = self.env.agent_pos
        direction = self.desc.direction

        def validate_direction(positional_direction: CardinalDirection):
            if direction == positional_direction:
                return "success"
            elif self.strict:
                return "failure"
            else:
                return "continue"

        self.env: MiniGridEnv

        if y == 1:
            return validate_direction(CardinalDirection.north)
        if y == self.env.height - 2:
            return validate_direction(CardinalDirection.south)
        if x == 1:
            return validate_direction(CardinalDirection.west)
        if x == self.env.width - 2:
            return validate_direction(CardinalDirection.east)

        return "continue"


class FaceInstr(ActionInstr):
    """
    Pick up an object matching a given description
    eg: pick up the grey ball
    """

    def __init__(self, direction: CardinalDirection):
        super().__init__()
        self.direction = direction

    def surface(self, env):
        return "face " + self.direction.name

    def verify_action(self, action):
        self.env: MiniGridEnv
        return (
            "success"
            if self.env.agent_dir == [*CardinalDirection].index(self.direction)
            else "continue"
        )


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


class AndDoneInstr(AndInstr):
    def verify(self, action):
        if action is self.env.actions.done:
            a_done = self.instr_a.verify(action)
            b_done = self.instr_b.verify(action)
            return (
                "success"
                if (a_done == "success" and b_done == "success")
                else "failure"
            )
        return "continue"

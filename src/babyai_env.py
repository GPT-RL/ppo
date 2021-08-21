import babyai
import gym
from babyai.levels.verifier import GoToInstr, ObjDesc
from gym.spaces import Dict, Discrete
from gym_minigrid.minigrid import COLOR_NAMES, WorldObj


def all_object_types():
    for color in COLOR_NAMES:
        for object_type in ["key", "ball", "box"]:
            yield object_type, color


class ObsSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.observation_space = Dict(
            spaces=dict(
                **spaces,
                direction=Discrete(4),
            )
        )

    def observation(self, observation):
        return observation


class GoToLocalEnv(babyai.levels.iclr19_levels.Level_GoToLocal):
    def __init__(self, goal_objects, room_size, num_dists, seed):
        self.goal_objects = goal_objects
        super().__init__(
            seed=seed,
            room_size=room_size,
            num_dists=num_dists,
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(filter(self.can_be_goal, objs))
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

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

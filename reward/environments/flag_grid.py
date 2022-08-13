from collections import defaultdict

import numpy as np
from frozendict import frozendict
from msdm.domains import GridWorld
from msdm.core.utils.gridstringutils import string_to_element_array
from msdm.core.distributions.dictdistribution import \
    FiniteDistribution, DeterministicDistribution, \
    DictDistribution, UniformDistribution


class FlagGrid(GridWorld):
    """
    The Gridworld with 5 flags from the Potential Based Reward Shaping paper by Ng
    http://luthuli.cs.uiuc.edu/~daf/courses/games/AIpapers/ml99-shaping.pdf

    NOTE: because of how we update self.current_flag in self.reward(),
    this class is currently only compatible with our QLearning and RMax. 
    ValueIteratio should not be used on this class. 
    """
    def __init__(self,
                tile_array=None,
                flag_array=None,
                flag_rewards=None,
                # absorbing_features=("g", "x"),
                wall_features=("#",),
                # default_features=(".",),
                initial_features=("s",),
                step_cost=0,
                success_prob=1,
                discount_rate=1.0,
                custom_rewards=None,
                ):
        # set default grid config
        assert (tile_array is None) == (flag_array is None)
        if tile_array is None:
            tile_array = [
                '....g',
                '.....',
                '.....',
                '.....',
                's....',
            ]
        if flag_array is None:
            flag_array = [
                [None, None, None, None, 'g'],
                [None, 2, None, None, None],
                [None, None, None, None, None],
                [3, None, None, None, 1],
                [None, None, None, None, 4],
            ]

        self.tile_array = tile_array
        parseParams = {"colsep": "", "rowsep": "\n", "elementsep": "."}
        if not isinstance(tile_array, str):
            tile_array = "\n".join(tile_array)
        elementArray = string_to_element_array(tile_array, **parseParams)

        # figure out elements
        states = []
        walls = set()
        # absorbingStates = set()
        initStates = set()
        locFeatures = {}
        locFlags = {}
        for y_, row in enumerate(elementArray):
            y = len(elementArray) - y_ - 1
            for x, elements in enumerate(row):
                s = frozendict({'x': x, 'y': y})
                states.append(s)
                locFlags[s] = flag_array[y_][x]
                if len(elements) > 0:
                    f = elements[0]
                    locFeatures[s] = f
                    if f in initial_features:
                        initStates.add(s)
                    # if f in absorbing_features:
                    #     absorbingStates.add(s)
                    if f in wall_features:
                        walls.add(s)

        # the actions
        actions = [
            frozendict({'dx': 1, 'dy': 0}),
            frozendict({'dx': -1, 'dy': 0}),
            frozendict({'dy': 1, 'dx': 0}),
            frozendict({'dy': -1, 'dx': 0})
        ]

        # house keeping
        hash_state = lambda s: (s['x'], s['y'])
        hash_action = lambda a: (a['dx'], a['dy'])
        self._actions = sorted(actions, key=hash_action)
        self._states = sorted(states, key=hash_state)
        self._initStates = sorted(initStates, key=hash_state)
        # self._absorbingStates = sorted(absorbingStates, key=hash_state)
        self._walls = sorted(walls, key=hash_state)
        self._locFeatures = locFeatures
        self.success_prob = success_prob
        # if feature_rewards is None:
        #     feature_rewards = {'g': 0}
        # self._featureRewards = feature_rewards
        if flag_rewards is None:
            flag_rewards = {
                'g': +1,
            }
        self.flag_rewards = flag_rewards
        self.step_cost = step_cost
        self.discount_rate = discount_rate
        self._height = len(elementArray)
        self._width = len(elementArray[0])

        # custom reward
        if custom_rewards:
            assert type(custom_rewards) == dict
        self.custom_rewards = custom_rewards

        # flags
        self.location_flags = locFlags
        self.current_flag = None

    def is_terminal(self, s):
        return self.current_flag == 'g'

    def initial_state_dist(self) -> FiniteDistribution:
        self.current_flag = None
        return UniformDistribution(self.initial_states)

    def next_state_dist(self, s, a) -> FiniteDistribution:
        if self.is_terminal(s):
            return DeterministicDistribution(s)
        assert isinstance(s, frozendict)

        x, y = s['x'], s['y']
        ax, ay = a.get('dx', 0), a.get('dy', 0)
        nx, ny = x + ax, y + ay
        ns = frozendict({'x': nx, 'y': ny})

        # adjust next state
        if ns not in self._states:
            ns = s
        elif ns in self.walls:
            ns = s

        if self.success_prob != 1:
            # slip to the two sides
            if ax == 0:
                # agent moved in y direction
                slip_s1 = frozendict({'x': x-1, 'y': y})
                slip_s2 = frozendict({'x': x+1, 'y': y})
            else:
                # agent moved in x direction
                slip_s1 = frozendict({'x': x, 'y': y-1})
                slip_s2 = frozendict({'x': x, 'y': y+1})
            
            if slip_s1 not in self._states or slip_s1 in self.walls:
                slip_s1 = s
            if slip_s2 not in self._states or slip_s2 in self.walls:
                slip_s2 = s

            bdist_dict = defaultdict(float)
            bdist_dict[ns] += self.success_prob
            bdist_dict[slip_s1] += (1-self.success_prob)/2
            bdist_dict[slip_s2] += (1-self.success_prob)/2
            bdist = DictDistribution(bdist_dict)
        else:
            bdist = DeterministicDistribution(ns)

        return bdist

    def reward(self, s, a, ns) -> float:
        """
        NOTE: we update the current flag here
        NOTE: assume that mdp.reward() is always called after mdp.next_state_dist()
        """
        # update the flag
        self.current_flag = self.location_flags[ns]

        # rewards
        if self.custom_rewards is None:
            # use default rewards
            return self.flag_rewards.get(self.current_flag, self.step_cost)
        else:
            # use custom rewards
            return self.custom_rewards[self.current_flag]


def make_flag_grid(discount_rate, success_prob, step_cost=-1, flag_rewards=None):
    if flag_rewards is None:
        flag_rewards = {
            'g': +1,
            1: step_cost,
            2: step_cost,
            3: step_cost,
            4: step_cost,
        }
    gw = FlagGrid(
        flag_rewards=flag_rewards,
        step_cost=step_cost,
        success_prob=success_prob,
        discount_rate=discount_rate,
    )
    return gw


if __name__ == "__main__":
    # for debugging
    gw = make_flag_grid(0.9, 0.8)

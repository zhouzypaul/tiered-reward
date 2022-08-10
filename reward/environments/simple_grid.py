import math

import numpy as np

from frozendict import frozendict
from msdm.domains import GridWorld
from msdm.core.utils.gridstringutils import string_to_element_array
from msdm.core.distributions.dictdistribution import \
    FiniteDistribution, DeterministicDistribution, \
    DictDistribution


class SimpleGrid(GridWorld):
    """
    create a 2-D square gridworld, where the agent starts at the bottom left state,
    and the goal is at the top right state. 

    This environment seeks to extend the OneDimChain into 2D.

    the start and end state are both (r, c) coordinate tuples. r is the row number,
    indexed from the top from 0, and c is the column number, indexed from the left
    from 0. E.g:
    (0, 0), (0, 1), (0, 2)
    (1, 0), (1, 1), (1, 2)
    """
    def __init__(self,
                num_states=25,  # number of states, must be a square number
                start_state=None,
                goal_state=None,
                tile_array=None,
                feature_rewards=None,
                absorbing_features=("g",),
                wall_features=("#",),
                default_features=(".",),
                initial_features=("s",),
                step_cost=0,
                success_prob=1,
                discount_rate=1.0,
                custom_rewards=None,
                ):
        
        # info
        self.num_states = num_states
        assert num_states == int(math.sqrt(num_states))**2
        num_side_states = math.isqrt(num_states)
        self.num_side_states = num_side_states
    
        # create the tile 
        if tile_array is None:
            tile_array = np.array([['.'] * num_side_states] * num_side_states)
        if start_state is None:  # start state
            tile_array[-1][0] = 's'
        else:
            tile_array[start_state[0]][start_state[1]] = 's'
        if goal_state is None:  # goal state
            tile_array[0][-1] = 'g'
        else:
            tile_array[goal_state[0]][goal_state[1]] = 'g'
        # convert of list of list of strings
        tile_array = list(map(lambda x: ''.join(x), tile_array.tolist()))

        self.tile_array = tile_array
        parseParams = {"colsep": "", "rowsep": "\n", "elementsep": "."}
        if not isinstance(tile_array, str):
            tile_array = "\n".join(tile_array)
        elementArray = string_to_element_array(tile_array, **parseParams)

        # figure out elements
        states = []
        walls = set()
        absorbingStates = set()
        initStates = set()
        locFeatures = {}
        for y_, row in enumerate(elementArray):
            y = len(elementArray) - y_ - 1
            for x, elements in enumerate(row):
                s = frozendict({'x': x, 'y': y})
                states.append(s)
                if len(elements) > 0:
                    f = elements[0]
                    locFeatures[s] = f
                    if f in initial_features:
                        initStates.add(s)
                    if f in absorbing_features:
                        absorbingStates.add(s)
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
        self._absorbingStates = sorted(absorbingStates, key=hash_state)
        self._walls = sorted(walls, key=hash_state)
        self._locFeatures = locFeatures
        self.success_prob = success_prob
        if feature_rewards is None:
            feature_rewards = {'g': 0}
        self._featureRewards = feature_rewards
        self.step_cost = step_cost
        self.discount_rate = discount_rate
        self._height = len(elementArray)
        self._width = len(elementArray[0])

        # custom reward
        self.custom_rewards = custom_rewards

    def is_terminal(self, s):
        return self.location_features.get(s, '') == 'g'

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
            
            if slip_s1 not in self._states:
                slip_s1 = s
            if slip_s2 not in self._states:
                slip_s2 = s

            bdist = DictDistribution({
                ns: self.success_prob,
                slip_s1: (1 - self.success_prob) / 2,
                slip_s2: (1 - self.success_prob) / 2,
            })
        else:
            bdist = DeterministicDistribution(ns)

        return bdist

    def reward(self, s, a, ns) -> float:
        if self.custom_rewards is None:
            # use default rewards
            if self.is_terminal(s):
                return self._featureRewards['g']
            f = self._locFeatures.get(s, "")
            return self._featureRewards.get(f, self.step_cost)
        else:
            # use custom rewards
            idx_s = self.state_index[s]
            return self.custom_rewards[idx_s]
    
    @property
    def reward_vector(self):
        """
        a vector a length n_states
        """
        if self.custom_rewards is None:
            # use default rewards
            reward_vec = np.zeros((len(self.state_list), ))
            for s in self.state_list:
                # reward only depends on leaving at s
                reward_vec[self.state_index[s]] = self.reward(s, None, None)
            return reward_vec
        else:
            # use custom rewards
            return self.custom_rewards


def make_simple_grid(num_states, goal_reward, step_reward, discount_rate, start_state=None, goal_state=None, success_rate=1, custom_rewards=None):
    """
    create a simple grid environment
    """
    grid_params = dict(
        initial_features=('s',),  # start state
        absorbing_features=('g',),  # goals
        wall_features=('#',),  # walls
        default_features=('.',),  # hallway
        feature_rewards={
            'g': goal_reward,
        },
        step_cost=step_reward,
    )

    chain = SimpleGrid(
        **grid_params,
        num_states=num_states,
        discount_rate=discount_rate,
        start_state=start_state,
        goal_state=goal_state,
        success_prob=success_rate,
        custom_rewards=custom_rewards,
    )

    return chain


if __name__ == "__main__":
    # for testing this script
    from pathlib import Path
    from reward.environments.plot import visualize_grid_world_and_policy, plot_grid_reward

    results_dir = Path('results').joinpath('simplegrid_plot')
    results_dir.mkdir(exist_ok=True)
    
    chain = make_simple_grid(num_states=5, goal_reward=0, step_reward=-1, discount_rate=0.95, custom_rewards=None)
    visualize_grid_world_and_policy(gw=chain, plot_name_prefix='simple_grid', results_dir=results_dir)
    plot_grid_reward(gw=chain, plot_name_prefix='grid', results_dir=results_dir)

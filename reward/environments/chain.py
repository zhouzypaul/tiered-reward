import numpy as np

from frozendict import frozendict
from msdm.domains import GridWorld
from msdm.core.utils.gridstringutils import string_to_element_array
from msdm.core.distributions.dictdistribution import \
    FiniteDistribution, DeterministicDistribution, \
    DictDistribution


class OneDimChain(GridWorld):
    """
    create a 1-D chain grid world, where the agent starts in the leftmost state,
    and the goal state is in the rightmost state. By default, the chain is 
    deterministic and non-slipery
    The agent only has 2 actions, going left or going right.
    The agent starts at a specified start_state, and the goal state is always the right end.
    """
    def __init__(self,
                num_states=60,
                start_state=0,
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
        assert 0 <= start_state < num_states
        self.num_states = num_states
    
        # create the tile 
        if tile_array is None:
            tile_array = [
            '.' * start_state + 's' + '.' * (num_states - 2 - start_state) + 'g'
        ]
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
        assert ay == 0
        nx, ny = x + ax, y + ay
        ns = frozendict({'x': nx, 'y': ny})
    
        # adjust next state
        if ns not in self._states:
            ns = s
        elif ns in self.walls:
            ns = s
        
        if self.success_prob != 1:
            # slip in the reverse direction
            reverse_nx = x - ax
            prev_s = frozendict({'x': reverse_nx, 'y': ny})
            if prev_s not in self._states:
                prev_s = s
            bdist = DictDistribution({
                prev_s: 1 - self.success_prob,
                ns: self.success_prob
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


def make_one_dim_chain(num_states, goal_reward, step_reward, discount_rate, 
                        start_state=0, success_rate=1, custom_rewards=None):
    """
    create a one dimensional chain mdp
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

    chain = OneDimChain(
        **grid_params,
        num_states=num_states,
        start_state=start_state,
        discount_rate=discount_rate,
        success_prob=success_rate,
        custom_rewards=custom_rewards,
    )

    return chain


if __name__ == "__main__":
    # for testing this script
    from pathlib import Path
    from reward.environments.plot import visualize_grid_world_and_policy, plot_grid_reward

    results_dir = Path('results').joinpath('chain_plot')
    results_dir.mkdir(exist_ok=True)
    
    chain = make_one_dim_chain(num_states=60, goal_reward=0, step_reward=-1, discount_rate=0.95, custom_rewards=None)
    visualize_grid_world_and_policy(gw=chain, plot_name_prefix='sutton_grid', results_dir=results_dir)
    plot_grid_reward(gw=chain, plot_name_prefix='chain', results_dir=results_dir)

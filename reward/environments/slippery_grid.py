from collections import defaultdict

import numpy as np
from frozendict import frozendict
from msdm.domains import GridWorld
from msdm.core.utils.gridstringutils import string_to_element_array
from msdm.core.distributions.dictdistribution import \
    FiniteDistribution, DeterministicDistribution, \
    DictDistribution


class SlipperyGrid(GridWorld):
    """
    2-D grid world where the agent can move in all 4 directions. With each move, the agent
    has some probability of moving to that direction, while slipping to either side. 

    There is a goal state g, and may be a lava state x.
    """
    def __init__(self,
                tile_array=None,
                feature_rewards=None,
                absorbing_features=("g", "x"),
                wall_features=("#",),
                default_features=(".",),
                initial_features=("s",),
                step_cost=0,
                success_prob=1,
                discount_rate=1.0,
                custom_rewards=None,
                ):
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
        self.absorbing_features = absorbing_features

        # custom reward
        self.custom_rewards = custom_rewards

    def is_terminal(self, s):
        return self.location_features.get(s, '') in self.absorbing_features

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

            bdist_dict = defaultdict(float)
            bdist_dict[ns] += self.success_prob
            bdist_dict[slip_s1] += (1-self.success_prob)/2
            bdist_dict[slip_s2] += (1-self.success_prob)/2
            bdist = DictDistribution(bdist_dict)
        else:
            bdist = DeterministicDistribution(ns)

        return bdist

    def reward(self, s, a, ns) -> float:
        if self.custom_rewards is None:
            # use default rewards
            if self.is_terminal(ns):
                return self._featureRewards['g']
            f = self._locFeatures.get(ns, "")
            return self._featureRewards.get(f, self.step_cost)
        else:
            # use custom rewards
            idx_ns = self.state_index[ns]
            return self.custom_rewards[idx_ns]
    
    @property
    def reward_vector(self):
        """
        a vector a length n_states
        """
        if self.custom_rewards is None:
            # use default rewards
            reward_vec = np.zeros((len(self.state_list), ))
            for s in self.state_list:
                reward_vec[self.state_index[s]] = self.reward(None, None, s)
            return reward_vec
        else:
            # use custom rewards
            return self.custom_rewards


def make_slippery_grid(discount_rate, success_prob, step_cost, goal_reward, lava_penalty):
    tile_array = [
        's...',
        '.x.x',
        '...x',
        'x..g',
    ]
    gw = SlipperyGrid(
        tile_array=tile_array,
        feature_rewards={
            'g': goal_reward,
            'x': lava_penalty,
        },
        absorbing_features=('g', 'x'),
        step_cost=step_cost,
        success_prob=success_prob,
        discount_rate=discount_rate,
    )
    return gw


def make_single_goal_square_grid(num_side_states, discount_rate, success_prob, step_cost, goal_reward, custom_reward=None):
    """
    make a square grid that's num_side_states * num_side_states
    , where the agent starts at the bottom left corner, and the goa is top right corner
    """
    tile_array = [
        '.' * (num_side_states-1) + 'g',
    ] + [
        '.' * num_side_states,
    ] * (num_side_states-2) + [
        's' + '.' * (num_side_states-1),
    ]
    gw = SlipperyGrid(
        tile_array=tile_array,
        feature_rewards={
            'g': goal_reward,
        },
        absorbing_features=('g',),
        step_cost=step_cost,
        success_prob=success_prob,
        discount_rate=discount_rate,
        custom_rewards=custom_reward,
    )
    return gw


if __name__ == "__main__":
    # for debugging
    import matplotlib.pyplot as plt
    # gw = make_slippery_grid(discount_rate=0.9, success_prob=0.5, step_cost=-0.1, goal_reward=1, lava_penalty=-1)
    gw = make_single_goal_square_grid(num_side_states=5, discount_rate=0.9, success_prob=0.5, step_cost=-0.1, goal_reward=1)
    gw.plot()

    from msdm.algorithms import ValueIteration
    vi = ValueIteration()
    result = vi.plan_on(gw)
    policy = result.policy

    fig, axes = plt.subplots(2, 1)
    gw.plot(ax=axes[0]).plot_state_map(result.valuefunc).title("Value Function")
    gw.plot(ax=axes[1]).plot_policy(policy).title("Policy")

    save_path = 'results/debug.png'
    print('Saving to {}'.format(save_path))
    plt.savefig(fname=save_path)

    print(gw.state_list)

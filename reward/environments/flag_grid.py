from collections import defaultdict

import numpy as np
from frozendict import frozendict
from msdm.core.utils.gridstringutils import string_to_element_array
from msdm.core.problemclasses.mdp.quicktabularmdp import QuickTabularMDP
from msdm.core.distributions.dictdistribution import \
    FiniteDistribution, DeterministicDistribution, \
    DictDistribution, UniformDistribution


def make_flag_grid(discount_rate, step_cost=-1, flag_rewards=None):
    """
    The Gridworld with 5 flags from the Potential Based Reward Shaping paper by Ng
    http://luthuli.cs.uiuc.edu/~daf/courses/games/AIpapers/ml99-shaping.pdf
    """
    # default grid config
    tile_array = [
        '....g',
        '.....',
        '.....',
        '.....',
        's....',
    ]
    height = len(tile_array)
    width = len(tile_array[0])

    default_flag = 0
    goal_flag = 5
    flag_array = [
        [default_flag, default_flag, default_flag, default_flag, goal_flag],
        [default_flag, 2, default_flag, default_flag, default_flag],
        [default_flag, default_flag, default_flag, default_flag, default_flag],
        [3, default_flag, default_flag, default_flag, 1],
        [default_flag, default_flag, default_flag, default_flag, 4],
    ]

    if flag_rewards is None:
        flag_rewards = {
            goal_flag: +1,
            1: step_cost,
            2: step_cost,
            3: step_cost,
            4: step_cost,
        }

    # parse tile array
    parseParams = {"colsep": "", "rowsep": "\n", "elementsep": "."}
    if not isinstance(tile_array, str):
        tile_array = "\n".join(tile_array)
    elementArray = string_to_element_array(tile_array, **parseParams)

    # find elements
    initStates = []
    locFlags = {}
    for y_, row in enumerate(elementArray):
        y = len(elementArray) - y_ - 1
        for x, elements in enumerate(row):
            s = frozendict({'x': x, 'y': y, 'flag': flag_array[y_][x]})
            locFlags[(x, y)] = flag_array[y_][x]
            if len(elements) > 0:
                f = elements[0]
                if f == 's':
                    initStates.append(s)
    
    # the actions
    actions = [
        frozendict({'dx': 1, 'dy': 0}),
        frozendict({'dx': -1, 'dy': 0}),
        frozendict({'dy': 1, 'dx': 0}),
        frozendict({'dy': -1, 'dx': 0}),
    ]

    def sample_random_action():
        return np.random.choice(actions)

    def initial_state_dist() -> FiniteDistribution:
        return UniformDistribution(initStates)
    
    def is_terminal(s):
        return s['flag'] == goal_flag
    
    def reward(s, a, ns):
        f = ns['flag']
        return flag_rewards.get(f, step_cost)
    
    def is_valid_loc(x, y):
        return 0 <= x <= width-1 and 0 <= y <= height-1
    
    def apply_action(s, a):
        # location change
        x, y, flag = s['x'], s['y'], s['flag']
        ax, ay = a.get('dx', 0), a.get('dy', 0)
        nx, ny = x + ax, y + ay
        
        # check if hit a perimiter
        if not is_valid_loc(nx, ny):
            nx, ny = x, y
        
        # flag change
        n_location_flag = locFlags[(nx, ny)]
        if n_location_flag == flag + 1:
            nflag = n_location_flag
        else:
            nflag = flag

        ns = frozendict({'x': nx, 'y': ny, 'flag': nflag})
        return ns

    def next_state_dist(s, a):
        """
        move 1 step in the intended direction 0.8 of the time and a random direction 0.2 of the time
        """
        if is_terminal(s):
            return DeterministicDistribution(s)

        assert isinstance(s, frozendict)
        bdist_dict = defaultdict(float)

        ns = apply_action(s, a)
        bdist_dict[ns] = 0.8

        # random action with 0.2 probability
        rand_a = sample_random_action()
        rand_ns = apply_action(s, rand_a)
        bdist_dict[rand_ns] = 0.2

        bdist = DictDistribution(bdist_dict)
        return bdist

    gw = QuickTabularMDP(
        next_state_dist=next_state_dist,
        reward=reward,
        actions=actions,
        initial_state_dist=initial_state_dist,
        is_terminal=is_terminal,
        discount_rate=discount_rate
    )

    gw.location_features = {
        frozendict({'x': 4, 'y': 4, 'flag': 5}): 'g',
    }

    return gw


if __name__ == "__main__":
    # for debugging
    gw = make_flag_grid(0.9, 0.8)
    print(len(gw.state_list))

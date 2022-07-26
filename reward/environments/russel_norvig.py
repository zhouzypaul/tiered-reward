import copy
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Circle
from matplotlib import colors
from matplotlib import cm as cmx
from msdm.core.problemclasses.mdp.quicktabularmdp import QuickTabularMDP
from msdm.core.distributions import DictDistribution, UniformDistribution, DeterministicDistribution


def make_russell_norvig_grid(
        discount_rate=0.95,
        slip_prob=0.8,
        goal_reward=1,
        lava_penalty=-1,
        step_cost=-0.04,
):
    # g is goal, x is lava, # is the wall, and s in the agent's starting location
    grid_string = '''
        ...g
        .#.x
        s...
    '''
    grid = [list(r.strip()) for r in grid_string.split('\n') if len(r.strip()) > 0]

    # states are xy tuples indexed at 0
    loc_to_feature = {}
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            loc_to_feature[(x, len(grid) - 1 - y)] = c

    def initial_state_dist():
        return UniformDistribution([s for s, f in loc_to_feature.items() if f == 's'])

    # actions are tuples: the following are right, left, up, down
    actions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # the two terminal states are the goal and lava
    def is_terminal(s):
        return loc_to_feature[s] == 'g' or loc_to_feature[s] == 'x'

    # standard reward function assigns +1 for goal, -1 for lava, -0.04 for all other states
    def reward(s, a, ns):
        if loc_to_feature.get(ns, '') == 'g':
            return goal_reward
        elif loc_to_feature.get(ns, '') == 'x':
            return lava_penalty
        elif loc_to_feature.get(ns, '') == '.' or loc_to_feature.get(ns, '') == 's':
            return step_cost
        raise Exception("Invalid state for reward", ns)

    # valid locations are those on the board, excluding the wall location at (1, 1)
    def is_valid_loc(s):
        return (s in loc_to_feature) and (loc_to_feature[s] != '#')

    def is_x_move(a):
        return a[0] in [1, -1]

    def apply_op(s, op):
        ns = (s[0] + op[0], s[1] + op[1])
        # 'bouncing' if the next state is a wall or off the grid
        if not is_valid_loc(ns):
            return s
        return ns

    def next_state_dist(s, a):
        if is_terminal(s):
            return DeterministicDistribution(s)

        # the two potential 'slips'
        slip_op1 = (0, -1) if is_x_move(a) else (-1, 0)
        slip_op2 = (0, 1) if is_x_move(a) else (1, 0)

        # the next state distribution
        ns_dist = defaultdict(float)

        # the intended next state
        int_ns = apply_op(s, a)
        ns_dist[int_ns] += slip_prob

        # adding the slips to the next state distribution
        slip_ns1 = apply_op(s, slip_op1)
        ns_dist[slip_ns1] += (1 - slip_prob) / 2
        slip_ns2 = apply_op(s, slip_op2)
        ns_dist[slip_ns2] += (1 - slip_prob) / 2
        return DictDistribution(ns_dist)

    def state_string(s):
        s = (s[0], len(grid) - 1 - s[1])
        new_grid = copy.deepcopy(grid)
        new_grid[s[1]][s[0]] = '@'
        return '\n'.join([''.join(r) for r in new_grid])

    gw = QuickTabularMDP(
        next_state_dist=next_state_dist,
        reward=reward,
        actions=actions,
        initial_state_dist=initial_state_dist,
        is_terminal=is_terminal,
        discount_rate=discount_rate
    )

    gw.state_string = state_string

    gw.action_to_string = {
        (1, 0): "right",
        (-1, 0): "left",
        (0, -1): "down",
        (0, 1): "up"
    }

    return gw


def visualize_rn_grid_policy(
    policy,
    ax,
    arrow_width=.1,
):
    # default 
    value_range=[0, 1]
    color_value_func=lambda v: 'k'
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-0.1, 3.1)
    ax.set_aspect('equal')

    # set up value range
    allvals = sum(
        [list(av.values()) for s, av in policy.items()],
        [])
    absvals = [abs(v) for v in allvals]
    absvmax = max(absvals)
    if value_range is None:
        value_range = [-absvmax, absvmax]
    else:
        absvmax = max([abs(v) for v in value_range])
    vmin, vmax = value_range

    # choose colors
    if isinstance(color_value_func, str):
        colorrange = plt.get_cmap(color_value_func)
        color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
        color_value_map = cmx.ScalarMappable(norm=color_norm,
                                                cmap=colorrange)
        color_value_func = lambda v: color_value_map.to_rgba(v)

    # set up dict for plotting
    to_plot = {}
    for s, av in policy.items():
        to_plot[s] = {}
        for a, prob in av.items():
            to_plot[s][a] = prob

    # plot
    for s, av in to_plot.items():
        x, y = s
        for a, v in av.items():
            dx, dy = a
            arrowColor = color_value_func(v)
            mag = abs(v) / absvmax
            mag *= .5
            if (dx != 0) or (dy != 0):
                patch = Arrow(x + .5, y + .5, dx * mag, dy * mag,
                                width=arrow_width,
                                color=arrowColor)
            else:
                patch = Circle((x + .5, y + .5), radius=mag * .9,
                                fill=False, color=arrowColor)
            ax.add_patch(patch)
    
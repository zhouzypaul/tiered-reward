from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from msdm.core.distributions.dictdistribution import DeterministicDistribution, DictDistribution, \
    UniformDistribution
from msdm.core.problemclasses.mdp.quicktabularmdp import QuickTabularMDP


def make_puddle_world(discount_rate=.99, 
                        slip_prob=0., 
                        step_cost=0, 
                        puddle_cost=-0.3,
                        goal_reward=1, 
                        custom_rewardf=None,
                        grid=None, 
                        feature_groups=('s.', 'g', 'p'), 
                        term_features='g', 
                        verbose=False,
                      ):
    """
    Note: 
    s = start location
    . = background state
    p = puddle state (or lava)
    g = goal state
    
    :param discount_rate: float 
    :param custom_rewardf: None | function(s, a) -> float - if you want a specific reward function other than step_cost,
    goal_reward, or lava_penalty, use this
    :param step_cost: float
    :param goal_reward: float
    :param lava_penalty: float
    :param grid: str | None - if you want to specify a puddle architecture other than the default (see below - grid_string)
    use this. Otherwise, leave as None to use the default.
    :param slip_prob: the probability of not slipping
    :param feature_groups: str - controls what features are grouped together for the purposes of the LP assigning rewards. 
    A list means each state will be a part of that feature group. 'all' means each state is its own feature. 'columns' assigns
    features along columns of the world. 
    :param term_features: str - include each letter for each feature that is terminal.
    - default is 'g', meaning only the goal state is terminal
    - to make puddles (lava) AND goal terminal, you would set term_features='pg'
    just include all letters of features that are terminal
    """

    # error checking inputs
    assert 0 < discount_rate < 1, "discount_rate must be between 0 and 1"
    assert 0 <= slip_prob <= 1, "slip_prob must be between 0 and 1"
    if custom_rewardf is None:
        assert step_cost is not None, 'must specify step_cost if custom_rewardf is None'
        assert goal_reward is not None, 'must specify goal_reward if custom_rewardf is None'
        assert puddle_cost is not None, 'must specify puddle cost if custom_rewardf is None'
    else:
        print("Custom reward specified; ignoring step_cost, goal_reward, and puddle cost arguments")

    # default world
    grid_string = grid or """
        ...p......
        ...p.p.g.p
        .s.p.p...p
        ...p.p..p.
        ...p.ppp..
        ...p......
        ...p......
        ....pppp..
        .ppp......
        ..........
    """

    action_idx_to_tuple = { 0: (-1, 0), 1: (0, -1), 2: (0, 1), 3: (1, 0), }
    action_tuple_to_idx = { v: k for k, v in action_idx_to_tuple.items() }

    grid = [list(r.strip()) for r in grid_string.split('\n') if len(r.strip())]
    loc_to_feature = { (y, len(grid) + ~x): c for x, row in enumerate(grid) for y, c in enumerate(row) }
    if verbose: print(*loc_to_feature.items(), sep='\n')

    walls = set(k for k, i in loc_to_feature.items() if i == '#')
    states = loc_to_feature.keys()
    nonwalls = sorted(states - walls)
    sind_nw = { k: i for i, k in enumerate(nonwalls) }
    rstate_ind = { sind_nw[k]: k for k in sind_nw.keys() }

    features = sorted(loc_to_feature.values())
    walls = set(k for k, i in loc_to_feature.items() if i == '#')
    initial_states = set(k for k, i in loc_to_feature.items() if i == 's')
    sind_all = { k: i for i, k in enumerate(sorted(loc_to_feature.keys())) }

    actions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def initial_state_dist(): 
        return UniformDistribution([s for s, f in loc_to_feature.items() if f == 's'])

    def is_terminal(s):
        if term_features is not None:
            x = loc_to_feature[s] in term_features
            # print(s, x)
            return x
        return False

    def reward(s, a, ns):
        try: 
            tuple(ns)
        except TypeError: 
            ns = rstate_ind[ns]
            print('twas fucked up', s, 'into', ns)

        # our custom reward function
        if custom_rewardf is not None: 
            return custom_rewardf[sind_nw[ns]]

        if loc_to_feature.get(ns, '') in '.s': 
            return step_cost
        elif loc_to_feature.get(ns, '') == 'g': 
            return goal_reward
        elif loc_to_feature.get(ns, '') in 'p': 
            return puddle_cost
        else: 
            raise ValueError(f'Invalid rf or state for:\n{ns}')

    def is_valid_loc(s): 
        return s in loc_to_feature and loc_to_feature[s] != '#'

    def is_x_move(a): 
        return a[0] != 0

    def apply_op(s, op):  # moves to next state if valid, otherwise stays in state
        if isinstance(s, int): 
            print(s, ' is int')
            s = rstate_ind[s]
        ns = (s[0] + op[0], s[1] + op[1])

        if is_valid_loc(ns):
            return ns
        else:
            return s

    def next_state_dist(s, a):
        if is_terminal(s): 
            return DeterministicDistribution(s)
        if slip_prob == 1: 
            return DeterministicDistribution(apply_op(s, a))

        ns_dist = defaultdict(float)  # next state distribution
        int_ns = apply_op(s, a)
        ns_dist[int_ns] += slip_prob

        slip_op1 = (0, -1) if is_x_move(a) else (-1, 0)  # 'slipping'
        slip_op2 = (0, 1) if is_x_move(a) else (1, 0)
        slip_ns1 = apply_op(s, slip_op1)
        slip_ns2 = apply_op(s, slip_op2)
        ns_dist[slip_ns1] += (1 - slip_prob) / 2
        ns_dist[slip_ns2] += (1 - slip_prob) / 2
        return DictDistribution(ns_dist)

    def is_goal(s):
        return loc_to_feature[s] == 'g'

    gw = QuickTabularMDP(
            discount_rate=discount_rate, actions=actions, reward=reward,
            initial_state_dist=initial_state_dist, next_state_dist=next_state_dist, is_terminal=is_terminal,
    )
    gw.height, gw.width = len(grid), len(grid[0])

    gw.locFeatures = gw.loc_to_feature = loc_to_feature
    feature_groups = feature_groups or ['s.', 'g', 'p']
    try: 
        feature_groups.sort()
    except AttributeError: 
        pass
    gw.features = feature_groups
    if verbose: print('feature groups are', gw.features)

    gw.feature_set = features
    gw.wall_states = gw.walls = walls
    gw.nonwalls = nonwalls
    gw.sindall = gw.state_ind_all = sind_all
    gw.state_ind = sind_nw
    gw.rstate_ind = rstate_ind
    gw.initial_states = gw.initial = initial_states

    if term_features is not None:
        gw.absorbing_states = gw.tsx = set(gw.state_index[k] for k, i in loc_to_feature.items() if i in term_features)
    else:
        gw.absorbing_states = gw.tsx = set()

    def verify_feature_matrix(fm):
        ffm = fm.copy().sum(0)
        if not (ffm == 1).all(): 
            raise ValueError(f'Feature matrix rows must sum to 1, received {fm} which sums {ffm}')
        if fm.shape[1] != len(nonwalls): 
            raise ValueError(f'Feature matrix must have {len(nonwalls)} columns, not {fm}')
        return fm

    def get_feat_mat():
        nonlocal feature_groups
        if 'all' in feature_groups: return np.eye(len(nonwalls))
        elif 'columns' in feature_groups:
            nfeats = gw.width
            fm = np.zeros((nfeats + 1, len(nonwalls)))
            gstateind = sind_nw[gw.absorbing_states.pop()]

            for i in range(nfeats):
                fm[i, :] = [1 if (s[0] == i and sind_nw[s] != gstateind) else 0 for s in nonwalls]
            fm[-1, gstateind] = 1
            return fm
        else:
            if verbose: print("Using feature groups:", *feature_groups)
            return np.array(
                    [[1 if loc_to_feature[s] in f else 0 for s in nonwalls] for f in feature_groups if
                     f != '#'],
                    dtype=np.uint8)

    gw.ff = verify_feature_matrix(get_feat_mat())
    if verbose: print('final gw.ff:\n', gw.ff)

    def plot(all_elements=False, plot_walls=True, plot_initial_states=True, plot_absorbing_states=True,
             feature_colors=None):
        if all_elements: plot_initial_states, plot_absorbing_states = True, True
        featurecolors = feature_colors or {'s.': 'green', 'g': 'red', 'p': 'blue'}
        _, ax = plt.subplots(1, 1, figsize=(gw.width, gw.height))
        gwp = GridWorldPlotter(gw=gw, ax=ax)
        gwp.plot_features(featurecolors)
        if plot_walls: gwp.plot_walls()
        if plot_initial_states: gwp.plot_initial_states()
        if plot_absorbing_states: gwp.plot_absorbing_states()
        gwp.plot_outer_box()
        return gwp
    
    gw.plot = plot

    for i, k in enumerate(gw.action_list):
        if action_tuple_to_idx[k] != i: raise ValueError(f'action index mismatch at {k}: {action_tuple_to_idx[k]}, {i}')

    gw.atti = action_tuple_to_idx
    gw.aitt = action_idx_to_tuple
    gw.action_to_string = {
        (1, 0):  'right',
        (-1, 0): 'left',
        (0, -1): 'down',
        (0, 1):  'up'
    }

    return gw



#########  PLOTTING  ##############################################################################################
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from matplotlib.patches import Rectangle, Arrow, Circle
from typing import Mapping, Union, Callable
from frozendict import frozendict


class GridWorldPlotter:


    def get_contrast_color(self, color):
        r, g, b = colors.to_rgb(color)
        luminance = (0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2) ** .5
        if luminance < .7:
            return "white"
        return 'grey'


    def __init__(self, gw, ax: plt.Axes):
        self.gw = gw
        self.ax = ax
        self.ax.axis('off')
        self.ax.set_xlim(-0.1, self.gw.width + .1)
        self.ax.set_ylim(-0.1, self.gw.height + .1)
        # self.ax.axis('equal')
        ax.set_aspect('equal')

    def plot_features(self, featurecolors, edgecolor='darkgrey',
                      reward=False) -> "GridWorldPlotter":
        """Plot gridworld features"""
        ss = self.gw.state_list
        self.featurecolors = featurecolors
        for s in ss:
            # if self.gw.is_terminal(s):
            #     continue
            xy = (s[0], s[1])
            f = self.gw.locFeatures.get(s, '.')[0]
            color = featurecolors.get(f, 'w')
            square = Rectangle(xy, 1, 1,
                               facecolor=color,
                               edgecolor=edgecolor,
                               linewidth=2)
            self.ax.add_patch(square)
        return self

    def plot_outer_box(self):
        outerbox = Rectangle((0, 0), self.gw.width, self.gw.height,
                             fill=False, edgecolor='black',
                             linewidth=2)
        self.ax.add_patch(outerbox)
        return self

    def plot_walls(self, facecolor='k', edgecolor='darkgrey'):
        for ws in self.gw.walls:
            xy = (ws[0], ws[1])
            square = Rectangle(xy, 1, 1,
                               facecolor=facecolor,
                               edgecolor=edgecolor,
                               linewidth=2)
            self.ax.add_patch(square)
        return self

    def plot_initial_states(self, markersize=15):
        for s in self.gw.initial_states:
            x, y = s[0], s[1]
            self.ax.plot(x + .5, y + .5,
                         markeredgecolor='cornflowerblue',
                         # marker='o',
                         markersize=markersize,
                         markeredgewidth=2,
                         fillstyle='none')
        return self

    def plot_absorbing_states(self, markersize=15):
        for s in self.gw.absorbing_states:
            if isinstance(s, int):
                s = self.gw.state_list[s]
            x, y = s[0], s[1]
            self.ax.plot(x + .5, y + .5,
                         markeredgecolor='cornflowerblue',
                         # marker='x',
                         markersize=markersize,
                         markeredgewidth=2)

    def pR(self, R, show_numbers=True, color_value_func="bwr_r",
           fontsize=7):
        vamax = np.abs(R).max()
        vmin, vmax = [-vamax, vamax]
        colorrange = plt.get_cmap(color_value_func)
        color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
        color_value_map = cmx.ScalarMappable(norm=color_norm,
                                             cmap=colorrange)
        color_value_func = lambda v: color_value_map.to_rgba(v)
        state_ind = self.gw.rstate_ind
        for i, s in enumerate(R):
            xy = state_ind[i]
            color = color_value_func(s)
            square = Rectangle(xy, 1, 1,
                               color=color,
                               ec='k', lw=2)
            self.ax.add_patch(square)
            if show_numbers:
                self.ax.text(xy[0] + .5, xy[1] + .15,
                             f"{s : .4f}",
                             fontsize=fontsize,
                             color=self.get_contrast_color(color),
                             horizontalalignment='center',
                             verticalalignment='center')

    def plot_state_map(self,
                       state_map: Mapping,
                       plot_over_walls=False,
                       fontsize=10,
                       show_numbers=True,
                       value_range=None,
                       show_colors=True,
                       is_categorical=False,
                       color_value_func="bwr_r") -> "GridWorldPlotter":
        DISTINCT_COLORS = [
            '#A9A9A9', '#e6194b', '#3cb44b',
            '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#46f0f0', '#f032e6',
            '#bcf60c', '#fabebe', '#008080',
            '#e6beff', '#9a6324', '#fffac8',
            '#800000', '#aaffc3', '#808000',
            '#ffd8b1', '#000075', '#808080',
            '#ffffff', '#000000'
        ]

        if len(state_map) == 0:
            return self
        # state map - colors / numbers
        vmax_abs = max(abs(v) for k, v in state_map.items())
        if value_range is None:
            value_range = [-vmax_abs, vmax_abs]
        vmin, vmax = value_range
        if is_categorical:
            color_value_func = lambda i: DISTINCT_COLORS[
                int(i) % len(DISTINCT_COLORS)]
        elif isinstance(color_value_func, str):
            colorrange = plt.get_cmap(color_value_func)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            color_value_map = cmx.ScalarMappable(norm=color_norm,
                                                 cmap=colorrange)
            color_value_func = lambda v: color_value_map.to_rgba(v)
        for s, v in state_map.items():
            if self.gw.is_terminal(s):
                continue
            if (not plot_over_walls) and (s in self.gw.walls):
                continue
            if isinstance(s, (dict, frozendict)):
                xy = s[0], s[1]
            elif isinstance(s, tuple) or isinstance(s, list):
                xy = s
            else:
                raise Exception("unknown state representation")

            color = 'w'
            if show_colors:
                color = color_value_func(v)
                square = Rectangle(xy, 1, 1,
                                   color=color,
                                   ec='k', lw=2)
                self.ax.add_patch(square)
            if show_numbers:
                self.ax.text(xy[0] + .5, xy[1] + .5,
                             f"{v : .3f}",
                             fontsize=fontsize,
                             color=self.get_contrast_color(color),
                             horizontalalignment='center',
                             verticalalignment='center')
        return self

    def plot_state_action_map(self,
                              state_action_map: Mapping,
                              plot_over_walls=False,
                              value_range=None,
                              color_value_func: Union[Callable, str] = "bwr_r",
                              arrow_width=.1,
                              show_numbers=False,
                              numbers_kw=None,
                              visualization_type="arrow"
                              ) -> "GridWorldPlotter":
        # set up value range
        allvals = sum(
                [list(av.values()) for s, av in state_action_map.items()],
                [])
        absvals = [abs(v) for v in allvals]
        absvmax = max(absvals)
        if value_range is None:
            value_range = [-absvmax, absvmax]
        else:
            absvmax = max([abs(v) for v in value_range])
        vmin, vmax = value_range

        if isinstance(color_value_func, str):
            colorrange = plt.get_cmap(color_value_func)
            color_norm = colors.Normalize(vmin=vmin, vmax=vmax)
            color_value_map = cmx.ScalarMappable(norm=color_norm,
                                                 cmap=colorrange)
            color_value_func = lambda v: color_value_map.to_rgba(v)

        # format mapping for plotting
        if isinstance(next(iter(state_action_map)), (dict, frozendict)):
            to_plot = { }
            for s, a_v in state_action_map.items():
                if self.gw.is_terminal(s):
                    continue
                if (not plot_over_walls) and (s in self.gw.walls):
                    continue
                s_ = (s[0], s[1])
                to_plot[s_] = { }
                for a, v in a_v.items():
                    a_ = (a.get('dx', 0), a.get('dy', 0))
                    to_plot[s_][a_] = v
        elif isinstance(next(iter(state_action_map)), (tuple, list)):
            to_plot = { }
            for s, a_v in state_action_map.items():
                if self.gw.is_terminal(s):
                    continue
                if (not plot_over_walls) and (s in self.gw.walls):
                    continue
                to_plot[s] = { **a_v }
        else:
            raise Exception("unknown state representation")

        def plot_state_action_map_as_arrows():
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
                    self.ax.add_patch(patch)

        def plot_state_action_map_as_triangles():
            sav_params = []
            for (x, y), a_v in to_plot.items():
                for (dx, dy), v in a_v.items():
                    vertices = {
                        (0, 0):  [(.3, .3), (.7, .3), (.7, .7), (.3, .7)],
                        (-1, 0): [(.5, .5), (0, 0), (0, 1)],
                        (1, 0):  [(.5, .5), (1, 0), (1, 1)],
                        (0, 1):  [(.5, .5), (0, 1), (1, 1)],
                        (0, -1): [(.5, .5), (0, 0), (1, 0)],
                    }[(dx, dy)]
                    vertices = [(x + ix, y + iy) for ix, iy in vertices]
                    av_params = list(zip(*vertices)) + [
                        colors.to_hex(color_value_func(v))]
                    if (dx, dy) == (0, 0):
                        sav_params.extend(av_params)
                    else:
                        sav_params = av_params + sav_params
            _ = self.ax.fill(*sav_params)

        def plot_state_action_map_numbers():
            for (x, y), a_v in to_plot.items():
                for (dx, dy), v in a_v.items():
                    ann_params = {
                        (0, 0):  { "xy": (.5, .5), "ha": "center",
                                   "va": "center" },
                        (-1, 0): { "xy": (.05, .5), "ha": "left",
                                   "va": "center" },
                        (1, 0):  { "xy": (.95, .5), "ha": "right",
                                   "va": "center" },
                        (0, 1):  { "xy": (.5, .95), "ha": "center", "va": "top" },
                        (0, -1): { "xy": (.5, .05), "ha": "center",
                                   "va": "bottom" }
                    }[(dx, dy)]
                    ann_params['xy'] = (
                        ann_params['xy'][0] + x, ann_params['xy'][1] + y)
                    contrast_color = self.get_contrast_color(color_value_func(v))
                    contrast_color = contrast_color if contrast_color == 'white' else 'black'
                    self.ax.annotate(text=f"{v:+.1f}",
                                     **{ **dict(color=contrast_color),
                                         **numbers_kw, **ann_params })

        if "arrow" in visualization_type:
            plot_state_action_map_as_arrows()
        elif "triangle" in visualization_type:
            plot_state_action_map_as_triangles()
        else:
            raise ValueError("Unknown visualization type")
        if show_numbers:
            if numbers_kw is None:
                numbers_kw = dict(fontsize=10)
            if "arrow" in visualization_type:
                numbers_kw['color'] = "k"
            plot_state_action_map_numbers()
        return self

    def pP(self, policy, arrow_color='black'):
        for s, av in policy.items():
            x, y = s
            if self.gw.is_terminal(s):
                continue
            if isinstance(av, str):
                print('string. continuing.', s, av)
                continue
            if isinstance(av, (tuple, list)):
                print('is list')
                for i, act in enumerate(av):
                    dx, dy = act
                    print(i, act)
                    self.ax.add_patch(Arrow(x + .5, y + .5, dx * (.25 if not i else .15), dy * (.25 if not i else
                                                                                                .15),
                                            width=(arrow_width := .1
                                                   # if not i else .05
                                                   ),
                                            color=arrow_color))
            else:
                # is a dict
                for a, v in av.items():
                    dx, dy = a
                    # mag = abs(v) / absvmax
                    # mag *= .5
                    if (dx != 0) or (dy != 0):
                        # arrow_color = get_contrast_color(self.featurecolors[self.gw.locFeatures[s]])
                        # print(arrow_color)
                        self.ax.add_patch(Arrow(x + .5, y + .5, dx * .25, dy * .25,
                                                width=(arrow_width := .1),
                                                color=arrow_color))
                    else:
                        raise NotImplementedError

    def title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)
        return self

    def annotate(self,
                 s, a=None,
                 text="", outlinewidth=0.04,
                 outlinecolor='black', fontsize=10, ha='center', va='center', offset=None, **kwargs
                 ):
        if not isinstance(text, str):
            print(f"text is not string. got {text} ; continuing");
            return
        offset = offset or .5
        try: ox, oy = offset
        except TypeError: ox, oy = offset, offset
        if 's' in text or 'S' in text or 'p' in text or 'P' in text:
            ox, oy = .25, .25
            fontsize = 7
        kwargs = {
            'fontsize': fontsize,
            'ha':       ha,
            'va':       va,
            **kwargs }

        if isinstance(s, (tuple, list)):
            s = s
        elif isinstance(s, (dict, frozendict)):
            s = (s[0], s[1])
        text = self.ax.text(s[0] + oy, s[1] + ox, text, **kwargs)
        # if outlinewidth > 0:
        #     text.set_path_effects([
        #         path_effects.Stroke(linewidth=outlinewidth,
        #                             foreground=outlinecolor),
        #         path_effects.Normal()
        #     ])
        return self

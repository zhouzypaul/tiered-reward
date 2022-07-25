from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
from matplotlib.patches import Rectangle
from msdm.domains.gridworld.plotting import get_contrast_color

from reward.msdm_utils import get_ordered_state_action_list, ModifiedValueIteration


def plot_grid_reward(gw, plot_name_prefix, results_dir, reward_vec=None, plot_over_walls=False):
    """
    visualize the reward function of a gridworld
    """
    reward_vec = gw.reward_vector if reward_vec is None else reward_vec
    states, _ = get_ordered_state_action_list(gw)

    gw_plot = gw.plot().title(f'{plot_name_prefix}: reward function')

    # handle reward range
    if len(reward_vec) == 0:
        return gw_plot
    rmax_abs = abs(max(reward_vec))
    reward_range = [-rmax_abs, rmax_abs]
    rmin, rmax = reward_range

    # choose colors
    color_range = plt.get_cmap('bwr_r')
    color_norm = colors.PowerNorm(gamma=0.5, vmin=rmin, vmax=rmax)
    color_value_map = cmx.ScalarMappable(norm=color_norm, cmap=color_range)
    color_value_func = lambda v: color_value_map.to_rgba(v)

    # loop over rewards to draw them
    for i, r in enumerate(reward_vec):
        # find out which state reward belong to
        s = states[i]
        xy = s['x'], s['y']
        # paint color
        if plot_over_walls or (s not in gw.walls):
            color = color_value_func(r)
            square = Rectangle(xy, 1, 1, color=color, ec='k', lw=2)
            gw_plot.ax.add_patch(square)
        # show value
        gw_plot.ax.text(xy[0] + .5, xy[1] + .5,
                             f"{r : .2f}",
                             fontsize=10,
                             color=get_contrast_color(color),
                             horizontalalignment='center',
                             verticalalignment='center')
    
    # save plot
    file_path = results_dir.joinpath(plot_name_prefix + "_rewards.jpg")
    plt.savefig(fname=file_path)
    plt.close()


def visualize_grid_world_and_policy(gw, plot_name_prefix, results_dir, plot_simple=False):
    """
    for a msdm.GridWorld, visualize three things:
    1) the gridworld itself
    2) the trajectory sampled from its optimal policy
    3) the value function and policy from the gridworld
    the 3 plots are saved to results_dir
    """
    # plot the grid world
    if plot_simple:
        gw.plot().title(f"{plot_name_prefix}: Grid World Plot")
        gw_file_path = results_dir.joinpath(plot_name_prefix + '.jpg')
        plt.savefig(fname=gw_file_path)
        plt.close()

    # plot the value function and policy
    value_iter = ModifiedValueIteration()
    planning_result = value_iter.plan_on(gw)
    policy = planning_result.policy
    fig, axes = plt.subplots(2, 1)
    gw.plot(ax=axes[0]).plot_state_map(planning_result.valuefunc).title("Value Function")
    gw.plot(ax=axes[1]).plot_policy(policy).title("Policy")
    # plt.tight_layout()
    vf_policy_file_path = results_dir.joinpath(plot_name_prefix + "_value_func_and_policy.jpg")
    plt.savefig(fname=vf_policy_file_path)
    plt.close()

    # plot the trajectories
    trajectories = [policy.run_on(gw, max_steps=100) for _ in range(20)]
    traj_plt = gw.plot(ax=plt.gca()).title(f"{plot_name_prefix}: sampled traj from optimal policy by VI")
    for traj in trajectories:
        traj_plt.plot_trajectory(traj.state_traj)

    traj_file_path = results_dir.joinpath(plot_name_prefix + "_traj.jpg")
    plt.savefig(fname=traj_file_path)
    plt.close()

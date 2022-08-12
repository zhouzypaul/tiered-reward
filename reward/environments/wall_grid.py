import numpy as np

from reward.environments.slippery_grid import SlipperyGrid


class WallGrid(SlipperyGrid):
    """
    Basically a grid with lava and goal, but hey, there are walls in the middle 
    """
    def __init__(self, 
                tile_array=None, 
                feature_rewards=None, 
                step_cost=-1,
                success_prob=0.8,
                discount_rate=0.9, 
                custom_rewards=None
        ):
        if tile_array is None:
            tile_array=[
                '......#...s',
                '........###',
                'xxxx.....x#',
                '......##...',
                '......g....',
            ]
        # hand-code the distance function of each state to goal
        wall_grid_location_distance = [
            [10, 9, 8, 7, 6, 5, np.nan, 7, 8, 9, 10],
            [9, 8, 7, 6, 5, 4, 5, 6, np.nan, np.nan, np.nan],
            [np.inf, np.inf, np.inf, np.inf, 4, 3, 4, 5, 4, np.inf, np.nan], 
            [7, 6, 5, 4, 3, 2, np.nan, np.nan, 3, 4, 5], 
            [6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
        ]

        super().__init__(
            tile_array=tile_array,
            tile_distance=wall_grid_location_distance,
            feature_rewards=feature_rewards,
            absorbing_features=('g', 'x'),
            wall_features=('#'),
            default_features=('.', ),
            initial_features=('s', ),
            step_cost=step_cost,
            success_prob=success_prob,
            discount_rate=discount_rate,
            custom_rewards=custom_rewards,
        )


def make_wall_grid(discount_rate, success_prob, goal_reward, step_cost, lava_penalty, custom_rewards=None):
    gw = WallGrid(
        feature_rewards={
            'g': goal_reward,
            'x': lava_penalty,
        },
        step_cost=step_cost,
        success_prob=success_prob,
        discount_rate=discount_rate,
        custom_rewards=custom_rewards,
    )
    return gw


if __name__ == "__main__":
    # for debugging
    import matplotlib.pyplot as plt
    gw = make_wall_grid(discount_rate=0.9, success_prob=1, goal_reward=1, step_cost=-0.1, lava_penalty=-1)
    gw.plot()

    print(gw.location_distances)
    print(gw.state_list)

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

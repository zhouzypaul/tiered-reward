import numpy as np

from reward.environments.slippery_grid import SlipperyGrid


class FrozenLake(SlipperyGrid):
    """
    frozen lake, the same as the toy text environment in OpenAI gym.
    Checkout the gym documentation for more information on how the gridworld works
    
    Basically, it's a reward.environments.slippery_grid.SlipperyGrid
    However, there are holes (feature h) on the frozen lake, which will trap you inside.
    
    The success probabiliy is always 1/3, with 1/3 probability of slipping to either side
    """
    def __init__(self, 
                tile_array=None, 
                feature_rewards=None, 
                step_cost=0, 
                discount_rate=1, 
                custom_rewards=None
        ):
        if tile_array is None:
            tile_array = [
                's.......',
                '........',
                '...h....',
                '.....h..',
                '...h....',
                '.hh...h.',
                '.h..h.h.',
                '...h...g',
            ]
        # hand-code the distance function of each state to goal
        frozen_lake_location_distance = [
            [14, 13, 12, 11, 10, 9, 8, 7], 
            [13, 12, 11, 10, 9, 8, 7, 6], 
            [12, 11, 10, np.inf, 8, 7, 6, 5],
            [11, 10, 9, 8, 7, np.inf, 5, 4],
            [12, 11, 10, np.inf, 6, 5, 4, 3],
            [13, np.inf, np.inf, 6, 5, 4, np.inf, 2],
            [12, np.inf, 8, 7, np.inf, 3, np.inf, 1],
            [11, 10, 9, np.inf, 3, 2, 1, 0],
        ]

        super().__init__(
            tile_array=tile_array,
            tile_distance=frozen_lake_location_distance,
            feature_rewards=feature_rewards,
            absorbing_features=('g', 'h'),
            wall_features=(),
            default_features=('.', ),
            initial_features=('s', ),
            step_cost=step_cost,
            success_prob=1/3,
            discount_rate=discount_rate,
            custom_rewards=custom_rewards,
        )


def make_frozen_lake(discount_rate, goal_reward, step_cost, hole_penalty, custom_rewards=None):
    gw = FrozenLake(
        feature_rewards={
            'g': goal_reward,
            'h': hole_penalty,
        },
        step_cost=step_cost,
        discount_rate=discount_rate,
        custom_rewards=custom_rewards,
    )
    return gw


if __name__ == "__main__":
    # for debugging
    import matplotlib.pyplot as plt
    gw = make_frozen_lake(discount_rate=0.9, goal_reward=1, step_cost=-0.1, hole_penalty=-1)
    gw.plot()

    print(gw.location_distances)

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

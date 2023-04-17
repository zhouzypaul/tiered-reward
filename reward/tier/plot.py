import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_flag_grid_learning_results(results_dir, gamma=0.9, q_init=0):
    """
    find the progress.csv inside results_dir and plot:
        1. the episodic lengths during learning
    the plot is averaged across different random seeds
    """
    csv_path = os.path.join(results_dir, 'progress.csv')
    assert os.path.exists(csv_path), csv_path

    df = pd.read_csv(csv_path)
    df = df[df.Episode <= 160]  # zoom in on earlier episodes
    df = df.sort_values(by='Reward Type')
    sns.lineplot(
        data=df,
        x='Episode',
        y='Episode Length',
        hue='Reward Type',
        style='Reward Type',
    )
    plt.title(r'Flag Grid: $\gamma=$' + str(gamma) + r', $Q_{init} = $' + str(q_init))
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken to Reach Goal')
    # plt.ylim(0, 700)
    # plt.legend(loc="upper right")
    save_path = os.path.join(os.path.dirname(results_dir), f'flag_grid_{gamma}.png')
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


def compare_goal_hitting_stat_with_different_tiers(results_dir, tiers_to_compare):
    """
    make a plot with different tiers on the x axis and time to goal on the y
    the progress.csv must be gathered from different directories, and only select the tier_reward data rows
    """
    # find all the csv
    env_name = os.path.basename(results_dir).split('-')[0]

    data = []
    for tier in tiers_to_compare:
        saving_dir = os.path.join(results_dir, f'{tier}-tier')
        csv_path = os.path.join(saving_dir, 'progress.csv')
        assert os.path.exists(csv_path), csv_path
        df = pd.read_csv(csv_path)
        df['tier'] = int(tier)
        df = df[['tier', 'time_till_goal', 'num_goals_hit', 'seed', 'Reward Type']]
        # separate each reward type
        clean_df = []
        for r_type, r_type_df in df.groupby('Reward Type'):
            r_type_df = r_type_df.copy().groupby('seed', as_index=False).mean()  # mean to remove repetitive data
            r_type_df['Reward Type'] = r_type
            clean_df.append(r_type_df)
        clean_df = pd.concat(clean_df, ignore_index=True)
        data.append(clean_df)
    data = pd.concat(data, ignore_index=True)

    # time till goal
    sns.lineplot(
        data=data,
        x='tier',
        y='time_till_goal',
        hue='Reward Type',
        style='Reward Type',
    )
    plt.title(f'{pretty_env_name(env_name)}')
    plt.xlabel('Number of Tiers')
    plt.ylabel('Steps Till First Reaching Goal')
    plt.xticks(tiers_to_compare)
    save_path = os.path.join(results_dir, 'learning_time.png')
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()

    # number of goals hit
    sns.lineplot(
        data=data,
        x='tier',
        y='num_goals_hit',
        hue='Reward Type',
        style='Reward Type',
    )
    plt.title(f'Number of Goals Hit: {pretty_env_name(env_name)}')
    plt.xlabel('Number of Tiers')
    plt.ylabel('Number of Goals Hit During Learning')
    plt.xticks(tiers_to_compare)
    save_path = os.path.join(results_dir, 'num_goals_hit.png')
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()

def pretty_env_name(env_name):
    by_letters = env_name.split('_')
    by_letters = [capitalize(s) for s in by_letters]
    return ' '.join(by_letters)

def capitalize(s):
    return s[0].upper() + s[1:]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", "-l", type=str, required=True, help="path to results dir")
    parser.add_argument("--plot_hitting_time", "-p", action="store_true", help="plot hitting time")
    parser.add_argument("--tiers", "-t", nargs="+", type=int, help="tiers to compare")
    args = parser.parse_args()

    if args.plot_hitting_time:
        compare_goal_hitting_stat_with_different_tiers(args.load, args.tiers)
    else:
        plot_flag_grid_learning_results(args.load)

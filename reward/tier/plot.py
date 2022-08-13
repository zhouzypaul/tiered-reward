import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_q_learning_results(results_dir):
    """
    find the progress.csv inside results_dir and plot:
        1. the episodic lengths during learning
    the plot is averaged across different random seeds
    """
    csv_path = os.path.join(results_dir, 'progress.csv')
    assert os.path.exists(csv_path)

    df = pd.read_csv(csv_path)
    sns.lineplot(
        data=df,
        x='Episode',
        y='Episode Length',
        hue='Reward Type',
    )
    plt.title('Episodic Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps taken to reach goal')
    save_path = os.path.join(results_dir, 'episodic_lengths.png')
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
    )
    plt.title(f'Learning Time: {env_name}')
    plt.xlabel('Tier')
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
    )
    plt.title(f'Number of Goals Hit: {env_name}')
    plt.xlabel('Tier')
    plt.ylabel('Number of Goals Hit During Learning')
    plt.xticks(tiers_to_compare)
    save_path = os.path.join(results_dir, 'num_goals_hit.png')
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


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
        plot_q_learning_results(args.load)

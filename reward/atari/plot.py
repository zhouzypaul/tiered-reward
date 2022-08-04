import os

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tier_visitation_over_time(saving_dir, num_tiers):
    """
    given a progress.csv visualize how the visitation of each tier changes over time
    """
    csv_path = os.path.join(saving_dir, 'progress.csv')
    assert os.path.exists(csv_path)
    df = pandas.read_csv(csv_path)

    for i in range(num_tiers):
        plt.plot(df.steps, df[f"tier_{i}_hitting_count"], label=f"tier_{i}")
    save_path = os.path.join(saving_dir, 'tier_visitation_over_time.png')
    plt.legend()
    plt.title('Tier visitation over time')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Hitting Count')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()


def plot_eval_stats(saving_dir, num_tiers):
    """
    given a progress_eval.csv, visualize
        1. how the visitation of each tier changes over time
        2. how the episodic reward changes over time
    """
    csv_path = os.path.join(saving_dir, 'progress_eval.csv')
    assert os.path.exists(csv_path)
    df = pandas.read_csv(csv_path)

    # average the n eval episodes
    averaged_df = df.groupby('steps', as_index=False).mean()

    # visitation
    for i in range(num_tiers):
        plt.plot(averaged_df.steps, averaged_df[f"eval_tier_{i}_hitting_count"], label=f"tier_{i}")
    save_path = os.path.join(saving_dir, 'eval_tier_visitation_over_time.png')
    plt.legend()
    plt.title('Eval tier visitation over time')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Hitting Count')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()

    # episode return
    plt.plot(averaged_df.steps, averaged_df.eval_episode_returns)
    save_path = os.path.join(saving_dir, 'eval_episode_returns.png')
    plt.title('Eval episode returns')
    plt.xlabel('Steps')
    plt.ylabel('Episode Returns')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()

    # episode length
    plt.plot(averaged_df.steps, averaged_df.eval_episode_lens)
    save_path = os.path.join(saving_dir, 'eval_episode_lens.png')
    plt.title('Eval episode lengths')
    plt.xlabel('Steps')
    plt.ylabel('Episode Lengths')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()



def plot_final_tier_visitation(exp_name, num_tiers):
    """
    go through each of the dirs, get progress.csv, and plot a bar plot 
    comparing the latest tier visitation of each tier
    """
    dirs_to_plot = [
        os.path.join('results', exp_name + f'-{num_tiers}-tiers')
    ]

    dfs = []
    for dir in dirs_to_plot:
        csv_path = os.path.join(dir, 'progress.csv')
        assert os.path.exists(csv_path)
        dfs.append(pandas.read_csv(csv_path))

    largest_common_timestep = min(df.steps.max() for df in dfs)

    for df in dfs:
        for tier in range(num_tiers):
            count = df[f'tier_{tier}_hitting_count'][df.steps == largest_common_timestep]

    # TODO


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', type=str, required=True, help='path to the directory containing the progress.csv')
    parser.add_argument('--num_tiers', '-t', type=int, default=5, help='number of tiers')
    args = parser.parse_args()

    plot_tier_visitation_over_time(args.load, num_tiers=args.num_tiers)
    plot_eval_stats(args.load, num_tiers=args.num_tiers)

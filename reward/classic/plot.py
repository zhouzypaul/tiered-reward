import os

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tier_visitation_over_time(base_dir, seed, num_tiers):
    """
    given a progress.csv visualize how the visitation of each tier changes over time
    """

    tiers_dir = os.path.join(base_dir, f'{num_tiers}-tiers')
    assert os.path.exists(tiers_dir)

    seed_dir = os.path.join(tiers_dir, 'tiered-reward', f'seed_{seed}')
    assert os.path.exists(seed_dir)

    csv_path = os.path.join(seed_dir, 'progress.csv')
    assert os.path.exists(csv_path)

    df = pandas.read_csv(csv_path)

    for i in range(num_tiers):
        plt.plot(df.steps, df[f"tier_{i}_hitting_count"], label=f"tier_{i}")
    save_path = os.path.join(seed_dir, 'tier_visitation_over_time.png')

    plt.legend()
    plt.title('Tier visitation over time')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Hitting Count')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()


def plot_different_reward_comparison(base_dir, num_tiers):
    """
    given a base dir, descend into the seed subdirs and find all the progress_eval.csv files
    plot:
        1. the episodic return of the tiered reward compared to original reward
        2. the visitation of each tier at the final timestep, tiered reward compared to original reward
    """

    tiers_dir = os.path.join(base_dir, f'{num_tiers}-tiers')
    assert os.path.exists(tiers_dir)

    original_reward_dir = os.path.join(tiers_dir, 'original-reward')
    tiered_reward_dir = os.path.join(tiers_dir, 'tiered-reward')
    assert os.path.exists(original_reward_dir)
    assert os.path.exists(tiered_reward_dir)

    def _gather_dataframes(type_dir):
        """
        given a dir, gather all the evaluation data from the sub seed directories
        """
        dfs = []
        for seed_dir in os.listdir(type_dir):
            subdir = os.path.join(type_dir, seed_dir)
            if not os.path.isdir(subdir):
                continue
            csv_path = os.path.join(subdir, 'progress_eval.csv')
            df = pandas.read_csv(csv_path)
            averaged_df = df.groupby('steps', as_index=False).mean()  # average the n eval episodes
            
            averaged_df['seed'] = int(seed_dir[-1])
            dfs.append(averaged_df)
        df = pandas.concat(dfs, ignore_index=True)
        df = averaged_df = df.groupby('steps', as_index=False).mean()  # average the n eval episodes
        return df
    
    tiered_reward_df = _gather_dataframes(tiered_reward_dir)
    tiered_reward_df['reward_type'] = 'tiered'
    original_reward_df = _gather_dataframes(original_reward_dir)
    original_reward_df['reward_type'] = 'original'

    # print(tiered_reward_df)
    # print(original_reward_df)

    data = pandas.concat([tiered_reward_df, original_reward_df], ignore_index=True)

    # episodic return
    sns.lineplot(
        data=data,
        x='steps',
        y='eval_episode_returns',
        hue='reward_type',
    )

    plt.xlabel('Steps')
    plt.ylabel('Episodic Returns')
    plt.title(f"{os.path.basename(base_dir)}")
    save_path = os.path.join(tiers_dir, 'compare_returns.png')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()

    # episodic length
    sns.lineplot(
        data=data,
        x='steps',
        y='eval_episode_lens',
        hue='reward_type',
    )
    plt.xlabel('Steps')
    plt.ylabel('Episodic Lengths')
    plt.title(f"{os.path.basename(base_dir)}")
    save_path = os.path.join(tiers_dir, 'compare_lens.png')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()

    # visitation of each tier
    window_size = 10
    tiered_final_data = tiered_reward_df.nlargest(window_size, 'steps').groupby('seed', as_index=False).mean()
    tiered_final_data['reward_type'] = 'tiered'
    original_final_data = original_reward_df.nlargest(window_size, 'steps').groupby('seed', as_index=False).mean()
    original_final_data['reward_type'] = 'original'
    final_data = pandas.concat([tiered_final_data, original_final_data], axis=0, ignore_index=True)
    visitation_data = []
    for tier in range(num_tiers):
        df = final_data[['steps', 'reward_type']].copy()
        df['tier'] = tier
        df['visitation'] = final_data[f'eval_tier_{tier}_hitting_count']
        visitation_data.append(df)
    visitation_data = pandas.concat(visitation_data, ignore_index=True)
    sns.lineplot(
        data=visitation_data,
        x='tier',
        y='visitation',
        hue='reward_type',
    )
    plt.xlabel('Tier')
    plt.ylabel('Visitation Count')
    plt.title(f"{os.path.basename(base_dir)}: Visitation of Trained Agent")
    save_path = os.path.join(tiers_dir, 'compare_visitation.png')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()


def plot_eval_stats(base_dir, seed, num_tiers):
    """
    given a progress_eval.csv, visualize
        1. how the visitation of each tier changes over time
        2. how the episodic reward changes over time
    """

    tiers_dir = os.path.join(base_dir, f'{num_tiers}-tiers')
    assert os.path.exists(tiers_dir)

    seed_dir = os.path.join(tiers_dir, 'tiered-reward', f'seed_{seed}')
    assert os.path.exists(seed_dir)

    csv_path = os.path.join(seed_dir, 'progress_eval.csv')
    assert os.path.exists(csv_path)
    df = pandas.read_csv(csv_path)

    # average the n eval episodes
    averaged_df = df.groupby('steps', as_index=False).mean()

    # visitation

    for i in range(num_tiers):
        plt.plot(averaged_df.steps, averaged_df[f"eval_tier_{i}_hitting_count"], label=f"tier_{i}")
    save_path = os.path.join(seed_dir, 'eval_tier_visitation_over_time.png')
    plt.legend()
    plt.title('Eval tier visitation over time')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Hitting Count')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()

    # episode return
    plt.plot(averaged_df.steps, averaged_df.eval_episode_returns)
    save_path = os.path.join(seed_dir, 'eval_episode_returns.png')
    plt.title('Eval episode returns')
    plt.xlabel('Steps')
    plt.ylabel('Episode Returns')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()

    # episode length
    plt.plot(averaged_df.steps, averaged_df.eval_episode_lens)
    save_path = os.path.join(seed_dir, 'eval_episode_lens.png')
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
    parser.add_argument('--dir', '-d', type=str, required=True, help='path to env-name directory')
    parser.add_argument('--num_tiers', '-t', type=int, default=5, help='number of tiers')
    parser.add_argument('--seed', '-s', type=int, required=True, help='seed for tier visitation plot')
    parser.add_argument('--compare_rewards', '-c', action='store_true', help='plot the stats of the tiered reward compared to original reward')
    args = parser.parse_args()

    if args.compare_rewards:
        plot_different_reward_comparison(args.dir, args.num_tiers)
    else:
        plot_tier_visitation_over_time(args.dir, args.seed, args.num_tiers)
        plot_eval_stats(args.dir, args.seed, args.num_tiers)

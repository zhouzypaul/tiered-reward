import os

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot_different_reward_comparison(base_dir, title=None):
    """
    given a base dir, descend into the subdirs, and the seed dirs,
    and find all the log.csv files
    plot:
        the episodic return of each reward type
        the episode length of each reward type
    """
    assert os.path.exists(base_dir)

    def _gather_seed_dataframes(dir):
        """
        given a dir, gather all the data from the seed directories
        """
        dfs = []
        for seed_dir in os.listdir(dir):
            subdir = os.path.join(dir, seed_dir)
            if not os.path.isdir(subdir):
                continue
            csv_path = os.path.join(subdir, 'log.csv')
            df = pandas.read_csv(csv_path)
            
            # convert frames to float so that plotting won't have many x-ticks
            df['frames'] = df['frames'].astype(int)
            
            df['seed'] = int(seed_dir[-1])
            dfs.append(df)
        df = pandas.concat(dfs, ignore_index=True)
        return df
    
    all_reward_df = []
    for reward_type in os.listdir(base_dir):
        reward_dir = os.path.join(base_dir, reward_type)
        if not os.path.isdir(reward_dir):
            continue
        reward_df = _gather_seed_dataframes(reward_dir)
        reward_df['reward_type'] = reward_type
        all_reward_df.append(reward_df)
    data = pandas.concat(all_reward_df, ignore_index=True)
    data = data.sort_values(by=['reward_type'])

    # episodic return
    sns.lineplot(
        data=data,
        x='frames',
        y='original_return_mean',
        hue='reward_type',
    )
    plt.xlabel('Frames')
    plt.ylabel('Episodic Returns')
    plt_title = title or os.path.basename(base_dir)
    plt.title(plt_title)
    save_path = os.path.join(base_dir, 'compare_returns.png')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', type=str, required=True, help='path to the directory containing the progress.csv')
    parser.add_argument('--title', '-t', type=str, default=None, help='title of the plot')
    args = parser.parse_args()

    plot_different_reward_comparison(args.load, args.title)

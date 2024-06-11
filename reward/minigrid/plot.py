import os

import wandb
import pandas as pd
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
            df = pd.read_csv(csv_path)
            
            # convert frames to float so that plotting won't have many x-ticks
            df['frames'] = df['frames'].astype(int)
            
            df['seed'] = int(seed_dir[-1])
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return df
    
    all_reward_df = []
    for reward_type in os.listdir(base_dir):
        reward_dir = os.path.join(base_dir, reward_type)
        if not os.path.isdir(reward_dir):
            continue
        reward_df = _gather_seed_dataframes(reward_dir)
        reward_df['reward_type'] = reward_type
        all_reward_df.append(reward_df)
    data = pd.concat(all_reward_df, ignore_index=True)
    
    
    
    if 'Multi' not in base_dir:
        palette = {'Tiered Reward (3 tiers)': 'tab:blue', 'Action Penalty' : 'tab:green', 'Tier Based Shaping' : 'tab:red'}
        data = data.replace('3-tiers','Tiered Reward (3 tiers)')
        data = data.replace('step_penalty', 'Action Penalty')
        data = data.replace('tier_based_shaping','Tier Based Shaping')
        data['Reward Type'] = data['reward_type']
    
        data = data.sort_values(by=['Reward Type'])
    else:
        
        palette = {3 : 'tab:blue', 5: 'tab:green', 7:'tab:orange', 9:'tab:red', 12:'tab:purple'}
        data = data.replace('3-tiers',3)
        data = data.replace('5-tiers',5)
        data = data.replace('7-tiers',7)
        data = data.replace('9-tiers',9)
        data = data.replace('12-tiers',12)
        data['Number of Tiers'] = data['reward_type']


    # episodic return
    
    sns.lineplot(
        data=data,
        x='frames',
        y='original_return_mean',
        hue='Reward Type' if 'Multi' not in base_dir else 'Number of Tiers',
        palette = palette
    )
    plt.xlabel('Steps')
    plt.ylabel('Episodic Returns')
    plt_title = title or os.path.basename(base_dir)
    plt.title(plt_title)
    save_path = os.path.join(base_dir, 'compare_returns.png')
    plt.savefig(save_path)
    print('saved to', save_path)
    plt.close()


def plot_from_wandb_data(data_path="project.pkl", title=None):
    
    if not os.path.exists(data_path):
        print("data file not found. Pull from wandb.")
    
        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs("zhouzypaul/tiered-reward-compare")

        summary_list, config_list, name_list, history_list = [], [], [], []
        for run in runs: 
            if run.state == "finished":
                # .summary contains the output keys/values for metrics like accuracy.
                #  We call ._json_dict to omit large files 
                summary_list.append(run.summary._json_dict)

                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config_list.append(
                    {k: v for k,v in run.config.items()
                    if not k.startswith('_')})

                # .name is the human-readable name of the run.
                name_list.append(run.name)
                
                # .history() returns the pandas dataframe containg the metrics
                history_list.append(run.history())

        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "history": history_list,
            })

        runs_df.to_pickle(data_path)
        
    else:
        print("found data file. Load from locally cached wandb file.")
        runs_df = pd.read_pickle(data_path)

    # iterate through runs_df and gather everything into a dataframe
    all_reward_df = []
    for i in range(len(runs_df['history'])):
        reward_df = runs_df['history'][i]
        config = runs_df['config'][i]
        # add the variables of config
        for key, value in config.items():
            if key not in ('frames'):
                reward_df[key] = value
        all_reward_df.append(reward_df)
    data = pd.concat(all_reward_df, ignore_index=True)
    
    # iterate over the different envs and plot
    envs = data['env'].unique()
    envs = [
        'MiniGrid-Empty-8x8-v0',
        'MiniGrid-DoorKey-5x5-v0',
        'MiniGrid-FourRooms-v0',
    ]
    data['Reward Type'] = data['reward_function']
    # change key names
    data = data.replace('step_penalty', 'Action Penalty')
    data = data.replace('tier_based_shaping','Tier Based Shaping')
    data = data.replace('tier','Tiered (Ours)')
    for env in envs:
        env_data = data[data['env'] == env]
        # episodic return
        sns.lineplot(
            data=env_data,
            x='frames',
            y='original_return_mean',
            hue='Reward Type',
            palette = {'Tiered (Ours)':'tab:green', 'Action Penalty': 'tab:blue', 'Tier Based Shaping': 'tab:orange'},
            errorbar="se",
        )
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Episodic Returns', fontsize=12)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        # increase the font size (legend only)
        if 'Empty' in env:
            plt.rcParams.update({'font.size': 12})
            # order the legend
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 2, 1]
            handles = [handles[idx] for idx in order]
            labels = [labels[idx] for idx in order]
            plt.legend(handles, labels, title='Reward Type', fontsize=12, loc='lower right')
        else:
            # don't show legend for other envs
            plt.legend().remove()
        # expand the figure lower
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt_title = title or f"{env.split('-')[1]}"
        if plt_title == 'Empty':
            plt_title = 'EmptyGrid'
        plt.title(plt_title)
        save_path = os.path.join(f'compare_returns_{env}.png')
        plt.savefig(save_path)
        print('saved to', save_path)
        plt.close()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', type=str, default=None, help='path to the directory containing the progress.csv')
    parser.add_argument('--title', '-t', type=str, default=None, help='title of the plot')
    parser.add_argument('--plot_from_wandb', '-w', action='store_true', default=False, 
                        help="plot from wandb instead of local files")
    args = parser.parse_args()

    if args.plot_from_wandb:
        plot_from_wandb_data(title=args.title)
    else:
        plot_different_reward_comparison(args.load, args.title)

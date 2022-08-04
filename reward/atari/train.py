import os
import time
import argparse
import logging
from collections import deque

import numpy as np
from pfrl import experiments
from pfrl import utils
from pfrl.experiments.evaluator import save_agent

from reward.atari.agent import make_agent
from reward.atari.env import make_batch_env, make_env
from reward.atari import logger as kvlogger


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def batch_run_eval_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns in a batch manner."""
    with agent.eval_mode():
        assert not agent.training
        assert (n_steps is None) != (n_episodes is None)

        logger = logger or logging.getLogger(__name__)
        num_envs = env.num_envs
        episode_returns = dict()
        episode_lengths = dict()
        episode_tier_hitting_count = dict()
        episode_indices = np.zeros(num_envs, dtype="i")
        episode_idx = 0
        for i in range(num_envs):
            episode_indices[i] = episode_idx
            episode_idx += 1
        episode_r = np.zeros(num_envs, dtype=np.float64)
        episode_len = np.zeros(num_envs, dtype="i")

        obss = env.reset()
        rs = np.zeros(num_envs, dtype="f")

        termination_conditions = False
        timestep = 0
        while True:
            # a_t
            actions = agent.batch_act(obss)
            timestep += 1
            # o_{t+1}, r_{t+1}
            obss, rs, dones, infos = env.step(actions)
            episode_r += rs
            episode_len += 1
            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = episode_len == max_episode_len
            resets = np.logical_or(
                resets, [info.get("needs_reset", False) for info in infos]
            )

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            for index in range(len(end)):
                if end[index]:
                    episode_returns[episode_indices[index]] = episode_r[index]
                    episode_lengths[episode_indices[index]] = episode_len[index]
                    episode_tier_hitting_count[episode_indices[index]] = infos[index]['tiers_hitting_count']
                    # Give the new episode an a new episode index
                    episode_indices[index] = episode_idx
                    episode_idx += 1

            episode_r[end] = 0
            episode_len[end] = 0

            # find first unfinished episode
            first_unfinished_episode = 0
            while first_unfinished_episode in episode_returns:
                first_unfinished_episode += 1

            # Check for termination conditions
            eval_episode_returns = []
            eval_episode_lens = []
            eval_epsisode_tier_hitting_count = []
            if n_steps is not None:
                total_time = 0
                for index in range(first_unfinished_episode):
                    total_time += episode_lengths[index]
                    # If you will run over allocated steps, quit
                    if total_time > n_steps:
                        break
                    else:
                        eval_episode_returns.append(episode_returns[index])
                        eval_episode_lens.append(episode_lengths[index])
                        eval_epsisode_tier_hitting_count.append(episode_tier_hitting_count[index])
                termination_conditions = total_time >= n_steps
                if not termination_conditions:
                    unfinished_index = np.where(
                        episode_indices == first_unfinished_episode
                    )[0]
                    if total_time + episode_len[unfinished_index] >= n_steps:
                        termination_conditions = True
                        if first_unfinished_episode == 0:
                            eval_episode_returns.append(episode_r[unfinished_index])
                            eval_episode_lens.append(episode_len[unfinished_index])
                            eval_epsisode_tier_hitting_count.append(episode_tier_hitting_count[unfinished_index])

            else:
                termination_conditions = first_unfinished_episode >= n_episodes
                if termination_conditions:
                    # Get the first n completed episodes
                    for index in range(n_episodes):
                        eval_episode_returns.append(episode_returns[index])
                        eval_episode_lens.append(episode_lengths[index])
                        eval_epsisode_tier_hitting_count.append(episode_tier_hitting_count[index])

            if termination_conditions:
                # If this is the last step, make sure the agent observes reset=True
                resets.fill(True)

            # Agent observes the consequences.
            agent.batch_observe(obss, rs, dones, resets)

            if termination_conditions:
                break
            else:
                obss = env.reset(not_end, reset_count=True)

        for i, (epi_len, epi_ret) in enumerate(
            zip(eval_episode_lens, eval_episode_returns)
        ):
            logger.info("evaluation episode %s length: %s R: %s", i, epi_len, epi_ret)

        results = {
            'eval_episode_returns': eval_episode_returns,
            'eval_episode_lens': eval_episode_lens,
            'eval_epsisode_tier_hitting_count': eval_epsisode_tier_hitting_count,
        }
    
    assert agent.training
    return results


def train_agent_batch_with_evaluation(
    agent,
    env,
    eval_env,
    steps,
    outdir,
    eval_n_steps=None,
    eval_n_episodes=None,
    eval_interval=None,
    checkpoint_freq=None,
    log_interval=None,
    max_episode_len=None,
    step_offset=0,
    step_hooks=(),
    return_window_size=100,
    logger=None,
):
    """
    Basically the same as pfrl's method, but we add a custom logger and log additonal stats
    
    Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    Returns:
        List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)
    train_logger = make_train_logger(outdir)
    eval_logger = make_eval_logger(outdir)
    os.makedirs(outdir, exist_ok=True)

    recent_returns = deque(maxlen=return_window_size)
    recent_original_returns = deque(maxlen=return_window_size)

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_original_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")

    time_start = time.perf_counter()

    # o_0, r_0
    obss = env.reset()
    print("Starting training, for {} steps".format(steps))

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    try:
        while True:
            # a_t
            actions = agent.batch_act(obss)
            # o_{t+1}, r_{t+1}
            obss, rs, dones, infos = env.step(actions)
            episode_r += rs
            episode_original_r += np.array([info['original_reward'] for info in infos])
            episode_len += 1

            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = episode_len == max_episode_len
            resets = np.logical_or(
                resets, [info.get("needs_reset", False) for info in infos]
            )
            # Agent observes the consequences
            agent.batch_observe(obss, rs, dones, resets)

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end
            recent_returns.extend(episode_r[end])
            recent_original_returns.extend(episode_original_r[end])

            for _ in range(num_envs):
                t += 1
                if checkpoint_freq and t % checkpoint_freq == 0:
                    save_agent(agent, t, outdir, logger, suffix="_checkpoint")

                for hook in step_hooks:
                    hook(env, agent, t)

            # logging
            if (
                log_interval is not None
                and t >= log_interval
                and t % log_interval < num_envs
            ):
                time_now = time.perf_counter()
                fps = int(t * 4 / (time_now - time_start))
                train_logger.logkv("fps", fps)
                train_logger.logkv('steps', t)
                train_logger.logkv('ep_reward_mean', np.mean(episode_r))
                train_logger.logkv('ep_len_mean', np.mean(episode_len))
                train_logger.logkv('recent_returns', safe_mean(recent_returns))
                train_logger.logkv('ep_original_reward_mean', np.mean(episode_original_r))
                train_logger.logkv('ep_original_returns', safe_mean(recent_original_returns))
                tiers_hitting_count = np.sum([info['tiers_hitting_count'] for info in infos], axis=0)
                for tier, count in enumerate(tiers_hitting_count):
                    train_logger.logkv('tier_{}_hitting_count'.format(tier), count)
                for stats in agent.get_statistics():
                    train_logger.logkv(stats[0], stats[1])
                train_logger.dumpkvs()

            # evaluation
            if (
                eval_interval is not None
                and t >= eval_interval
                and t % eval_interval < num_envs
            ):
                eval_results = batch_run_eval_episodes(
                    env=eval_env,
                    agent=agent,
                    n_steps=eval_n_steps,
                    n_episodes=eval_n_episodes,
                    max_episode_len=max_episode_len,
                    logger=logger,
                )
                # log results 
                for i in range(eval_n_episodes):
                    # log each eval episode's stats on a new line
                    eval_logger.logkv('steps', t)
                    eval_logger.logkv('eval_episode_idx', i)
                    for stat, stat_val in eval_results.items():
                        val = stat_val[i]
                        if 'tier' in stat:
                            for i_tier in range(len(val)):
                                eval_logger.logkv(f'eval_tier_{i_tier}_hitting_count', val[i_tier])
                        else:
                            eval_logger.logkv(stat, val)
                    eval_logger.dumpkvs()

            if t >= steps:
                break

            # Start new episodes if needed
            episode_r[end] = 0
            episode_original_r[end] = 0
            episode_len[end] = 0
            obss = env.reset(not_end)

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        # env.close()
        # if evaluator:
        #     evaluator.env.close()
        train_logger.close()
        eval_logger.close()
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix="_finish")
        train_logger.close()
        eval_logger.close()

    return eval_stats_history


def make_train_logger(log_dir):
    return kvlogger.configure(dir=log_dir, format_strs=['csv', 'stdout'])


def make_eval_logger(log_dir):
    return kvlogger.configure(dir=log_dir, format_strs=['csv'], log_suffix='_eval')


def main():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--env", type=str, default="Breakout")
    parser.add_argument("--steps", type=int, default=2 * 10**7)
    parser.add_argument("--num-tiers", "-t", type=int, default=15,
                        help="Number of tiers to use in the custom reward function")
    parser.add_argument("--original-reward", "-o", action="store_true", default=False,
                        help="Use the original reward function")
    parser.add_argument("--num-envs", type=int, default=32)

    # configs
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )

    # training settings
    parser.add_argument(
        "--agent", type=str, default="DQN", choices=["DQN", "DoubleDQN", "PAL"]
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="nature",
        choices=["nature", "nips", "dueling", "doubledqn"],
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )

    # hyperparams
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--final-exploration-frames", type=int, default=10**6)
    parser.add_argument("--final-epsilon", type=float, default=0.01)
    parser.add_argument("--eval-epsilon", type=float, default=0.001)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument("--replay-start-size", type=int, default=5 * 10**4)
    parser.add_argument("--target-update-interval", type=int, default=3 * 10**4)
    parser.add_argument("--eval-interval", type=int, default=10**5)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--eval-n-runs", type=int, default=10)
    parser.add_argument("--no-clip-delta", dest="clip_delta", action="store_false")
    parser.set_defaults(clip_delta=True)
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--prioritized",
        action="store_true",
        default=False,
        help="Use prioritized experience replay.",
    )
    parser.add_argument("--n-step-return", type=int, default=1)

    args = parser.parse_args()

    # process args
    args.env = args.env + 'NoFrameskip-v4'

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # logging
    import logging
    logging.basicConfig(level=args.log_level)

    # saving dir
    experiment_name = f"{args.env}-{args.num_tiers}-tiers"
    if args.original_reward:
        experiment_name += "-original-reward"
    args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id=experiment_name, make_backup=False)
    print("Output files are saved in {}".format(args.outdir))

    # agent
    sample_env = make_env(args.env, seed=0, num_tiers=args.num_tiers, max_frames=args.max_frames, test=False)
    agent = make_agent(args, n_actions=sample_env.action_space.n)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs, dtype=int) + args.seed * args.num_envs
    assert process_seeds.max() < 2**32

    train_agent_batch_with_evaluation(
        agent=agent,
        env=make_batch_env(args.env, args.num_envs, process_seeds, args.max_frames, num_tiers=args.num_tiers, original_reward=args.original_reward, test=False),
        eval_env=make_batch_env(args.env, args.num_envs, process_seeds, args.max_frames, num_tiers=args.num_tiers, original_reward=True, test=True),
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        outdir=args.outdir,
        log_interval=1000,
    )


if __name__ == "__main__":
    main()

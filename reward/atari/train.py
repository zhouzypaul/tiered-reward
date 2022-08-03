import os
import argparse
import logging
from collections import deque

import numpy as np
from pfrl import experiments
from pfrl import utils
from pfrl.experiments.evaluator import Evaluator, save_agent

from reward.atari.agent import make_agent
from reward.atari.env import make_batch_env, make_env
from reward.atari import logger as kvlogger


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def train_agent_batch(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    log_interval=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
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
    recent_returns = deque(maxlen=return_window_size)
    recent_original_returns = deque(maxlen=return_window_size)

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_original_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")

    # o_0, r_0
    obss = env.reset()
    kvlogger.info("Starting training, for {} steps".format(steps))

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
            if max(rs) > 0:
                print(rs)
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
                kvlogger.logkv('steps', t)
                kvlogger.logkv('ep_reward_mean', np.mean(episode_r))
                kvlogger.logkv('ep_len_mean', np.mean(episode_len))
                kvlogger.logkv('recent_returns', safe_mean(recent_returns))
                kvlogger.logkv('ep_original_reward_mean', np.mean(episode_original_r))
                kvlogger.logkv('ep_original_returns', safe_mean(recent_original_returns))
                tiers_hitting_count = np.sum([info['tiers_hitting_count'] for info in infos], axis=0)
                for tier, count in enumerate(tiers_hitting_count):
                    kvlogger.logkv('tier_{}_hitting_count'.format(tier), count)
                for stats in agent.get_statistics():
                    kvlogger.logkv(stats[0], stats[1])
                kvlogger.dumpkvs()

            if evaluator:
                eval_score = evaluator.evaluate_if_necessary(
                    t=t, episodes=np.sum(episode_idx)
                )
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                    if (
                        successful_score is not None
                        and evaluator.max_score >= successful_score
                    ):
                        break

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
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix="_finish")

    return eval_stats_history


def train_agent_batch_with_evaluation(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    return_window_size=100,
    eval_env=None,
    log_interval=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    logger=None,
):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        env: Environment train the againt against.
        steps (int): Number of total time steps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        checkpoint_freq (int): frequency with which to store networks
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
    Returns:
        agent: Trained agent.
        eval_stats_history: List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent_batch:
            raise ValueError(
                "{} does not support train_agent_batch_with_evaluation().".format(hook)
            )

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        evaluation_hooks=evaluation_hooks,
        save_best_so_far_agent=save_best_so_far_agent,
        use_tensorboard=use_tensorboard,
        logger=logger,
    )

    eval_stats_history = train_agent_batch(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
    )

    return agent, eval_stats_history


def make_logger(log_dir):
    kvlogger.configure(dir=log_dir, format_strs=['csv', 'stdout'])


def main():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--env", type=str, default="Breakout")
    parser.add_argument("--steps", type=int, default=2 * 10**7)
    parser.add_argument("--num_tiers", type=int, default=15,
                        help="Number of tiers to use in the custom reward function")
    parser.add_argument("--original_reward", "-o", action="store_true", default=False,
                        help="Use the original reward function")
    parser.add_argument("--num-envs", type=int, default=8)

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
    make_logger(args.outdir)

    # agent
    sample_env = make_env(args.env, seed=0, max_frames=args.max_frames, test=False)
    agent = make_agent(args, n_actions=sample_env.action_space.n)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs, dtype=int) + args.seed * args.num_envs
    assert process_seeds.max() < 2**32

    train_agent_batch_with_evaluation(
        agent=agent,
        env=make_batch_env(args.env, args.num_envs, process_seeds, args.max_frames, num_tiers=args.num_tiers, original_reward=args.original_reward, test=False),
        eval_env=make_batch_env(args.env, args.num_envs, process_seeds, args.max_frames, num_tiers=args.num_tiers, original_reward=args.original_reward, test=True),
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        outdir=args.outdir,
        save_best_so_far_agent=False,
        log_interval=10000,
    )


if __name__ == "__main__":
    main()

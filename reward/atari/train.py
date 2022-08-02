import argparse

import numpy as np
from pfrl import experiments
from pfrl import utils

from reward.atari.agent import make_agent
from reward.atari.env import make_batch_env, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
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
    parser.add_argument("--final-exploration-frames", type=int, default=10**6)
    parser.add_argument("--final-epsilon", type=float, default=0.01)
    parser.add_argument("--eval-epsilon", type=float, default=0.001)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument(
        "--arch",
        type=str,
        default="doubledqn",
        choices=["nature", "nips", "dueling", "doubledqn"],
    )
    parser.add_argument("--steps", type=int, default=5 * 10**7)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--replay-start-size", type=int, default=5 * 10**4)
    parser.add_argument("--target-update-interval", type=int, default=3 * 10**4)
    parser.add_argument("--eval-interval", type=int, default=10**5)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--eval-n-runs", type=int, default=10)
    parser.add_argument("--no-clip-delta", dest="clip_delta", action="store_false")
    parser.set_defaults(clip_delta=True)
    parser.add_argument(
        "--agent", type=str, default="DQN", choices=["DQN", "DoubleDQN", "PAL"]
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--prioritized",
        action="store_true",
        default=False,
        help="Use prioritized experience replay.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-step-return", type=int, default=1)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs, dtype=int) + args.seed * args.num_envs
    assert process_seeds.max() < 2**32

    args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id=args.env, make_backup=False)
    print("Output files are saved in {}".format(args.outdir))

    sample_env = make_env(args.env, seed=0, max_frames=args.max_frames, test=False)

    agent = make_agent(args, n_actions=sample_env.action_space.n)

    experiments.train_agent_batch_with_evaluation(
        agent=agent,
        env=make_batch_env(args.env, args.num_envs, process_seeds, args.max_frames, test=False),
        eval_env=make_batch_env(args.env, args.num_envs, process_seeds, args.max_frames, test=True),
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        outdir=args.outdir,
        save_best_so_far_agent=False,
        log_interval=1000,
    )


if __name__ == "__main__":
    main()
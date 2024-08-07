import sys
import time
import argparse
import datetime
import tempfile

import torch_ac
import tensorboardX
import wandb

import reward.minigrid.utils as utils
import reward.utils as general_utils
from reward.minigrid.utils import device
from reward.minigrid.agent import MyPPO
from reward.minigrid.model import ImpalaCNN, StateSpaceACModel
from reward.minigrid.envs import environment_builder
from reward.minigrid.minigrid_wrappers import get_num_goal_reaches

# Parse arguments
#global num_goal_reaches
#num_goal_reaches = 0

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--experiment_name", "-e", default=None,
                    help="name of the experiment (default: {ENV}_{ALGO}_{TIME}). Used to name the saving dir.")
parser.add_argument("--debug", action="store_true", default=False,
                    help="Debug mode. Don't log wandb online.")
parser.add_argument("--project", type=str, default="tiered-reward",
                    help="project id for wandb")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--log-interval", type=int, default=10,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16). This is also the number of parallel envs.")
parser.add_argument("--frames", type=float, default=1e7,
                    help="number of frames of training (default: 1e7)")

# Params for environment
parser.add_argument("--reward-function", "-r", default="original",
                    help="What kind of reward function to use for the environment", 
                    choices=['original', 'sparse', 'step_penalty', 'tier', 'tier_based_shaping'])
parser.add_argument("--num-tiers", "-t", type=int, default=5,
                    help="Number of tiers to use in the custom reward function")
parser.add_argument("--normalize-reward", "-n", action="store_true", default=False,
                    help="Normalize the reward function so that its absolute value is in [0, 1]")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor of MDP.")
parser.add_argument("--delta", type=float, default=5,
                    help="offset used in the custom reward function")
parser.add_argument("--max-steps", type=int, default=None, help="change max steps for environment from default value")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    exp_name = args.experiment_name or default_model_name
    if args.reward_function == 'tier':
        sub_dir = f"{args.num_tiers}-tiers"
    else:
        sub_dir = args.reward_function
    model_dir = utils.get_model_dir(exp_name, sub_dir, args.seed)
    general_utils.create_log_dir(model_dir, remove_existing=True)

    # Set wandb
    wandb_output_dir = tempfile.mkdtemp()  # redirect wandb output to a temp dir
    mode = 'online' if not args.debug else 'disabled'
    wandb.init(
        project=args.project,
        sync_tensorboard=True,
        name=exp_name,
        dir=wandb_output_dir,
        config=vars(args),
        mode=mode,
    )

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        env = environment_builder(
            level_name=args.env,
            seed=args.seed + 10000 * i,
            gamma=args.gamma,
            delta=args.delta,
            num_tiers=args.num_tiers,
            use_img_obs=True,
            reward_fn=args.reward_function,
            normalize_reward=args.normalize_reward,
            grayscale=False,
            max_steps=args.max_steps,
            render_mode=None
        )
        envs.append(env)
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    if 'MiniGrid' in args.env:
        obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
        if "vocab" in status:
            preprocess_obss.vocab.load_vocab(status["vocab"])
        txt_logger.info("Observations preprocessor loaded")
    else:
        obs_space = envs[0].observation_space
        preprocess_obss = None

    # Load model
    if 'MiniGrid' in args.env:
        acmodel = ImpalaCNN(obs_space['image'], num_outputs=envs[0].action_space.n, use_memory=args.mem, use_text=args.text)
    elif 'antmaze' in args.env:
        acmodel = StateSpaceACModel(obs_space, envs[0].action_space, args.mem, args.text)
    else:
        raise NotImplementedError('environment not supported')
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = MyPPO(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")
    
    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    already_written_header = False

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
        
        
        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            original_return_per_episode = utils.synthesize(logs["original_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["original_return_" + key for key in original_return_per_episode.keys()]
            data += original_return_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.5f} {:.5f} {:.5f} {:.5f} | oR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()
            
            #txt_logger.info('Goal Reaches: {}'.format(get_num_goal_reaches()))
            if not already_written_header:
                csv_logger.writerow(header)
                already_written_header = True
            csv_logger.writerow(data)
            csv_file.flush()
            
            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)
            
            
        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")


    #print('Goal Reaches: ', get_num_goal_reaches())

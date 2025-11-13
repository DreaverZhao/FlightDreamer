"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with DreamerV3.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import signal
import sys

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import flipper.tasks  # noqa: F401

import attridict
from colorama import Fore, Style

from dreamer import Dreamer
from utils import seedEverything
from envs import getVecEnvProperties, IsaacGymWrapper

# config shortcuts
agent_cfg_entry_point = "dreamer_cfg_entry_point"

# Global flag for graceful shutdown
interrupt_received = False


def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully."""
    global interrupt_received
    if not interrupt_received:
        print(Fore.YELLOW + "\n[FlightDreamer] Keyboard interrupt received. Stopping evaluation gracefully..." + Style.RESET_ALL)
        print(Fore.YELLOW + "[FlightDreamer] Press Ctrl+C again to force quit." + Style.RESET_ALL)
        interrupt_received = True
    else:
        print(Fore.RED + "\n[FlightDreamer] Force quit requested. Exiting immediately..." + Style.RESET_ALL)
        sys.exit(1)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with dreamer."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # find the newest checkpoint
    if args_cli.checkpoint is None:
        checkpoints_folder = "./logs/dreamer/checkpoints"
        if not os.path.exists(checkpoints_folder):
            raise FileNotFoundError(f"Log folder {checkpoints_folder} does not exist. Please run the training first.")
        # find the latest checkpoint in the log folder
        checkpoints = [f for f in os.listdir(checkpoints_folder) if f.startswith(args_cli.task) and f.endswith(".pth")]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints for task {args_cli.task} found in {checkpoints_folder}. Please run the training first.")
        checkpoints.sort(key=lambda t: os.path.getmtime(os.path.join(checkpoints_folder, t)), reverse=True)
        args_cli.checkpoint = os.path.join(checkpoints_folder, checkpoints[0])
    checkpointToLoad = args_cli.checkpoint

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # load config for dreamer
    config = attridict(agent_cfg)
    seedEverything(config.seed)

    # create isaac environment
    env = IsaacGymWrapper(gym.make(args_cli.task, cfg=env_cfg))

    observationShape, actionSize, actionLow, actionHigh = getVecEnvProperties(env)
    print(Fore.GREEN + f"[FlightDreamer] envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}" + Style.RESET_ALL)
    print(Fore.GREEN + f"[FlightDreamer] Using device: {env_cfg.sim.device}" + Style.RESET_ALL)
    dreamer = Dreamer(env_cfg.scene.num_envs, observationShape, actionSize, actionLow, actionHigh, env_cfg.sim.device, config.dreamer)

    print(Fore.BLUE + f"[FlightDreamer] Loading checkpoint: {checkpointToLoad}" + Style.RESET_ALL)
    print(Fore.LIGHTCYAN_EX + "[FlightDreamer] By default, the latest checkpoint is loaded from the checkpoints folder")
    print("[FlightDreamer] You can change the checkpoint by modifying the config file or specify one in the command line arguments by --checkpoint." + Style.RESET_ALL)
    dreamer.loadCheckpoint(checkpointToLoad)

    print(Fore.BLUE + "[FlightDreamer] Start evaluating, press Ctrl+C to stop..." + Style.RESET_ALL)
    # register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    while not interrupt_received:
        dreamer.environmentInteraction(env, 1, seed=config.seed, evaluation=True)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

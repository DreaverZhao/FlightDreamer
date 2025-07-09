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
import random
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import flipper.tasks  # noqa: F401

import torch
import attridict
from tqdm import tqdm

from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import getVecEnvProperties, IsaacGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders

# config shortcuts
agent_cfg_entry_point = "dreamer_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with dreamer."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # load config for dreamer
    config = attridict(agent_cfg)
    seedEverything(config.seed)

    runName                 = f"{args_cli.task}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    
    # create isaac environment
    env = IsaacGymWrapper(gym.make(args_cli.task, cfg=env_cfg))

    observationShape, actionSize, actionLow, actionHigh = getVecEnvProperties(env)
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}")
    print(f"Using device: {env_cfg.sim.device}")
    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, env_cfg.sim.device, config.dreamer)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    print("Warm starting...")
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio
    print(f"Training for {iterationsNum} iterations, {config.replayRatio} replay iterations per iteration, total gradient steps: {config.gradientSteps}")
    pbar = tqdm(total=iterationsNum, unit="iteration")
    for _ in range(iterationsNum):
        pbar.update(1)
        for r in range(config.replayRatio):
            sampledData                         = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics    = dreamer.worldModelTraining(sampledData)
            behaviorMetrics                     = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
            pbar.set_description(f"Replay iteration {r + 1}/{config.replayRatio}")
        pbar.set_description("Collecting data...")
        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        if config.saveMetrics:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{args_cli.task}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

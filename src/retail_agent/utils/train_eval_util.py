import numpy as np
from tqdm import tqdm
import tensorflow as tf
from typing import List, Union
from tf_agents.trajectories import Trajectory, TimeStep
from tf_agents.policies.utils import PolicyInfo
import logging
import hydra
from pathlib import Path
import os

from retail_agent.agents import get_retail_agent
from retail_agent.utils.trajectory_util import (
    get_env,
    get_random_policy,
    get_one_trajectory,
    get_fixed_arm_policy,
)
from retail_agent.utils.storage_util import StorageUtil, CloudStorageUtil, LocalStorageUtil

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_last_time_step(trajectory: Trajectory):
    """Extract the last time step of a trajectory.

    Parameters
    ----------
        trajectory (Trajectory): Trajectory object.

    Returns
    -------
        TimeStep: Last time step of the trajectory.
    """
    last_time_step = tf.nest.map_structure(lambda x: x[:, -1], trajectory)
    last_time_step = TimeStep(
        step_type=last_time_step.step_type,
        reward=last_time_step.reward,
        discount=last_time_step.discount,
        observation=last_time_step.observation,
    )
    return last_time_step


def collect_offline_data(config: dict) -> tuple:
    """Collect offline data from the environment.

    Parameters
    ----------
        config (dict): Configuration dictionary, containing the config for the synthesizer and how many simulations to run for the offline data

    Returns
    -------
        List[Trajectory]: a list of trajectories
        TFEnvironment: the environment used to collect the offline data
        TimeStep: the last time step of the last trajectory
    """
    # collect offline data
    n_offline_data_step = config["train"]["n_offline_data_step"]
    random_seed_range = config["train"]["random_seed_range"]
    if config["train"]["offline_data_policy"]["type"] == "fixed_arm":
        if config["train"]["offline_data_policy"]["fixed_seed"]:
            seed = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", np.random.randint(0, random_seed_range)))
        else:
            seed = np.random.randint(0, random_seed_range)
        config["synthetic_data"]["random_seed"] = seed
    elif config["train"]["offline_data_policy"]["type"] == "random":
        config["synthetic_data"]["random_seed"] = np.random.randint(0, random_seed_range)
    # initialize a new environment with the same prior distributions for all parameters
    env = get_env(config)
    if config["train"]["offline_data_policy"]["type"] == "random":
        policy = get_random_policy(env)
    elif config["train"]["offline_data_policy"]["type"] == "fixed_arm":
        policy = get_fixed_arm_policy(env, config["train"]["offline_data_policy"]["arm"])
    trajectory = get_one_trajectory(env, policy, n_offline_data_step)

    last_time_step = get_last_time_step(trajectory)
    return trajectory, env, last_time_step


def initialize_agent(cfg):
    """Create necessary inputs for agent and initialize agent objects.

    Parameters
    ----------
        agent_type (str): Type of the agent.
        agent_config (dict): Configuration dictionary for the agent.
        time_step_spec (tf.TensorSpec): Time step specification.
        action_spec (tf.TensorSpec): Action specification.

    Returns
    -------
        List[tf_agent.TFAgent]: List of agents (TFAgent objects).
    """
    agent_type = cfg["agent"]["agent_type"]
    env = get_env(cfg)
    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()
    agent_params = cfg["agent"].get("agent_params", {})

    agent = get_retail_agent(
        agent_type,
        time_step_spec,
        action_spec,
        agent_params,
        product_category_mapping=env.synthesizer.product_category_mapping._value,
    )

    return agent


def prepare_trajectory(trajectories: List[Trajectory]) -> Trajectory:
    """Concatenate list of trajectories and prepare them ready for training.

    Read M trajectories for N customers, and concatenate them into a single trajectory of M*N customers.

    Parameters
    ----------
        trajectories (List[Trajectory]): List of trajectories.

    Returns
    -------
        Trajectory: Concatenated trajectory.
    """
    observation = {feature_name: tf.concat([traj.observation[feature_name] for traj in trajectories], axis=0) for feature_name in trajectories[0].observation.keys()}
    return Trajectory(
        observation=observation,
        reward=tf.concat([traj.reward for traj in trajectories], axis=0),
        action=tf.concat([traj.action for traj in trajectories], axis=0),
        step_type=tf.concat([traj.step_type for traj in trajectories], axis=0),
        next_step_type=tf.concat([traj.next_step_type for traj in trajectories], axis=0),
        policy_info=PolicyInfo(),
        discount=tf.concat([traj.discount for traj in trajectories], axis=0),
    )


def prepare_trajectory_for_ppo_agent(trajectories: List[Trajectory], n_arm: int = 11) -> Trajectory:
    """Concatenate list of trajectories and prepare them ready for training.

    Read M trajectories for N customers, and concatenate them into a single trajectory of M*N customers.

    Parameters
    ----------
        trajectories (List[Trajectory]): List of trajectories.

    Returns
    -------
        Trajectory: Concatenated trajectory.
    """
    observation = {feature_name: tf.concat([traj.observation[feature_name] for traj in trajectories], axis=0) for feature_name in trajectories[0].observation.keys()}
    logits_shape = next(iter(observation.values())).shape[:2] + [n_arm]
    policy_info = {"dist_params": {"logits": tf.zeros(logits_shape)}}
    return Trajectory(
        observation=observation,
        reward=tf.concat([traj.reward for traj in trajectories], axis=0),
        action=tf.concat([traj.action for traj in trajectories], axis=0),
        step_type=tf.concat([traj.step_type for traj in trajectories], axis=0),
        next_step_type=tf.concat([traj.next_step_type for traj in trajectories], axis=0),
        policy_info=policy_info,
        discount=tf.concat([traj.discount for traj in trajectories], axis=0),
    )


def train_agent(cfg, output_dir: Union[str, Path] = None):
    """Train agent using offline data trajectories stored in the output directory."""
    storage: StorageUtil
    if cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()
    elif cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])

    logging.info("Loading offline data.")
    if output_dir is None:
        output_dir = str(cfg["experiment"]["output_dir"])
    else:
        output_dir = str(output_dir)
    offline_data_files = storage.list_dir(output_dir + "/offline_data/")
    train_data_size = cfg["train"]["train_data_size"]
    offline_data_files = [file for file in offline_data_files[:train_data_size] if "pkl" in file]

    logging.info(f"Initialize {cfg['agent']['agent_type']} agent.")
    agent = initialize_agent(cfg)
    train_loss = []
    train_params = cfg["agent"]["train_params"]
    batch_size = min(len(offline_data_files), train_params["batch_size"])
    callback = hydra.utils.instantiate(cfg["train"]["callback"])
    callback.model = agent
    callback.model.stop_training = False

    logging.info(f"Train the {cfg['agent']['agent_type']} agent.")
    callback.on_train_begin()
    for i in tqdm(range(train_params["n_train_epoch"]), desc="Training epochs", position=-1):
        if callback.model.stop_training:
            logging.info("Stop training triggered by early stopping callback.")
            break
        epoch_ids = np.random.choice(len(offline_data_files), batch_size, replace=False)

        trajectory = []
        for epoch_id in tqdm(epoch_ids):
            trajectory.append(storage.load_obj(offline_data_files[epoch_id]))
        if cfg["agent"]["agent_type"] == "ppo":
            trajectory = prepare_trajectory_for_ppo_agent(trajectory, cfg["pyenv"]["n_coupon_levels"])
        else:
            trajectory = prepare_trajectory(trajectory)

        callback.on_epoch_begin(i)
        loss = agent.train(trajectory)
        train_loss.append(float(loss.loss))
        print(f"epoch {i}, loss {loss.loss}")
        callback.on_epoch_end(i, {"loss": float(loss.loss)})

    callback.on_train_end()

    return agent, train_loss


def collect_eval_trajectory(cfg, eval_env_index: int = 0, agent_path: str = None):
    """Collect evaluation trajectories using trained policy stored in the agent output directory."""
    output_dir = cfg["experiment"]["output_dir"]
    if cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()
        # For multi-run sweep jobs, store the offline data in the experiment directory.
        if "eval" in str(Path.cwd()):
            output_dir = Path.cwd().parent.parent / output_dir
    elif cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])  # type: ignore

    logging.info("Loading trained policy")
    agent = initialize_agent(cfg)
    agent_path = agent_path or str(Path(output_dir, "agent", cfg["agent"]["agent_type"], f"{agent.name}_optimal", str(cfg["experiment"]["run_id"])))
    agent = storage.load_policy(agent, str(Path(agent_path, "checkpoint")))

    logging.info("Loading eval environment")
    synthesizer = storage.list_dir(Path(output_dir, "synthesizer"))
    last_time_step = storage.list_dir(Path(output_dir, "last_time_step"))
    if eval_env_index >= len(synthesizer):
        return f"eval_env_index {eval_env_index} is out of range. The number of synthesizers is {len(synthesizer)}"
    eval_synthesizer = storage.load_obj(synthesizer[eval_env_index])
    eval_env = get_env(cfg, eval_synthesizer)

    logging.info("Collect evaluation trajectories")
    eval_time_step = storage.load_obj(last_time_step[eval_env_index])
    eval_traj = get_one_trajectory(eval_env, agent.policy, cfg["eval"]["n_eval_step"], time_step=eval_time_step)

    logging.info("Store evaluation trajectories")
    id = synthesizer[eval_env_index].split("/")[-1].split(".")[0]
    storage.store_obj(eval_traj, str(Path(agent_path, "eval_result", f"{id}.pkl")))

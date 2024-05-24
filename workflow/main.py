import hydra
from omegaconf import OmegaConf
import logging
import uuid
from pathlib import Path
from botocore.exceptions import ClientError
import os
import numpy as np
import re
from collections import defaultdict
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from tf_agents.trajectories import Trajectory

from retail_agent.utils.storage_util import CloudStorageUtil, LocalStorageUtil
from retail_agent.utils.train_eval_util import collect_offline_data
from retail_agent.utils.train_eval_util import train_agent
from retail_agent.utils.train_eval_util import collect_eval_trajectory
from retail_agent.utils.trajectory_util import get_env
from retail_agent.utils.metric_util import generate_evaluation_report


def collect_train_data(cfg) -> None:
    # wrapper function to collect one offline data trajectory.
    logging.info(f"Collect offline data.")
    offline_data, env, last_time_step = collect_offline_data(cfg)

    logging.info(f"Offline data collected. Saving to {cfg['experiment']['storage']} storage.")
    unique_id = str(uuid.uuid4())
    if cfg["train"]["offline_data_policy"]["type"] == "random":
        offline_type = "random"
    elif cfg["train"]["offline_data_policy"]["type"] == "fixed_arm":
        offline_type = f'fixed_arm_{cfg["train"]["offline_data_policy"]["arm"]}'

    output_dir = Path(cfg["experiment"]["output_dir"])
    if cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])
    elif cfg["experiment"]["storage"] == "local":
        # For multi-run sweep jobs, store the offline data in the experiment directory.
        if "collect_train_data" in str(Path.cwd()):
            output_dir = Path.cwd().parent.parent / output_dir
        storage = LocalStorageUtil()

    storage.store_obj(offline_data, output_dir / f"offline_data/{offline_type}/{unique_id}.pkl")
    storage.store_obj(env.synthesizer, output_dir / f"synthesizer/{unique_id}.pkl")
    storage.store_obj(last_time_step, output_dir / f"last_time_step/{unique_id}.pkl")
    storage.store_obj(cfg, output_dir / "cfg.pkl")


def format_trajectory_to_df(traj: Trajectory, seg_info: dict):
    """Format the trajectory to a pandas data frame.

    Parameters
    ----------
        traj (Trajectory): a trajectory object.
        seg_info (dict): a dictionary that contains the segment information.

    Returns
    -------
        pd.DataFrame: a pandas data frame that contains the trajectory information.
    """
    df = pd.DataFrame()
    for feature in traj.observation.keys():
        if feature in ["product_price", "previous_transaction", "observed_customer_product_feature"]:
            df[f"avg_{feature}"] = traj.observation[feature].numpy().mean(axis=-1).flatten()
        else:
            df[feature] = traj.observation[feature].numpy().flatten()
    df["reward"] = traj.reward.numpy().flatten()
    df["action"] = traj.action.numpy().flatten()
    n_customer, n_time_step = traj.reward.numpy().shape
    segment = np.zeros(n_customer)
    for seg_id, idx in seg_info.items():
        segment[idx] = seg_id
    df["segment"] = np.tile(segment[:, np.newaxis], n_time_step).flatten()
    df["timestep"] = np.tile(np.arange(n_time_step)[np.newaxis, :], n_customer).flatten()

    return df


def feature_selection(cfg) -> None:
    # wrapper function to convert offline data trajectories to a pandas dataframe.
    logging.info(f"Load offline training data and store to a pandas data frame.")

    if cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])
    elif cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()

    offline_data_files = storage.list_dir(cfg["experiment"]["output_dir"] + "/offline_data/random")
    offline_data_files = [file for file in offline_data_files if ".pkl" in file]
    logging.info(f"Loading offline data files: {offline_data_files}")
    sample_size = min(cfg["summary"]["feature_selection_sample_size"], len(offline_data_files))
    offline_data_files = np.random.choice(offline_data_files, sample_size, replace=False)
    all_segment_info = storage.load_obj(cfg["experiment"]["output_dir"] + "/customer_seg_by_price_sensitivity.pkl")

    dfs = []
    for file in tqdm(offline_data_files, desc="loading offline data"):
        traj = storage.load_obj(file)
        file_name = file.split("/")[-1].split(".")[0]
        seg_info = all_segment_info[file_name]
        df = format_trajectory_to_df(traj, seg_info)
        df["env_id"] = file_name
        dfs.append(df)

    dfs = pd.concat(dfs)

    storage.store_obj(dfs, cfg["experiment"]["output_dir"] + "/summary/feature_selection_data.pkl")


def train(cfg) -> None:
    # wrapper function to train one agent once.
    logging.info(f"Train one agent.")
    output_dir = cfg["experiment"]["output_dir"]

    if cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])
    elif cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()
        # For multi-run sweep jobs, store the offline data in the experiment directory.
        if "train" in str(Path.cwd()):
            output_dir = Path.cwd().parent.parent / output_dir

    if cfg["train"]["read_params_from_tuning_log"]:
        try:
            agent_type = cfg["agent"]["agent_type"]
            tuning_log = storage.load_obj(cfg["experiment"]["output_dir"] + f"/agent/{agent_type}/tuning_job/tuning_log.pkl")
            best_params = tuning_log.iloc[tuning_log["accumulated_revenue"].idxmax()].agent_params
            cfg["agent"]["agent_params"] = best_params
            logging.info("Override agent parameters with the optimal one in the tuning log.")
        except ClientError:
            logging.info("Tuning log not found in the cloud directory.")
            pass
        except FileNotFoundError:
            logging.info("Tuning log not found in the local directory.")
            pass

    agent, train_loss = train_agent(cfg, output_dir)

    logging.info(f"Training completed. Saving the training loss and evaluation trajectories to {cfg['experiment']['storage']} path.")
    id_directory = Path(output_dir, "agent", cfg["agent"]["agent_type"], f"{agent.name}_optimal", str(cfg["experiment"]["run_id"]))
    storage.store_obj(train_loss, str(Path(id_directory, "train_loss.pkl")))
    storage.store_obj(cfg["agent"], str(Path(id_directory, "agent_cfg.pkl")))
    storage.store_policy(agent, str(Path(id_directory, "checkpoint")))


def eval(cfg) -> None:
    # wrapper function to collect one evaluation trajectory.
    traj_id = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", cfg["experiment"]["instance_eval_id"]))

    collect_eval_trajectory(cfg, traj_id)


def segment_customer(cfg) -> None:
    # wrapper function to segment customer equally to two segments by price sensitivities.
    logging.info("Segment customers")

    n_segment = cfg["summary"]["n_customer_segment"]
    dir = cfg["experiment"]["output_dir"]

    if cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])
    elif cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()

    # read the price sensitivity coefficient from all environments
    synthesizers = storage.list_dir(str(Path(dir, "synthesizer")))
    assert len(synthesizers) > 0, ValueError("No synthesizer found in the directory.")
    coefs = {}
    for syn_name in synthesizers:
        syn = storage.load_obj(syn_name)
        syn_name = syn_name.split("/")[-1].split(".")[0]
        coefs[syn_name] = syn.utility_beta_ui_w.mean(axis=-1)

    # Compute the sensitivity threshold to segment customers
    all_coef = np.array(list(coefs.values())).flatten()
    threshold = np.quantile(all_coef, np.linspace(0, 1, n_segment + 1)).tolist()
    threshold[0] = -np.inf
    threshold[-1] = np.inf

    seg_idxs = {}
    for syn_name, coef in coefs.items():
        seg_idx = {}
        for i in range(n_segment):
            seg_idx[i] = np.where((coef >= threshold[i]) & (coef < threshold[i + 1]))[0].tolist()
        seg_idxs[syn_name] = seg_idx

    storage.store_obj(seg_idxs, str(Path(dir, "customer_seg_by_price_sensitivity.pkl")))


def compute_metrics(cfg) -> None:
    # wrapper function to compute metrics for all customers.
    logging.info("Evaluate multiple agents")

    if cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])
    elif cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()

    logging.info(f"Load evaluation trajectories from {cfg['experiment']['storage']} directory.")
    eval_results = defaultdict(list)
    env_ids = defaultdict(list)
    for file in storage.list_dir(str(Path(cfg["experiment"]["output_dir"], "agent"))):
        if ("eval_result" in file) and ("tuning_job" not in file):
            agent_name = re.sub("_optimal", "", file.split("/")[3])
            if agent_name in cfg["summary"]["agents"]:
                eval_result = storage.load_obj(file)
                eval_results[agent_name].append(eval_result)
                env_id = file.split("/")[-1].split(".")[0]
                env_ids[agent_name].append(env_id)

    logging.info("Analyze the evaluation trajectories.")
    env = get_env(cfg)
    window_length = cfg["summary"]["customer_retention_window_length"]

    summary_dfs = generate_evaluation_report(eval_results, window_length=window_length, product_category_mapping=env.synthesizer.product_category_mapping, n_arm=cfg["pyenv"]["n_coupon_levels"], coupon_unit=cfg["pyenv"]["coupon_unit"], env_ids=env_ids)

    logging.info(f"Save result figures to a {cfg['experiment']['storage']} directory")

    dir = Path(cfg["experiment"]["output_dir"], "summary")
    dir.mkdir(parents=True, exist_ok=True)
    storage.store_obj(summary_dfs, str(Path(dir, f"{cfg['summary']['summary_file_prefix']}all_customer_dfs.pkl")))


def compute_metrics_by_segment(cfg) -> None:
    # wrapper function to compute metrics for each customer segment.
    logging.info("Evaluate multiple agents")

    n_customer_segment = cfg["summary"]["n_customer_segment"]

    if cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])
    elif cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()

    logging.info(f"Load evaluation trajectories from {cfg['experiment']['storage']} directory.")
    seg_idxs = storage.load_obj(str(Path(cfg["experiment"]["output_dir"], "customer_seg_by_price_sensitivity.pkl")))
    eval_results = {seg_id: defaultdict(list) for seg_id in range(n_customer_segment)}
    env_ids = defaultdict(list)
    files_under_dir = storage.list_dir(str(Path(cfg["experiment"]["output_dir"], "agent")))
    for file in files_under_dir:
        if ("eval_result" in file) and ("tuning_job" not in file):
            agent_name = re.sub("_optimal", "", file.split("/")[3])
            if agent_name in cfg["summary"]["agents"]:
                traj = storage.load_obj(file)
                synthesizer_id = file.split("/")[-1].split(".")[0]
                seg_idx = seg_idxs[synthesizer_id]
                for seg_id in seg_idx.keys():
                    eval_results[seg_id][agent_name].append(tf.nest.map_structure(lambda x: tf.gather(x, seg_idx[seg_id]), traj))
                env_id = file.split("/")[-1].split(".")[0]
                env_ids[agent_name].append(env_id)

    logging.info("Analyze the evaluation trajectories.")
    env = get_env(cfg)
    window_length = cfg["summary"]["customer_retention_window_length"]

    segmented_summary_dfs = defaultdict(list)
    for seg_name, eval_result in eval_results.items():
        summary_df = generate_evaluation_report(eval_result, window_length=window_length, product_category_mapping=env.synthesizer.product_category_mapping, n_arm=cfg["pyenv"]["n_coupon_levels"], coupon_unit=cfg["pyenv"]["coupon_unit"], env_ids=env_ids)
        for df_name, df in summary_df.items():
            df["segment"] = seg_name
            segmented_summary_dfs[df_name].append(df)
    segmented_summary_dfs = {df_name: pd.concat(dfs) for df_name, dfs in segmented_summary_dfs.items()}

    logging.info(f"Save result figures to a {cfg['experiment']['storage']} directory")

    dir = Path(cfg["experiment"]["output_dir"], "summary")
    dir.mkdir(parents=True, exist_ok=True)
    storage.store_obj(segmented_summary_dfs, str(Path(dir, f"{cfg['summary']['summary_file_prefix']}segmented_dfs.pkl")))


@hydra.main(version_base=None, config_path="cfg", config_name="agent_training")
def main(cfg):
    cfg = OmegaConf.to_object(cfg)
    step = cfg["step"]

    logging.info(f"Running step: {step}")
    if step == "collect_train_data":
        collect_train_data(cfg)
    elif step == "feature_selection":
        feature_selection(cfg)
    elif step == "train":
        train(cfg)
    elif step == "eval":
        eval(cfg)
    elif step == "segment_customer":
        segment_customer(cfg)
    elif step == "compute_metrics":
        compute_metrics(cfg)
        compute_metrics_by_segment(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()

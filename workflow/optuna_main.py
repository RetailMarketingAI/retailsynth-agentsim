# This script takes charges to submit jobs to the AWS batch and track the hyperparameter tuning process.
import optuna
from optuna.distributions import UniformDistribution, CategoricalDistribution, IntUniformDistribution
import hydra
import time
import pandas as pd
from pathlib import Path
from collections import deque
import logging

from retail_agent.utils.storage_util import CloudStorageUtil, create_client
from retail_agent.utils.metric_util import compute_accumulated_revenue, compute_final_training_loss


def create_study(cfg):
    # create an optuna study to track the hyperparameter tuning
    study = optuna.create_study(
        directions=cfg["study"]["directions"],
        sampler=hydra.utils.instantiate(cfg["study"]["sampler"]),
    )
    return study


def get_nested_value(data, keys):
    """Retrieve the value from nested dictionaries using a list of keys."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data


def add_new_trials(study, df: pd.DataFrame, cfg: dict):
    """Read the tuning log and add the completed trials to the study.

    Parameters
    ----------
        study (optuna.study.Study): optuna study object
        df (pd.DataFrame): tuning log
        cfg (dict): configuration dictionary for the aws batch job
    """
    n_completed_trails = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    for i, row in df[n_completed_trails:].iterrows():
        param_values = {param: get_nested_value(row["agent_params"], param.split(".")) for param in cfg["params"].keys()}
        param_distributions = {}
        for param, param_setup in cfg["params"].items():
            if param_setup["type"] == "float":
                param_distributions[param] = UniformDistribution(
                    low=param_setup["low"],
                    high=param_setup["high"],
                )
            elif param_setup["type"] == "categorical":
                param_distributions[param] = CategoricalDistribution(choices=param_setup["value"])
            elif param_setup["type"] == "int":
                param_distributions[param] = IntUniformDistribution(
                    low=param_setup["low"],
                    high=param_setup["high"],
                )
        try:
            trial = optuna.trial.create_trial(
                params=param_values,
                distributions=param_distributions,
                value=row["accumulated_revenue"],
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(trial)
        except ValueError:
            print("OHhh no")
            pass


def read_tuning_log(storage, cfg: dict, n_past_trials: int = 0):
    """Read the tuning log from the cloud storage.

    Parameters
    ----------
        storage: cloud storage util object
        cfg (dict): configuration dictionary for the aws client connection
        n_past_trials (int, optional): number of trials to ignore. Defaults to 0.

    Returns
    -------
        pd.DataFrame: tuning log
    """
    try:
        tuning_log = storage.load_obj(cfg["experiment"]["output_dir"] + "/tuning_log.pkl")
        tuning_log = tuning_log.iloc[n_past_trials:]
    except:
        tuning_log = pd.DataFrame(columns=["agent_params", "accumulated_revenue", "final_training_loss"])

    return tuning_log


def check_job_status(batch_client, current_job):
    """Check the status of the current job.

    Parameters
    ----------
        batch_client: boto3.client object connected to batch
        current_job (str): current job id

    Returns
    -------
        str: job status, like RUNNING, SUCCEEDED, FAILED, etc.
    """
    response = batch_client.describe_jobs(jobs=[current_job])
    job_status = response["jobs"][0]["status"]
    return job_status


def log_job_performance(config, storage, job_id):
    """Log the performance of the current job.

    Parameters
    ----------
        config (dict): configuration dictionary for the aws batch job
        job_id (str): job id
    """
    eval_result_path = str(Path(config["experiment"]["output_dir"], job_id))
    eval_result_files = storage.list_dir(eval_result_path)
    eval_result = []
    training_loss = []
    for file in eval_result_files:
        if "eval_result" in file:
            eval_result.append(storage.load_obj(file))
        elif "training_loss" in file:
            training_loss.append(storage.load_obj(file))
    accumulated_revenue = compute_accumulated_revenue(eval_result)["accumulated_revenue"]
    accumulated_revenue_mean = accumulated_revenue.mean()
    training_loss = compute_final_training_loss(training_loss)["final_training_loss"]
    agent_cfg = storage.load_obj(str(Path(eval_result_path, "agent_cfg.pkl")))

    try:
        tuning_log = storage.load_obj(str(Path(config["experiment"]["output_dir"], "tuning_log.pkl")))
    except:
        tuning_log = pd.DataFrame(columns=["agent_params", "accumulated_revenue", "training_loss"])
    tuning_log = tuning_log.append(
        {
            "agent_params": agent_cfg,
            "accumulated_revenue": accumulated_revenue_mean,
            "training_loss": training_loss,
        },
        ignore_index=True,
    )
    storage.store_obj(tuning_log, str(Path(config["experiment"]["output_dir"], "tuning_log.pkl")))


def suggest_new_params(study, config):
    """Suggest new hyperparameters for the next trial.

    Parameters
    ----------
        study (optuna.study.Study): optuna study object
        config (dict): configuration dictionary for the aws batch job

    Returns
    -------
        dict: suggested hyperparameters
    """
    trial = study.ask()
    params = {}
    for param_name, param in config["params"].items():
        if param["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param["value"])
        elif param["type"] == "float":
            params[param_name] = trial.suggest_float(param_name, param["low"], param["high"])
        elif param["type"] == "int":
            params[param_name] = trial.suggest_int(param_name, param["low"], param["high"])
    return params


def submit_job(batch_client, params, config, run_id):
    """Submit a new job to the AWS batch.

    Parameters
    ----------
        batch_client: boto3.client object connected to batch
        params (dict): hyperparameters for the current trial
        config (dict): configuration dictionary for the aws batch job
        run_id (int): current run id

    Returns
    -------
        str: new job id
    """
    config_path = config["batch_job_train"]["config_path"] if "config_path" in config["batch_job_train"] else "cfg"
    format_params = [f"agent.agent_params.{param}={value}" for param, value in params.items()]
    command = list(config["batch_job_train"]["command"]) + [f"agent={config['agent_type']}"] + format_params
    job_ids = []
    for instance_id in range(config["n_instance_per_trial"]):
        train_job_id = batch_client.submit_job(
            jobName=config["batch_job_train"]["job_name"],
            jobQueue=config["batch_job_train"]["job_queue"],
            jobDefinition=config["batch_job_train"]["job_definition"],
            containerOverrides={
                "command": command + [f"experiment.run_id={run_id}", f"experiment.instance_id={instance_id}", f"--config-path={config_path}"],
                "resourceRequirements": [
                    {"type": "VCPU", "value": str(config["batch_job_train"]["vcpus"])},
                    {"type": "MEMORY", "value": str(config["batch_job_train"]["memory"])},
                ],
            },
            timeout={"attemptDurationSeconds": config["batch_job_train"]["timeout"]},
        )["jobId"]

        command = list(config["batch_job_eval"]["command"]) + [f"agent={config['agent_type']}"] + format_params
        eval_job_id = batch_client.submit_job(
            jobName=config["batch_job_eval"]["job_name"],
            jobQueue=config["batch_job_eval"]["job_queue"],
            jobDefinition=config["batch_job_eval"]["job_definition"],
            dependsOn=[
                {"jobId": train_job_id},
            ],
            arrayProperties={"size": config["n_eval_trajectories"]},
            containerOverrides={
                "command": command + [f"experiment.run_id={run_id}", f"experiment.instance_id={instance_id}", f"--config-path={config_path}"],
                "resourceRequirements": [
                    {"type": "VCPU", "value": str(config["batch_job_eval"]["vcpus"])},
                    {"type": "MEMORY", "value": str(config["batch_job_eval"]["memory"])},
                ],
            },
            timeout={"attemptDurationSeconds": config["batch_job_eval"]["timeout"]},
        )["jobId"]

        job_ids.append(eval_job_id)

    return job_ids


@hydra.main(version_base=None, config_path="cfg", config_name="optuna_tuning")
def optuna_tuning_experiment(cfg):
    logging.info("Run optuna tuning experiment on cloud.")
    assert cfg["experiment"]["storage"] == "cloud", "This is a script to run hyperparameter tuning on cloud only."

    study = create_study(cfg["optuna"])
    trial_count = 0
    current_running_jobs = deque(maxlen=cfg["optuna"]["n_job_in_parallel"] * cfg["optuna"]["n_instance_per_trial"])
    current_run_ids = deque(maxlen=cfg["optuna"]["n_job_in_parallel"])
    batch_client = create_client(cfg["cloud"], service="batch")
    storage = CloudStorageUtil(cfg["cloud"])

    if not cfg["optuna"]["consider_past_trials"]:
        tuning_log = read_tuning_log(storage, cfg)
        n_past_trials = len(tuning_log)
    else:
        n_past_trials = 0

    logging.info(f"Start hyperparameter tuning for {cfg['optuna']['agent_type']} agent.")
    while True:
        job_status = [check_job_status(batch_client, job_id) for job_id in current_running_jobs]

        if all([status == "SUCCEEDED" for status in job_status]) and (trial_count >= cfg["optuna"]["n_trial"]):
            break

        elif all([status == "SUCCEEDED" for status in job_status]):
            for run_id in current_run_ids:
                log_job_performance(cfg, storage, str(run_id))
            tuning_log = read_tuning_log(storage, cfg, n_past_trials=n_past_trials)
            add_new_trials(study, tuning_log, cfg["optuna"])

            for _ in range(cfg["optuna"]["n_job_in_parallel"]):
                new_params = suggest_new_params(study, cfg["optuna"])
                job_ids = submit_job(batch_client, new_params, cfg["optuna"], trial_count)
                for job_id in job_ids:
                    logging.info(f"{trial_count}. Job {job_id} submitted.")
                current_running_jobs.extend(job_ids)
                current_run_ids.append(trial_count)
                trial_count += 1

        elif "FAILED" in job_status:
            raise ValueError("Job failed")

        logging.info(f"Sleep for {cfg['optuna']['sleep_time']} seconds.")
        time.sleep(cfg["optuna"]["sleep_time"])

    logging.info(f"Hyperparameter tuning for {cfg['optuna']['agent_type']} agent completed.")
    logging.info(f"Best parameters: {study.best_params}")
    logging.info(f"Best value: {study.best_value}")


if __name__ == "__main__":
    optuna_tuning_experiment()

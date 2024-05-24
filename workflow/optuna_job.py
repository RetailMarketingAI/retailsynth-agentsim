from pathlib import Path
import hydra
import os
from omegaconf import OmegaConf
import logging
from retail_agent.utils.storage_util import CloudStorageUtil, LocalStorageUtil
from retail_agent.utils.train_eval_util import train_agent, collect_eval_trajectory


@hydra.main(version_base=None, config_path="cfg", config_name="agent_training")
def main(cfg) -> None:
    cfg = OmegaConf.to_object(cfg)

    if cfg["experiment"]["storage"] == "cloud":
        storage = CloudStorageUtil(cfg["cloud"])
    elif cfg["experiment"]["storage"] == "local":
        storage = LocalStorageUtil()

    agent_path = Path(cfg["experiment"]["output_dir"], "agent", cfg["agent"]["agent_type"], "tuning_job", str(cfg["experiment"]["run_id"]), str(cfg["experiment"]["instance_id"]))
    if cfg["step"] == "train":
        logging.info("Training agent.")
        agent, training_loss = train_agent(cfg)
        storage.store_policy(agent, str(Path(agent_path, "checkpoint")))
        storage.store_obj(training_loss, str(Path(agent_path, "training_loss.pkl")))
        storage.store_obj(cfg["agent"]["agent_params"], str(Path(agent_path.parent, "agent_cfg.pkl")))
    elif cfg["step"] == "eval":
        logging.info("Collecting evaluation trajectory.")
        job_id = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", "0"))
        agent_path = str(agent_path)
        collect_eval_trajectory(cfg, job_id, agent_path)
    else:
        raise ValueError(f"Unknown experiment id: {cfg['experiment']['experiment_id']}")


if __name__ == "__main__":
    main()

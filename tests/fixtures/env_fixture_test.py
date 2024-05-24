import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from retail_agent.utils.trajectory_util import get_env

@pytest.fixture(scope="module", autouse=True)
def config():
    with initialize(version_base=None, config_path="../cfg"):
        cfg = compose(config_name="agent_training")
        cfg = OmegaConf.to_object(cfg)
    return cfg


@pytest.fixture(scope="module")
def env(config):
    config["pyenv"]["_target_"] == "retail_agent.envs.retail_env.RetailEnv"
    return get_env(config)

@pytest.fixture(scope="module")
def cb_env(config):
    return get_env(config)

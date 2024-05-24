from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.agents.ppo import ppo_agent
import tensorflow as tf
import hydra
from typing import Dict, Any

from retail_agent.agents.dqn import FeatureConcatLayer


def get_ppo_agent(
    time_step_spec: tf.TensorSpec,
    action_spec: tf.TensorSpec,
    config: Dict[str, Any],
):
    """Create a tf-agents PPO agent.

    Parameters
    ----------
        time_step_spec (tf.TensorSpec): time step spec of the environment
        action_spec (tf.TensorSpec): action spec of the environment
        config (Dict[str, Any]): hyperparameters for the DQN agent

    Returns
    -------
        ppo_agent.PPOAgent: initialized PPO agent
    """
    agent_config = config.copy()
    # Define preprocessing layers
    context_features = agent_config.pop("context_features", ["avg_purchase_price"])
    preprocessing_layers = FeatureConcatLayer(context_features=context_features)

    # Define the actor and value networks with preprocessing layers
    action_net_config = agent_config.pop("actor_net", {"fc_layer_params": (100,)})
    actor_net = actor_distribution_network.ActorDistributionNetwork(time_step_spec.observation, action_spec, preprocessing_combiner=preprocessing_layers, **action_net_config)
    value_net_config = agent_config.pop("value_net", {"fc_layer_params": (100,)})
    value_net = value_network.ValueNetwork(time_step_spec.observation, preprocessing_combiner=preprocessing_layers, **value_net_config)

    # create the optimizer
    optimizer_config = agent_config.pop("optimizer", ValueError("optimizer not found in config"))
    optimizer = hydra.utils.instantiate(optimizer_config)

    # Define the PPO agent
    ppo = ppo_agent.PPOAgent(time_step_spec, action_spec, optimizer=optimizer, actor_net=actor_net, value_net=value_net, **agent_config)

    ppo.initialize()

    return ppo

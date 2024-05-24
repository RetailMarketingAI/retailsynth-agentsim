import tensorflow as tf
from tf_agents.networks import network
from tf_agents.trajectories import time_step
from typing import Dict, Any, List
import hydra
from tf_agents.bandits.agents.neural_boltzmann_agent import NeuralBoltzmannAgent

from retail_agent.agents.dqn import FeatureConcatLayer


class RewardNet(network.Network):
    """Neural net for reward prediction.

    Parameters
    ----------
        observation_spec (tf.TensorSpec): observation spec of the environment
        action_spec (tf.TensorSpec): action spec of the environment
        context_features (List[str]): context features to input to the neural net
        n_arm (int): number of arms in the bandit
    """

    def __init__(self, observation_spec: tf.TensorSpec, action_spec: tf.TensorSpec, context_features: List[str], hidden_layers: tuple = (128, 64), n_arms: int = 11):
        super(RewardNet, self).__init__(input_tensor_spec=observation_spec, state_spec=(), name="RewardNet")

        self._action_spec = action_spec

        # Define layers
        self._hidden_layers = [FeatureConcatLayer(context_features)]
        for units in hidden_layers:
            self._hidden_layers.append(tf.keras.layers.Dense(units, activation="relu"))

        self._output_layer = tf.keras.layers.Dense(n_arms, activation=None)

    def call(self, observations, step_type=(), network_state=()):
        # Pass observations through hidden layers
        hidden = observations
        for layer in self._hidden_layers:
            hidden = layer(hidden)

        # Compute reward
        reward = self._output_layer(hidden)

        return reward, network_state


def get_neural_boltzmann_agent(
    time_step_spec: time_step.TimeStep,
    action_spec: tf.TensorSpec,
    config: Dict[str, Any],
):
    """Create a tf-agents Neural Boltzmann agent.

    Parameters
    ----------
        time_step_spec (tf.TensorSpec): time step spec of the environment
        action_spec (tf.TensorSpec): action spec of the environment
        config (Dict[str, Any]): hyperparameters for the DQN agent

    Returns
    -------
        neural_boltzmann_agent.NeuralBoltzmannAgent: initialized Neural Boltzmann agent
    """
    agent_config = config.copy()
    context_features = agent_config.pop("context_features", ["avg_purchase_price"])
    n_arms = action_spec.maximum - action_spec.minimum + 1
    hidden_layers = agent_config.pop("hidden_layers", (128, 64))
    reward_net = RewardNet(
        observation_spec=time_step_spec.observation,
        action_spec=action_spec,
        context_features=context_features,
        hidden_layers=hidden_layers,
        n_arms=n_arms,
    )

    # create the optimizer
    optimizer_config = agent_config.pop("optimizer", ValueError("optimizer not found in config"))
    optimizer = hydra.utils.instantiate(optimizer_config)

    agent = NeuralBoltzmannAgent(time_step_spec=time_step_spec, action_spec=action_spec, reward_network=reward_net, optimizer=optimizer, **agent_config)
    agent.initialize()
    return agent

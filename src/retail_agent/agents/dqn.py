from typing import Dict, Any
import hydra
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tensorflow.keras.layers import Dense, Concatenate


class FeatureConcatLayer(tf.keras.layers.Layer):
    """Concatenate the observation as one array to be the input of the DQN network."""

    def __init__(self, context_features):
        super().__init__()
        self.context_features = context_features

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # concatenating all the features into one array
        # output array in shape of (n_customer, n_feature_dim)
        output_features = []
        for name, input in inputs.items():
            if name in self.context_features:
                if name in ["previous_transaction", "product_price", "observed_customer_product_feature"]:
                    input = tf.math.reduce_mean(tf.cast(input, tf.float64), axis=(-1))
                    input = tf.cast(input, tf.float32)

                output_features.append(tf.expand_dims(input, axis=-1))

        return Concatenate(axis=-1)(output_features)


def get_dqn_agent(
    time_step_spec: tf.TensorSpec,
    action_spec: tf.TensorSpec,
    config: Dict[str, Any],
) -> dqn_agent.DqnAgent:
    """Create a tf-agents DQN agent.

    Parameters
    ----------
        time_step_spec (tf.TensorSpec): time step spec of the environment
        action_spec (tf.TensorSpec): action spec of the environment
        config (Dict[str, Any]): hyperparameters for the DQN agent

    Returns
    -------
        dqn_agent.DqnAgent: initialized DQN agent
    """
    agent_config = config.copy()
    # create the Q network
    context_features = agent_config.pop("context_features", ["avg_purchase_price"])
    n_arms = action_spec.maximum - action_spec.minimum + 1
    q_network_config = agent_config.pop("q_network", ValueError("q_network not found in config"))
    q_network = [FeatureConcatLayer(context_features)]
    for layer_name, layer_config in q_network_config.items():
        q_network.append(hydra.utils.instantiate(layer_config))
    q_network.append(Dense(n_arms, activation=None))
    q_network = sequential.Sequential(q_network)

    # create the optimizer
    optimizer_config = agent_config.pop("optimizer", ValueError("optimizer not found in config"))
    optimizer = hydra.utils.instantiate(optimizer_config)

    agent = dqn_agent.DqnAgent(
        time_step_spec,
        action_spec,
        q_network=q_network,
        optimizer=optimizer,
        **agent_config,
    )
    agent.initialize()
    return agent

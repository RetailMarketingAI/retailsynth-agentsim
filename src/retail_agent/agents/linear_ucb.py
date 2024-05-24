from typing import Union, Tuple, List
import tensorflow as tf
from tf_agents.bandits.agents.lin_ucb_agent import LinearUCBAgent


def make_context_splitter(context_features: List[str] = ["avg_purchase_price"]):
    """Create a context splitter for linear ucb agent using factory function.

    Parameters
    ----------
        context_features (List[str], optional): list of context features to keep in the context. Defaults to ["customer_features"].
    """

    def context_splitter(observation: Union[tf.TensorSpec, tf.Tensor]) -> Tuple:
        # if given observations contain only tf.TensorSpec, this function is usually called in agent initialization
        # we need to return the definition of context space
        if all([isinstance(value, tf.TensorSpec) for value in observation.values()]):
            feature_dim = 0
            for name, feature in observation.items():
                if name in context_features:
                    if name == "customer_features":
                        feature_dim += feature.shape[-1]
                    else:
                        feature_dim += 1
            output = tf.TensorSpec(shape=(feature_dim,), dtype=tf.float32)
        # if given observations contain tf.Tensor, this function is usually called in data collection
        # we need to return the actual context tensor
        elif all([isinstance(value, tf.Tensor) for value in observation.values()]):
            output = []
            for name, feature in observation.items():
                if name in context_features:
                    if name in ["previous_transaction", "product_price", "observed_customer_product_feature"]:
                        feature = tf.math.reduce_mean(tf.cast(feature, tf.float64), axis=(-1))
                        feature = tf.cast(feature, tf.float32)

                    output.append(tf.expand_dims(feature, axis=-1))
            output = tf.concat(output, axis=-1)
        # tf agent object expects this context splitter to return a tuple of (context, action)
        # however, we don't need action when training or running the agent. Thus, we return None for action.
        return output, None

    def constant_context_splitter(observation: Union[tf.TensorSpec, tf.Tensor]) -> Tuple:
        # return 1 as the context
        if all([isinstance(value, tf.TensorSpec) for value in observation.values()]):
            output = tf.TensorSpec(shape=(1), dtype=tf.float32)
        elif all([isinstance(value, tf.Tensor) for value in observation.values()]):
            n_customer = observation["observed_customer_product_feature"].shape[0]
            output = tf.ones((n_customer, 1), dtype=tf.float32)
        return output, None

    if len(context_features) == 0:
        return constant_context_splitter
    else:
        return context_splitter


def get_linear_ucb_agent(
    time_step_spec: tf.TensorSpec,
    action_spec: tf.TensorSpec,
    config={},
) -> LinearUCBAgent:
    # create a tf-agent linear ucb agent
    agent_config = config.copy()
    context_features = agent_config.pop("context_features", ["avg_purchase_price"])
    context_splitter = make_context_splitter(context_features)
    agent = LinearUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        observation_and_action_constraint_splitter=context_splitter,
        **agent_config,
    )
    agent.initialize()
    return agent

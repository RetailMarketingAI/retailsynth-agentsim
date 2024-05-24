import tensorflow as tf
import numpy as np


def get_agent_parameter_count(agent) -> int:
    """Gather the number of trainable parameters in the agent.

    Parameters
    ----------
        agent: tf agent object

    Returns
    -------
        int: number of trainable parameters in the agent
    """
    param_count = 0
    for variable in agent.trainable_variables:
        shape = tf.shape(variable).numpy()
        param_count += np.prod(shape)
    return param_count

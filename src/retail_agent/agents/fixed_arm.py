import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.policies import tf_policy
from tf_agents.trajectories.policy_step import PolicyStep


class FixedArmPolicy(tf_policy.TFPolicy):
    """A static collection policy that recommends a specific arm all the time.

    Parameters
    ----------
        time_step_spec (tf.TensorSpec): time step spec of the environment
        action_spec (tf.TensorSpec): action spec of the environment
        chosen_arm (int): index of selected arm
        name (str): agent name
    """

    def __init__(self, time_step_spec, action_spec, arm=0, name=None):
        super().__init__(policy_state_spec=(), time_step_spec=time_step_spec, action_spec=action_spec, name=name)
        self.arm = arm

    def _variables(self):
        return []

    def _action(self, time_step, policy_state, seed):
        # broadcast the selected arm to the expected shape
        batch_size = time_step.reward.shape[0]
        chosen_arm = tf.ones(batch_size, dtype=tf.int32) * self.arm
        return PolicyStep(chosen_arm, state=(), info=())


class FixedArmAgent(tf_agent.TFAgent):
    """A baseline agent that recommends a specific arm all the time.

    Parameters
    ----------
        time_step_spec (tf.TensorSpec): time step spec of the environment
        action_spec (tf.TensorSpec): action spec of the environment
        chosen_arm (int): index of selected arm
        name (str): agent name
    """

    def __init__(self, time_step_spec: tf.TensorSpec, action_spec: tf.TensorSpec, chosen_arm: int, name: str = "FixedArmAgent"):
        self._name = f"{name}_{chosen_arm}"
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=FixedArmPolicy(time_step_spec, action_spec, chosen_arm),
            collect_policy=FixedArmPolicy(time_step_spec, action_spec, chosen_arm),
            train_sequence_length=None,
            debug_summaries=False,
            summarize_grads_and_vars=False,
            train_step_counter=tf.Variable(0),
        )

    def _train(self, experience, weights=None):
        # return empty loss
        return tf_agent.LossInfo(loss=tf.constant(0.0), extra=())

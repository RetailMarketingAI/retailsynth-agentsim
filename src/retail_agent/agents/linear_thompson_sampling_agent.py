from tf_agents.bandits.agents.linear_thompson_sampling_agent import LinearThompsonSamplingAgent
import tensorflow as tf

from retail_agent.agents.linear_ucb import make_context_splitter


def get_linear_ts_agent(
    time_step_spec: tf.TensorSpec,
    action_spec: tf.TensorSpec,
    config={},
) -> LinearThompsonSamplingAgent:
    # create a tf-agent linear ts agent
    agent_config = config.copy()
    context_features = agent_config.pop("context_features", ["avg_purchase_price"])
    context_splitter = make_context_splitter(context_features)
    agent = LinearThompsonSamplingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        observation_and_action_constraint_splitter=context_splitter,
        **agent_config,
    )
    agent.initialize()
    return agent

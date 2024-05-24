from tf_agents.agents.random.random_agent import RandomAgent
from retail_agent.agents.dqn import get_dqn_agent
from retail_agent.agents.linear_thompson_sampling_agent import get_linear_ts_agent
from retail_agent.agents.linear_ucb import get_linear_ucb_agent
from retail_agent.agents.fixed_arm import FixedArmAgent
from retail_agent.agents.neural_boltzmann import get_neural_boltzmann_agent
from retail_agent.agents.ppo import get_ppo_agent
from tf_agents.agents import tf_agent


def get_retail_agent(
    agent_type: str,
    time_step_spec,
    action_spec,
    config={},
    **kwargs,
) -> tf_agent.TFAgent:
    """Initialize tensorflow agent based on specs and config.

    Parameters
    ----------
        agent_type (str): support 'random', 'linear_ts', 'linear_ucb', 'dqn', 'ppo', 'neural_boltzmann'
        time_step_spec (tf.TensorSpec): time step spec of the environment
        action_spec (tf.TensorSpec): action spec of the environment
        config (dict): additional parameters to initialize the agent

    Returns
    -------
        tf_agent.TFAgent
    """
    if agent_type == "random":
        agent = RandomAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            name="Random",
        )
        agent.initialize()
        return agent
    elif agent_type == "linear_ts":
        return get_linear_ts_agent(time_step_spec=time_step_spec, action_spec=action_spec, config=config)
    elif agent_type == "linear_ucb":
        return get_linear_ucb_agent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            config=config,
        )
    elif agent_type == "dqn":
        return get_dqn_agent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            config=config,
        )
    elif agent_type == "fixed_arm":
        return FixedArmAgent(time_step_spec=time_step_spec, action_spec=action_spec, **config)
    elif agent_type == "neural_boltzmann":
        return get_neural_boltzmann_agent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            config=config,
        )
    elif agent_type == "ppo":
        return get_ppo_agent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            config=config,
        )
    # More agents to be added here
    else:
        raise NotImplementedError(f"Agent type {agent_type} not implemented yet.")

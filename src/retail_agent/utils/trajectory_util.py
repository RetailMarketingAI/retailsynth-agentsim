import hydra
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import TimeStep
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.environments import tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories import from_transition

from retailsynth.synthesizer.config import initialize_synthetic_data_setup

from retail_agent.agents.fixed_arm import FixedArmPolicy
from retail_agent.envs.retail_env import RetailEnv
from retail_agent.envs.retail_synthesizer import TargetedCouponDataSynthesizer


def get_env(config, synthesizer=None):
    """Initialize environment.

    Parameters
    ----------
        config (Dict[str, Any]): complete configuration to initialize env and specify number of steps to collect.
        synthesizer (TargetedCouponDataSynthesizer): synthesizer used in agent training. If None, initialize a new one using the config.

    """
    synthesizer_config = initialize_synthetic_data_setup(config["synthetic_data"])
    synthesizer_init = TargetedCouponDataSynthesizer(synthesizer_config)
    pyenv = hydra.utils.instantiate(config["pyenv"], synthesizer=synthesizer_init)
    pyenv.reset()
    if synthesizer is not None:
        pyenv.synthesizer = synthesizer
    env = tf_py_environment.TFPyEnvironment(pyenv)
    return env


def get_random_policy(env):
    """Initialize random policy using env time step spec and action spec."""
    return RandomTFPolicy(
        env.time_step_spec(),
        env.action_spec(),
    )


def get_fixed_arm_policy(env, arm):
    """Initialize fixed arm policy using env time step spec and action spec."""
    return FixedArmPolicy(
        env.time_step_spec(),
        env.action_spec(),
        arm,
    )


def describe_trajectory_attribute(obj):
    """Interpret trajectory attributes into a string."""
    s = ""
    s += f"shape: {obj.shape}\n"
    s += f"dtype: {obj.dtype}"
    return s


def get_one_trajectory(env: RetailEnv, policy: TFPolicy, n_step: int, time_step: TimeStep = None) -> TFUniformReplayBuffer:
    """Run one trajectory and store the trajectory in the replay buffer.

    Parameters
    ----------
        env (RetailEnv)
        policy (TFPolicy)
        n_step (int): number of steps to run in the trajectory
        time_step (TimeStep, optional): initial time step to start with. If None, reset the environment to get the time step. Defaults to None.

    Returns
    -------
        TFUniformReplayBuffer: the replay buffer containing the trajectory
    """
    replay_buffer = TFUniformReplayBuffer(
        data_spec=policy.trajectory_spec,
        batch_size=env.batch_size,
        max_length=n_step,
    )
    time_step = time_step or env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    for _ in range(n_step):
        action_step = policy.action(time_step=time_step, policy_state=policy_state)
        next_time_step = env.step(action_step.action)
        replay_buffer.add_batch(from_transition(time_step, action_step, next_time_step))
        time_step = next_time_step
        policy_state = action_step.state

    return replay_buffer.gather_all()

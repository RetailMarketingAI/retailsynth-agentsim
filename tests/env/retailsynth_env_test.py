from retail_agent.utils.trajectory_util import get_random_policy


def test_env_functionality(env):
    env._max_steps = 2
    random_policy = get_random_policy(env)
    time_step = env.reset()
    while not all(time_step.is_last()):
        action_step = random_policy.action(time_step)
        time_step = env.step(action_step.action)

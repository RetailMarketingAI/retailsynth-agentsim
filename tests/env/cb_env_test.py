from retail_agent.utils.trajectory_util import get_random_policy


def test_env_functionality(cb_env):
    cb_env._max_steps = 2
    random_policy = get_random_policy(cb_env)
    time_step = cb_env.reset()
    while not all(time_step.is_last()):
        action_step = random_policy.action(time_step)
        time_step = cb_env.step(action_step.action)

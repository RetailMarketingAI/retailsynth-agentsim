---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Repo Introduction with A Simplified Retail Scenario

```python
from hydra import compose, initialize
from omegaconf import OmegaConf
import logging
from collections import defaultdict
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

from retail_agent.utils.trajectory_util import get_env, get_random_policy, get_one_trajectory, get_fixed_arm_policy
from retail_agent.utils.train_eval_util import get_last_time_step, prepare_trajectory
from retail_agent.agents import get_retail_agent
from retail_agent.utils.segment_util import segment_customer_by_price_sensitivity
from retail_agent.utils.metric_util import generate_evaluation_report, generate_segmented_evaluation_report
from retail_agent.utils.viz_util import gather_all_customer_report
```

# Overview

This notebook shows an example of how to use `RetailSynth-AgentSim` as a bandit environment to train a bandit agent. We pick a basic agent, LinUCB, as the starting point to prototype a training routine, and showcase the visualization toolkit. We use hydra to manage configurations for the whole experiment workflow.

We do not provide a detail commentary on the results contained in this notebook, as the environment here is very simplified. Interested readers should checkout the paper referenced in the README for a more detailed discussion.

```python
with initialize(version_base=None, config_path="../tests/cfg"):
    config = compose(config_name="agent_training")
    config = OmegaConf.to_object(config)
```

```python
# Quick walkthrough of the whole process
n_train_traj = 3
n_train_step = 50
n_eval_traj = 3
n_eval_step = 20

# Collect offline data
train_trajectories = []
train_synthesizer = []
last_time_step = []
for _ in range(n_train_traj):
    env = get_env(config)
    random_policy = get_random_policy(env)
    trajectory = get_one_trajectory(env, random_policy, n_train_step)
    train_trajectories.append(trajectory)
    train_synthesizer.append(env.synthesizer)
    last_time_step.append(get_last_time_step(trajectory))

# Train LinUCB agent
agent = get_retail_agent(agent_type="linear_ucb", time_step_spec=env.time_step_spec(), action_spec=env.action_spec(), config=config["agent"]["agent_params"])
train_trajectories = prepare_trajectory(train_trajectories)
loss = agent.train(train_trajectories)

# Evaluate LinUCB agent
eval_trajectories = []
for i in range(n_eval_traj):
    env = get_env(config, train_synthesizer[i])
    trajectory = get_one_trajectory(env, agent.policy, n_eval_step, last_time_step[i])
    eval_trajectories.append(trajectory)
report = generate_evaluation_report({"LinUCB": eval_trajectories}, window_length=5, product_category_mapping=train_synthesizer[0].product_category_mapping, n_arm=env.n_coupon_levels, coupon_unit=env.coupon_unit)

report.keys()
```

## Environment

In this notebook, we are going to explore a simplified retail environment with 100 customers and 20 products from 3 categories.

```python
env = get_env(config)

print("n_customer: ", env.synthesizer.n_customer)
print("n_product: ", env.synthesizer.n_product)
print("n_category: ", env.synthesizer.n_category)
```

**Observation space** is a dictionary containing 4 primitive features 
- `previous_transaction`
- `product_price`
- `marketing_feature`
- `observed_customer_product_feature`

and 5 derived features
- `avg_purchase_quantity`
- `avg_purchase_probability`
- `avg_purchase_price`
- `avg_redeemed_discount`
- `avg_purchase_discount`

```python
env.observation_spec()
```

**Action Space** contains the store-wide coupon the retailer offers to each customer. This scaler feature is an integer ranging from 0 to 2, representing the distinct coupon levels. 0 means no coupon, 1 means 5% off and 2 means 10% off.

```python
env.action_spec()
```

**Reward space** represents the money spent by each customer.

```python
env.reward_spec()
```

## Train LinUCB Agent

The LinUCB agent is a linear model that uses the Upper Confidence Bound (UCB) algorithm to balance exploration and exploitation. The agent will learn a linear model of reward distribution for each action and use the UCB algorithm to select the action with the highest expected reward.


We provide utility functions to collect offline training data using random policy. 

```python
# Collect offline data
n_train_traj = 3
n_train_step = 50

train_trajectories = []
# store the training synthesizer and last time step of each trajectory
# to make sure the evaluation starts from the last time step of the training trajectory
train_synthesizer = []
last_time_step = []
for _ in range(n_train_traj):
    env = get_env(config)
    random_policy = get_random_policy(env)
    trajectory = get_one_trajectory(env, random_policy, n_train_step)
    train_trajectories.append(trajectory)
    train_synthesizer.append(env.synthesizer)
    last_time_step.append(get_last_time_step(trajectory))
```

All of our retail agents are implemented using `tf-agents`. We train LinUCB agent with offline data in one batch, but if you'd like to train agents with mini-batches for consideration of data size and computing resources, you can maintain a for loop of training epochs and monitor the training losses.

```python
agent = get_retail_agent(
    agent_type="linear_ucb",  # specify the agent type
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    config=config["agent"]["agent_params"],  # additional agent parameters you'd like to tune
)
train_trajectories = prepare_trajectory(train_trajectories)
loss = agent.train(train_trajectories)
```

## Evaluate LinUCB Agent

We will compare the trained LinUCB policy with fixed arm policies and random policy to illustrate our visualization toolkit.

```python
def evaluate_policy(policy, n_eval_traj, n_eval_step, train_synthesizer, last_time_step):
    eval_trajectories = []
    for i in range(n_eval_traj):
        env = get_env(config, train_synthesizer[i])
        # make sure the evaluation starts from the last time step of the training trajectory
        trajectory = get_one_trajectory(env, policy, n_eval_step, last_time_step[i])
        eval_trajectories.append(trajectory)
    return eval_trajectories


evaluation_results = {}
evaluation_results["LinUCB"] = evaluate_policy(agent.policy, n_eval_traj, n_eval_step, train_synthesizer, last_time_step)
policy = get_random_policy(env)
evaluation_results["Random"] = evaluate_policy(policy, n_eval_traj, n_eval_step, train_synthesizer, last_time_step)
for arm_idx in range(env.n_coupon_levels):
    policy = get_fixed_arm_policy(env, arm_idx)
    evaluation_results[f"FixedArm_{arm_idx}"] = evaluate_policy(policy, n_eval_traj, n_eval_step, train_synthesizer, last_time_step)

report = generate_evaluation_report(evaluation_results, window_length=5, product_category_mapping=train_synthesizer[0].product_category_mapping, n_arm=env.n_coupon_levels, coupon_unit=env.coupon_unit)
```

In the report dictionary, we compute the following:
- `metric_df`: a data frame containing the primary objective (accumulated revenue) and secondary metrics (accumulated demand, customer retention, category penetration, and empirical discount) for all policies.
- `selected_arm_df`: a data frame containing the probability of each arm being chosen for all policies.
- `revenue_df`: a data frame containing the accumulated revenue for all policies across all time step in evaluation trajectories. 

```python
report["metric_df"].head()
```

```python
report["selected_arm_df"].head()
```

```python
report["revenue_df"].head()
```

```python
fig, ax = gather_all_customer_report(
    report,
    order=["LinUCB", "Random", "FixedArm_0", "FixedArm_1", "FixedArm_2"],
)
ax[-2].legend(bbox_to_anchor=(0.8, -0.3))
```

We can also compare the same suite of metrics on customer segments. We partition customers into 3 segments based on their price sensitivity coefficients.

```python
n_segment = 2
customer_segment = segment_customer_by_price_sensitivity(train_synthesizer[0], n_segment)

segmented_evaluation_results = defaultdict(dict)
for segment in customer_segment.keys():
    for agent_name, traj in evaluation_results.items():
        seg_idx = customer_segment[segment]
        segmented_evaluation_results[segment][agent_name] = [tf.nest.map_structure(lambda x: tf.gather(x, seg_idx), t) for t in traj]

segmented_report = generate_segmented_evaluation_report(segmented_evaluation_results, window_length=5, product_category_mapping=train_synthesizer[0].product_category_mapping, n_arm=env.n_coupon_levels, coupon_unit=env.coupon_unit)
```

```python
segmented_report["metric_df"].head()
```

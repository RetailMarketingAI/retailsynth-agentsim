from typing import List, Dict
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import pandas as pd
from collections import defaultdict
from tf_agents.trajectories import Trajectory
from tf_agents.bandits.agents.lin_ucb_agent import LinearUCBAgent
from sklearn.feature_selection import mutual_info_regression


def get_revenue_by_coupon_level(traj: Trajectory) -> pd.DataFrame:
    """Get the revenue for each arm from one trajectory.

    Parameters
    ----------
        traj (Trajectory): a trajectory

    Returns
    -------
        pd.DataFrame: a dataframe contains columns ["arm", "reward"]
    """
    dfs = []
    arms = tf.unique(tf.reshape(traj.action, (-1,))).y
    for arm in arms:
        df = pd.DataFrame(columns=["arm", "reward"])
        df["reward"] = traj.reward[traj.action == arm]
        df["arm"] = arm.numpy()
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs


def compute_linear_ucb_arm_prob(agent: LinearUCBAgent, n_arm: int, eval_trajectories: List[Trajectory], customer_idx=None) -> pd.DataFrame:
    theta = {arm_idx: tf.reshape(agent.policy.variables()[-1][arm_idx], (-1, 1)) for arm_idx in range(n_arm)}
    alpha = {arm_idx: agent.policy.variables()[arm_idx] for arm_idx in range(n_arm)}
    beta = {arm_idx: agent.policy.variables()[n_arm + arm_idx] for arm_idx in range(n_arm)}
    alpha_inv = {arm_idx: tf.linalg.inv(alpha[arm_idx]) for arm_idx in range(n_arm)}

    context_recorder = []
    for eval_traj in eval_trajectories:
        context, _ = agent._observation_and_action_constraint_splitter(eval_traj.observation)
        if customer_idx is not None:
            context = context[customer_idx]
        context_recorder.append(context)
    context = tf.concat(context_recorder, axis=0)
    context = tf.reshape(context, (-1, 1, context.shape[-1]))

    arm_probs = []
    for arm_idx in range(n_arm):
        arm_prob = tf.matmul(context, theta[arm_idx]) + agent.alpha * tf.sqrt(tf.matmul(tf.matmul(context, alpha_inv[arm_idx]), tf.transpose(context, (0, 2, 1))))
        arm_probs.append(arm_prob)

    df = pd.DataFrame(tf.argmax(tf.concat(arm_probs, axis=1), axis=1).numpy().flatten(), columns=["arm"])
    df = df.groupby("arm").size() / df.size
    df = df.to_frame(name="prob").reset_index()

    return df


def get_accumulated_revenues(
    trajectories: List[Trajectory],
) -> np.ndarray:
    """Get a time series of accumulated from multiple trajectories.

    Parameters
    ----------
        trajectories (List[Trajectory]): a list of trajectories

    Returns
    -------
        np.ndarray: accumulated revenues time series, in shape of (n_traj, n_step)
    """
    # revenues array in shape of (n_traj, n_customer, n_step)
    revenues = []
    for traj in trajectories:
        accumulated_revenue = traj.reward.numpy().cumsum(axis=-1).sum(axis=0)
        revenues.append(accumulated_revenue)
    return np.stack(revenues)


def get_selected_arms(
    trajectories: List[Trajectory],
    customer_idx: List[int] = None,
) -> pd.DataFrame:
    """Get the selected arms from multiple trajectories.

    Parameters
    ----------
        eval_result (List[Trajectory]): a list of trajectories
        customer_idx (List[int], optional): a list of customer indices to count selected arms. Defaults to None.

    Returns
    -------
        pd.DataFrame: dataframe contains columns ["arm", "count", "prob", "traj_id", "agent"]
    """

    def count_arm_chosen_times(actions):
        reshaped_actions = tf.reshape(actions, (-1,))
        counts = tf.math.bincount(reshaped_actions).numpy()
        arms = np.arange(len(counts))
        df = pd.DataFrame({"arm": arms, "count": counts})
        df["prob"] = df["count"] / df["count"].sum()
        return df

    dfs = []
    if customer_idx is not None:
        actions = [tf.gather(traj.action, customer_idx, axis=0) for traj in trajectories]
    else:
        actions = [traj.action for traj in trajectories]
    for traj_id, action in enumerate(actions):
        df = count_arm_chosen_times(action)
        df["trajectory_id"] = traj_id
        dfs.append(df)
    return pd.concat(dfs)


def get_accumulated_revenues_for_multiple_policies(
    eval_episodes: Dict[str, List[Trajectory]],
) -> pd.DataFrame:
    """Get a time series of accumulated from multiple trajectories for multiple policies.

    Parameters
    ----------
        eval_episodes (Dict[str, List[Trajectory]]): a list of list of trajectories
        mode (str, optional): mode to calculate accumulated revenues, supporting "total", "mean" and "customer". Defaults to "total".
        customer_idx (List[int], optional): a list of customer indices to calculate accumulated revenues. Defaults to None.

    Raises
    ------
        ValueError: if mode to calculate accumulated revenues is not supported

    Returns
    -------
        pd.DataFrame: a dataframe contains the accumulated revenues time series for each policy
    """
    accumulated_revenues = []
    for agent_name, trajectories in eval_episodes.items():
        policy_revenue = get_accumulated_revenues(trajectories)
        n_traj, n_step = policy_revenue.shape
        df = pd.DataFrame(columns=["agent", "revenue", "timestep"])
        df["revenue"] = policy_revenue.flatten()
        df["agent"] = agent_name
        df["timestep"] = np.tile(np.arange(n_step), n_traj) + 1
        accumulated_revenues.append(df)
    return pd.concat(accumulated_revenues)


def get_arm_propensities_for_multiple_policies(
    eval_episodes: Dict[str, List[Trajectory]],
    customer_idx: List[int] = None,
) -> pd.DataFrame:
    """Find the optimal arm for each policy.

    Parameters
    ----------
        eval_episodes (Dict[str, List[Trajectory]]): a dictionary contains the trajectories for each policy
        customer_idx (List[int], optional): a list of customer indices to count selected arms. Defaults to None.

    Returns
    -------
        pd.DataFrame: a dataframe contains the probabilities of arms being chosen for each policy
    """
    selected_arms = []
    for agent, trajectories in eval_episodes.items():
        selected_arm_count = get_selected_arms(trajectories, customer_idx)
        selected_arm_count["agent"] = agent
        selected_arms.append(selected_arm_count)
    return pd.concat(selected_arms)


def get_arm_propensities_at_time_step(trajectories: List[Trajectory], n_arm: int, time_step: List[int] = None, customer_idx: List[int] = None) -> pd.DataFrame:
    """Compute probabilities of each arm being chosen at given time steps.

    Parameters
    ----------
        trajectories (List[Trajectory]): a list contains the trajectories for on policy.
        n_arm (int): number of arms in the bandit.
        time_step (List[int]): list of time steps to evaluate the probability of arm being chosen. If None, compute at the last time step.
        customer_idx (List[int], optional): a list of customer indices to count selected arms. Defaults to None.

    Returns
    -------
        pd.DataFrame: a dataframe contains the probabilities of arms being chosen for each policy
    """
    if time_step is None:
        prob = np.zeros(n_arm)
        actions = np.concatenate([traj.action.numpy() for traj in trajectories])
        unique_elements, counts = np.unique(actions, return_counts=True)
        prob[unique_elements] = counts / counts.sum()
        arm_prob = pd.DataFrame(
            {
                "arm": np.arange(n_arm),
                "prob": prob,
            }
        )
        return arm_prob
    else:
        arm_prob = []
        for time in time_step:
            prob = np.zeros(n_arm)
            actions = np.concatenate([traj.action.numpy()[:, time] for traj in trajectories])
            unique_elements, counts = np.unique(actions, return_counts=True)
            prob[unique_elements] = counts / counts.sum()
            df = pd.DataFrame({"arm": np.arange(n_arm), "prob": prob, "time": time})
            arm_prob.append(df)
        return pd.concat(arm_prob)


def get_arm_propensities_at_time_step_for_multiple_policies(
    eval_episodes: Dict[str, List[Trajectory]],
    n_arm: int,
    time_step: List[int] = None,
    customer_idx: List[int] = None,
) -> pd.DataFrame:
    """Compute probability of arm being chosen at given time steps.

    Parameters
    ----------
        eval_episodes (Dict[str, List[Trajectory]]): a dictionary contains the trajectories for each policy
        n_arm (int): number of arms in the bandit.
        time_step (List[int]): list of time steps to evaluate the probability of arm being chosen. If None, compute at the last time step.
        customer_idx (List[int], optional): a list of customer indices to count selected arms. Defaults to None.

    Returns
    -------
        pd.DataFrame: a dataframe contains the probabilities of arms being chosen for each policy
    """
    selected_arms = []
    for agent, trajectories in eval_episodes.items():
        selected_arm_count = get_arm_propensities_at_time_step(trajectories, n_arm, time_step, customer_idx)
        selected_arm_count["agent"] = agent
        selected_arms.append(selected_arm_count)
    return pd.concat(selected_arms)


def compute_accumulated_revenue(eval_trajectories: List[Trajectory], **kwargs) -> Dict[str, np.ndarray]:
    """Compute the final revenue from multiple trajectories.

    Parameters
    ----------
        eval_trajectories (List[Trajectory]): a list of trajectories

    Returns
    -------
        Dict[str, np.ndarray]: a dictionary contains the final revenue
    """
    revenues = []
    for traj in eval_trajectories:
        accumulated_revenue = traj.reward.numpy().sum()
        revenues.append(accumulated_revenue)
    return {
        "accumulated_revenue": np.stack(revenues),
    }


def compute_accumulated_demand(eval_trajectories: List[Trajectory], **kwargs) -> Dict[str, np.ndarray]:
    """Compute the final demand from multiple trajectories.

    Parameters
    ----------
        eval_trajectories (List[Trajectory]): a list of trajectories

    Returns
    -------
        Dict[str, np.ndarray]: a dictionary contains the final demand
    """
    demands = []
    for traj in eval_trajectories:
        accumulated_demand = traj.observation["previous_transaction"].numpy().astype(np.float64).sum()
        demands.append(accumulated_demand)
    return {"accumulated_demand": np.stack(demands)}


def compute_customer_retention(eval_trajectories: List[Trajectory], window_length: int = 3, **kwargs) -> Dict[str, np.ndarray]:
    """Compute the final customer retention from multiple trajectories.

    Parameters
    ----------
        eval_trajectories (List[Trajectory]): a list of trajectories
        customer_idx (List[int], optional): a list of customer indices to calculate final customer retention. Defaults to None.
        window_length (int, optional): the number of no-purchase time steps to define a customer as inactive. Defaults to 3.

    Returns
    -------
        Dict[str, np.ndarray]: a dictionary contains the customer retention
    """
    customer_retentions = []
    for traj in eval_trajectories:
        purchase_event = traj.reward.numpy() > 0
        n_customer, n_step = purchase_event.shape
        inactive_days = np.argmax(purchase_event[..., ::-1], axis=-1)
        no_purchase = np.sum(purchase_event, axis=-1) == 0
        inactive_days[no_purchase] = n_step
        customer_retention = np.sum(inactive_days < window_length, axis=-1) / n_customer
        customer_retentions.append(customer_retention)
    return {"customer_retention": np.stack(customer_retentions)}


def compute_category_penetration(eval_trajectories: List[Trajectory], product_category_mapping: jnp.ndarray = None, **kwargs) -> Dict[str, np.ndarray]:
    """Compute average number of categories purchased per customer as the category penetration.

    Parameters
    ----------
        eval_trajectories (List[Trajectory]): a list of trajectories
        customer_idx (List[int], optional): a list of customer indices to calculate final customer retention. Defaults to None.
        product_category_mapping (jnp.ndarray, optional): a indicator matrix describing which category a product belongs to. Defaults to None.

    Returns
    -------
        Dict[str, np.ndarray]: a dictionary contains the category penetration
    """
    avg_category_purchases = []
    for traj in eval_trajectories:
        trx = traj.observation["previous_transaction"]
        n_customer, _, _ = trx.shape
        category_purchase = tf.linalg.matmul(trx, product_category_mapping).numpy().sum()
        avg_category_purchases.append(category_purchase / n_customer)
    return {"category_penetration": np.stack(avg_category_purchases)}


def compute_empirical_coupon(eval_trajectories: List[Trajectory], coupon_unit: float = 0.05, **kwargs) -> Dict[str, np.ndarray]:
    """Compute the empirical coupon from multiple trajectories.

    Parameters
    ----------
        eval_trajectories (List[Trajectory]): a list of trajectories

    Returns
    -------
        Dict[str, np.ndarray]: a dictionary contains the empirical coupon
    """
    empirical_discount = []
    for traj in eval_trajectories:
        empirical_discount.append(traj.action.numpy().mean())
    return {"empirical_discount": np.stack(empirical_discount) * coupon_unit}


def compute_final_training_loss(train_loss: List[List[float]]) -> Dict[str, np.ndarray]:
    """Compute the final training loss from multiple training losses.

    Parameters
    ----------
        train_loss (Dict[List[float]]): a dictionary contains the training loss for each agent

    Returns
    -------
        Dict[str, np.ndarray]: a dictionary contains the final training loss
    """
    return {"final_training_loss": np.array([loss[-1] for loss in train_loss])}


def collect_metric_data(eval_result: Dict[str, List[Trajectory]], customer_idx: List[int] = None, window_length: int = 3, product_category_mapping: jnp.ndarray = None, coupon_unit: float = 0.05, env_ids: Dict[str, List[Trajectory]] = {}, **kwargs) -> pd.DataFrame:
    """Collect metric data from evaluation results.

    Parameters
    ----------
        eval_result (Dict[str, List[Trajectory]]): a dictionary contains the evaluation trajectories for each agent
        config (Dict[str, any]): a dictionary contains the configuration for the evaluation
        customer_idx (List[int], optional): a list of customer indices to calculate final customer retention. Defaults to None.

    Returns
    -------
        pd.DataFrame: a dataframe contains the metric data for each agent
    """
    metric_data_container = []
    for agent_name, eval_trajectories in eval_result.items():
        metric_data = {}
        for func in [compute_accumulated_revenue, compute_accumulated_demand, compute_customer_retention, compute_category_penetration, compute_empirical_coupon]:  # type: ignore
            metric = func(eval_trajectories=eval_trajectories, customer_idx=customer_idx, window_length=window_length, product_category_mapping=product_category_mapping, coupon_unit=coupon_unit)  # type: ignore
            metric_data.update(metric)
        metric_data["agent"] = agent_name
        if agent_name in env_ids:
            metric_data["env_id"] = env_ids[agent_name]
        metric_data_container.append(pd.DataFrame(metric_data))

    return pd.concat(metric_data_container)


def generate_evaluation_report(
    eval_result: Dict[str, List[Trajectory]],
    window_length: int,
    product_category_mapping: jnp.ndarray,
    coupon_unit: float,
    n_arm: int,
    env_ids: Dict[str, List[Trajectory]] = {},
):
    # integrate the evaluation results
    revenue_df = get_accumulated_revenues_for_multiple_policies(eval_result)
    selected_arm_df = get_arm_propensities_at_time_step_for_multiple_policies(eval_result, n_arm=n_arm)
    metric_df = collect_metric_data(eval_result, window_length=window_length, product_category_mapping=product_category_mapping, coupon_unit=coupon_unit, env_ids=env_ids)
    return {"revenue_df": revenue_df, "selected_arm_df": selected_arm_df, "metric_df": metric_df}


def generate_segmented_evaluation_report(
    eval_results: defaultdict[str, Dict[str, List[Trajectory]]],
    window_length: int,
    product_category_mapping: jnp.ndarray,
    coupon_unit: float,
    n_arm: int,
    env_ids: Dict[str, List[Trajectory]] = {},  # type: ignore
):
    segmented_summary_dfs = defaultdict(list)
    for seg_name, eval_result in eval_results.items():
        summary_df = generate_evaluation_report(eval_result, window_length=window_length, product_category_mapping=product_category_mapping, n_arm=n_arm, coupon_unit=coupon_unit, env_ids=env_ids)
        for df_name, df in summary_df.items():
            df["segment"] = seg_name
            segmented_summary_dfs[df_name].append(df)
    segmented_summary_dfs = {df_name: pd.concat(dfs) for df_name, dfs in segmented_summary_dfs.items()}
    return segmented_summary_dfs


def make_mi_scores(feature_df: pd.DataFrame, frac: float = 0.2):
    """
    Compute mutual information score for the offline data trajectories.

    Parameters
    ----------
        feature_df (pd.DataFrame): offline data trajectories summarized in a data frame
        frac (float): fraction of trajectories to sample

    Returns
    -------
        mi_scores (np.ndarray): mutual information score array
    """
    sampled_df = feature_df.sample(frac=frac)
    y = sampled_df["reward"]
    X = sampled_df.drop(columns=["reward", "action", "segment", "env_id", "timestep"])
    discrete_features = X.dtypes == int
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

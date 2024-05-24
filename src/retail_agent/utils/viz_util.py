import matplotlib.pyplot as plt
import seaborn as sns


def plot_reward_estimation_for_arms(df, ax=None):
    assert "arm" in df.columns
    assert "reward" in df.columns

    if ax is None:
        ax = plt.gca()
    sns.pointplot(data=df, x="arm", y="reward", errorbar=("ci", 95), linestyle="none", ax=ax)
    ax.set_title("Mean reward estimation for each coupon level")
    ax.set_xlabel("Arm: coupon level")
    ax.set_ylabel("Mean reward")


def plot_bernts_purchase_prob_for_arms(df, ax=None):
    assert "arm" in df.columns
    assert "prob" in df.columns
    assert "prob_se" in df.columns

    if ax is None:
        ax = plt.gca()
    plt.errorbar(df["arm"], df["prob"], yerr=df["prob_se"], fmt="o")
    ax.set_xlabel("Arm: coupon level")
    ax.set_ylabel("Purchase event probability")
    ax.set_title("Purchase event probability for each arm")


def plot_cumulative_revenue_trajectory(df, ax=None, hue="agent", show_legend=True, **kwargs):
    assert "agent" in df.columns
    assert "revenue" in df.columns
    assert "timestep" in df.columns

    if ax is None:
        ax = plt.gca()

    sns.lineplot(x="timestep", y="revenue", marker="o", hue=hue, errorbar=("ci", 95), data=df, ax=ax, legend=show_legend, **kwargs)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Revenue")
    ax.set_title("Cumulative revenue trajectory by agent")


def plot_arm_purchase_prob_for_arms(df, ax=None, hue="agent", order=None, **kwargs):
    assert "arm" in df.columns
    assert "prob" in df.columns
    assert "agent" in df.columns

    if ax is None:
        ax = plt.gca()

    sns.lineplot(data=df, x="arm", y="prob", marker="o", ax=ax, hue=hue, order=order, errorbar=("ci", 95), **kwargs)
    ax.set_title("Probability of choosing each arm")
    ax.set_xlabel("Arm")
    ax.set_ylabel("Probability")
    ax.set_axes_locator
    ax.set_xticks(df["arm"].unique())


def plot_stacked_arm_purchase_prob_for_arms(df, ax=None, hue="agent", order=None, colormap="tab20", **kwargs):
    assert "arm" in df.columns
    assert "prob" in df.columns
    assert hue in df.columns

    if ax is None:
        ax = plt.gca()

    df = df.pivot(index=hue, columns="arm", values="prob")
    order = order or df.index
    df = df.reindex(order)
    df.plot(kind="bar", stacked=True, ax=ax, legend=False, colormap=colormap, **kwargs)
    ax.set_title("Probability of choosing each arm")
    ax.set_xlabel(hue)
    ax.set_ylabel("Probability")


def plot_metric_in_barplot(df, ax=None, hue=None, x="agent", y="accumulated_revenue", order=None, y_label="Accumulated revenue", **kwargs):
    assert x in df.columns
    assert y in df.columns

    if ax is None:
        ax = plt.gca()
    sns.barplot(data=df, x=x, y=y, ax=ax, hue=hue, errorbar=("ci", 95), order=order, **kwargs)
    ax.set_title(f"{y_label} at the final time step")
    ax.set_ylabel(y_label)


def gather_all_customer_report(summary_data: dict, figsize=(25, 4), ylim_min=[None, None, None, None, None, None], ylim_max=[None, None, None, None, None, None], palette="tab20", order=None):
    fig, axes = plt.subplots(1, 6, figsize=figsize)
    plot_metric_in_barplot(summary_data["metric_df"], ax=axes[0], hue="agent", order=order, y="accumulated_revenue", y_label="Accumulated revenue")
    plot_metric_in_barplot(summary_data["metric_df"], axes[1], hue="agent", order=order, y="customer_retention", y_label="Customer retention")
    plot_metric_in_barplot(summary_data["metric_df"], axes[2], hue="agent", order=order, y="category_penetration", y_label="Category penetration")
    plot_metric_in_barplot(summary_data["metric_df"], axes[3], hue="agent", order=order, y="empirical_discount", y_label="Empirical discount")
    plot_cumulative_revenue_trajectory(summary_data["revenue_df"], axes[4])
    plot_stacked_arm_purchase_prob_for_arms(summary_data["selected_arm_df"], axes[5], colormap=palette, order=order)
    axes[4].legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    for i, ax in enumerate(axes):
        ax.set_ylim(ylim_min[i], ylim_max[i])
        ax.tick_params(axis="x", labelrotation=90)
    plt.tight_layout()

    return fig, axes

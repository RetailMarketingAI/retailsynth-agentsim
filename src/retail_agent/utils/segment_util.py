from sklearn.cluster import KMeans
from tf_agents.trajectories import Trajectory
from typing import List, Dict, Any
import jax.numpy as jnp
import numpy as np


def segment_customer_by_price_sensitivity(
    synthesizer,
    n_segment: int,
    segment_name: List[Any] = None,
):
    """Segment customers according their price sensitivity.

    Parameters
    ----------
        synthesizer (DataSynthesizer)
        n_segment (int, optional): number of segments to divide. Defaults to 4.
        segment_name (list, optional): list of names for segments. Defaults to None.

    Returns
    -------
        pd.DataFrame: customer info data frame with columns of customer_key and segment
    """
    utility_beta_ui_w = synthesizer.utility_beta_ui_w.mean(axis=-1)
    sorted_indices = jnp.argsort(utility_beta_ui_w)
    segment_size = len(sorted_indices) // n_segment
    segment_indices = [sorted_indices[i * segment_size : (i + 1) * segment_size] for i in range(n_segment)]

    if segment_name is None:
        segment_name = [str(i) for i in range(n_segment)]
    else:
        assert len(segment_name) == n_segment, "segment_name must have the same length as n_segment"
    customers = dict(zip(segment_name, segment_indices))
    return customers


def segment_customer_by_context(
    trajectories: List[Trajectory],
    n_segment: int,
    segment_name: List[str] = None,
) -> Dict[str, List[int]]:
    """Segment customers according to their context observed in the trajectories.

    Parameters
    ----------
        trajectories (List[Trajectory]): a list of trajectories
        n_segment (int): number of segments to divide
        segment_name (List[str], optional): segment names. Defaults to None.

    Returns
    -------
        Dict[str, List[int]]: a dictionary containing the segment name and the customer indices in the segment
    """
    if segment_name is None:
        segment_name = [str(i) for i in range(n_segment)]
    else:
        assert len(segment_name) == n_segment, "segment_name must have the same length as n_segment"

    context = [trajectory.observation["customer_features"] for trajectory in trajectories]
    context = np.nan_to_num(np.array(context).mean(axis=(0, 2)))  # compute the mean across trajectories and time steps
    km = KMeans(n_clusters=n_segment).fit(context)
    customer_seg = km.labels_
    customer_seg_idx = {segment_name[seg]: np.where(customer_seg == seg)[0].tolist() for seg in range(n_segment)}

    return customer_seg_idx

import numpy as np
import jax.numpy as jnp
import logging
from collections import defaultdict

from retailsynth.synthesizer.data_synthesizer import DataSynthesizer
from retailsynth.synthesizer.config import SyntheticParameters


class TargetedCouponDataSynthesizer(DataSynthesizer):
    """Reconstruct the synthesizer for the use case in the environment. Generates observations including product price, marketing features, and product-specific or customer-specific information. Incorporates coupons as actions."""

    def __init__(self, cfg_raw_data: SyntheticParameters):
        """Initialize the synthesizer with the configuration.

        Parameters
        ----------
            cfg_raw_data (SyntheticParameters): configuration for the synthetic data.
        """
        super().__init__(cfg_raw_data)

    def reset(self):
        """Reset the synthesizer to the initial state."""
        # reset the attributes of the synthesizer
        self.choice_decision_stats = defaultdict(list)
        self.elasticity_stats = defaultdict(list)
        self.category_utility_cache = self.category_utility_cache.at[:].set(0)
        self.discount_state = self.discount_state.at[:].set(0)

    def sample_observed_customer_product_feature(self):
        # This function generates the context feature in the product utility equation.
        # We take it out as a separate function mainly to expose the feature itself observed to the environment.
        x_dist = self._initialize_feature_distribution_standard_normal((self.n_customer, self.n_product), "x")
        x = self._sample_feature(x_dist)
        return x

    def get_price_info_for_next_step(self, **kwargs) -> tuple:
        """Prepare the product price and marketing features for next time step.

        Returns
        -------
            tuple: (product_price, marketing_feature, other observed_customer_product_feature).
        """
        discount = self.sample_discount()
        product_price = self.compute_product_price(discount)
        coupon = kwargs.get("coupon", jnp.zeros((self.n_customer,)))
        marketing_feature = self.sample_marketing_feature(discount=discount, coupon=coupon)
        observed_customer_product_feature = self.sample_observed_customer_product_feature()

        return product_price, marketing_feature, observed_customer_product_feature

    def compute_product_utility(self, price: jnp.ndarray, observed_feature: jnp.ndarray) -> jnp.ndarray:
        """Compute product utility.

        $$
        \mu^{prod}_{uit} &= \mathbf{\beta_{ui}^x} \mathbf{X_{uit}} + \beta_{ui}^{z} Z_{i} + \beta_{ui}^w log(P_{it}) + \epsilon_{uit}
        $$

        Parameters
        ----------
            price (jnp.ndarray): product price

        Returns
        -------
            jnp.ndarray: product utility in shape of (n_customer, n_product)
        """
        error = self._sample_error((self.n_customer, self.n_product), self.utility_error_distribution)
        assert observed_feature.shape == error.shape
        assert self.product_endogenous_feature.shape[-1] == observed_feature.shape[-1]

        product_utility = self.utility_beta_ui_x * observed_feature + self.utility_beta_ui_w * jnp.log(price) + self.utility_beta_ui_z * jnp.expand_dims(self.product_endogenous_feature, axis=0) + error
        # An extremely large product utility can lead to unreasonably large demand in the following step
        # To avoid this, we clip the product utility with an upper qua
        upper_bound = jnp.percentile(product_utility, self.utility_clip_percentile)
        return jnp.clip(product_utility, a_max=upper_bound)

    def sample_transaction_one_step(
        self,
        prev_store_visit: jnp.ndarray,
        prev_store_visit_prob: jnp.ndarray,
        product_price: jnp.ndarray,
        marketing_feature: jnp.ndarray,
        observed_customer_product_feature: jnp.ndarray,
        coupon: jnp.ndarray = None,
        compute_store_prob: bool = False,
    ):
        """Generate synthetic trajectories for one time step given the hidden state and the action (coupon).

        Execution order adjusted to following to allow for coupons as an action:
        1. Compute product price with coupon applied
        2. Compute the joint decision on purchase quantity
        3. Compute the revenue

        Parameters
        ----------
            prev_store_visit (jnp.ndarray): indicator of store visit in the previous step.
            prev_store_visit_prob (jnp.ndarray): probability of store visit in the previous step.
            product_price (jnp.ndarray): product unit price for the current step.
            marketing_feature (jnp.ndarray): marketing feature for the current step.
            observed_customer_product_feature (jnp.ndarray): observed customer product feature for the current step, used in the product utility function.
            coupon (jnp.ndarray): coupon that the customer redeemed at the current time step. Default to None.
            compute_store_prob (bool): flag to tell whether to compute the store visit probability recursively based on the previous store visit prob.

        Returns
        -------
            dict: dict of jnp.ndarray
        """
        product_price_with_coupon = self.compute_product_price_with_coupon(product_price, coupon)
        product_utility = self.compute_product_utility(product_price_with_coupon, observed_customer_product_feature)

        category_utility = self.compute_category_utility(product_utility)
        logging.debug(f"Avg. category utility: {category_utility.mean():.4f}")
        product_choice_prob = self.compute_product_purchase_conditional_probability(product_utility)
        logging.debug(f"Avg. product choice probability: {product_choice_prob.mean():.4f}")

        category_choice_prob = self.compute_category_purchase_conditional_probability(category_utility)
        logging.debug(f"Avg. category choice probability: {category_choice_prob.mean():.4f}")
        if compute_store_prob:
            assert prev_store_visit_prob is not None
            assert prev_store_visit is not None
            store_visit_prob = self.compute_store_visit_probability(
                self.category_utility_cache,
                prev_store_visit_prob,
                prev_store_visit,
                marketing_feature,
            )
        else:
            store_visit_prob = prev_store_visit_prob

        product_demand_mean = self.compute_product_demand_mean(product_utility)
        (
            store_visit,
            category_choice,
            product_choice,
            product_demand,
        ) = self._sample_decisions(
            store_visit_prob,
            category_choice_prob,
            product_choice_prob,
            product_demand_mean,
        )
        quantity_purchased = self.compute_joint_decision(store_visit, category_choice, product_choice, product_demand).astype(np.int32)
        coupon_redemption_indicator = (product_price > product_price_with_coupon) * (quantity_purchased > 0)

        self.category_utility_cache = category_utility

        return {
            "quantity": quantity_purchased,
            "store_visit": store_visit,
            "store_visit_prob": store_visit_prob,
            "product_price_with_coupon": product_price_with_coupon,
            "coupon_redemption_indicator": coupon_redemption_indicator,
        }

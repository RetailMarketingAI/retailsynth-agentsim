from typing import Dict

import numpy as np
from tf_agents.specs import array_spec

from retail_agent.envs.retail_synthesizer import TargetedCouponDataSynthesizer
from retail_agent.envs.retail_env import RetailEnv
from retail_agent.envs.features import AveragePurchaseQuantity, AveragePurchaseProbability, AveragePurchasePrice, AverageRedeemedCoupon, AveragePurchaseDiscount

integer_dtype = np.int32
float_dtype = np.float32


class CBRetailEnv(RetailEnv):
    def __init__(self, synthesizer: TargetedCouponDataSynthesizer, action_spec_mode: str = "store-wide", max_steps: int = 53, discount_factor: float = 0, coupon_unit: float = 0.05, n_coupon_levels: int = 5, name: str = "cbretailenv"):
        """Formulate the environment to provide customer features as observations for a contextual bandit which has different coupon levels as the arms.

        Attributes
        ----------
            synthesizer (TargetedCouponDataSynthesizer): key object to generate the synthetic data.
            action_spec_mode (str): type of the action spec. Support "store-wide" and "category". Default to be "category".
            max_steps (int): maximum number of steps in one trajectory
            discount_factor (float, optional): discount factor when accumulating the future revenue. Defaults to 0.
            coupon_unit (float, optional): the unit of coupon value. Defaults to 0.05.
            n_coupon_levels (int, optional): number of coupon levels. Defaults to 5.
            name (str, optional): environment name
        """
        super().__init__(synthesizer, action_spec_mode, max_steps, discount_factor, coupon_unit, n_coupon_levels, name)
        if action_spec_mode != "store-wide":
            raise NotImplementedError("Category action spec is not supported for contextual bandit env.")

        n_customers = self.synthesizer.n_customer

        self._features: Dict = {
            "avg_purchase_quantity": AveragePurchaseQuantity(n_customers),
            "avg_purchase_probability": AveragePurchaseProbability(n_customers),
            "avg_purchase_price": AveragePurchasePrice(n_customers),
            "avg_redeemed_discount": AverageRedeemedCoupon(n_customers),
            "avg_purchase_discount": AveragePurchaseDiscount(n_customers, self.synthesizer.product_price),
        }

        for feature in self._features.keys():
            self._observation_spec[feature] = array_spec.BoundedArraySpec(
                shape=(),
                dtype=float_dtype,
                name=feature,
            )

    def _prepare_observation(self, action=None):
        # override the function from the supclass and add logic to compute context features
        product_price = np.asarray(self.product_price, dtype=float_dtype)
        if action is not None:
            features = [
                self._features[feature].update(
                    np.asanyarray(self._transaction, dtype=float_dtype),
                    product_price=product_price,
                    action=np.asarray(action, dtype=float_dtype),
                    coupon_redemption_indicator=np.asanyarray(self.coupon_redemption_indicator, dtype=float_dtype),
                )
                for feature in self._features
            ]
        else:
            features = [np.zeros(self.synthesizer.n_customer) for _ in self._features]

        obs = super()._prepare_observation()

        for feature, value in zip(self._features.keys(), features):
            obs[feature] = np.nan_to_num(np.asarray(value, dtype=float_dtype), copy=False)

        return obs

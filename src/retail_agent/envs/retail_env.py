import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from retail_agent.envs.retail_synthesizer import TargetedCouponDataSynthesizer

integer_dtype = np.int32
float_dtype = np.float32


class RetailEnv(py_environment.PyEnvironment):
    def __init__(
        self,
        synthesizer: TargetedCouponDataSynthesizer,
        action_spec_mode: str = "store-wide",
        max_steps: int = 53,
        discount_factor: float = 0,
        coupon_unit: float = 0.05,
        n_coupon_levels: int = 5,
        name: str = "retailenv"
    ):
        """Initialize the environment.

        Attributes
        ----------
            synthesizer (TargetedCouponDataSynthesizer): key object to generate the synthetic data.
            action_spec_mode (str): type of the action spec. Support "store-wide" and "category". Default to be "store-wide".
            max_steps (int): maximum number of steps in one trajectory
            discount_factor (float, optional): discount factor when accumulating the future revenue. Defaults to 0.
            name (str, optinal): environment name
        """
        # initialize the synthesizer
        self.name = name
        self.synthesizer = synthesizer
        self.store_visit_prob = self.synthesizer.initial_store_visit_prob.copy()
        self.store_visit = self.synthesizer.initial_store_visit.copy()
        (
            self.product_price,
            self.marketing_feature,
            self.observed_customer_product_feature,
        ) = self.synthesizer.get_price_info_for_next_step()
        # set up the environment variables
        # 1. action space
        self.action_spec_mode = action_spec_mode
        if action_spec_mode == "store-wide":
            action_spec_shape = ()
        elif action_spec_mode == "category":
            raise NotImplementedError("Category action spec is not supported yet.")
        else:
            raise ValueError(f"Unrecognized action spec mode: {action_spec_mode}")
        self._action_spec = array_spec.BoundedArraySpec(
            shape=action_spec_shape,
            dtype=integer_dtype,
            minimum=0,
            maximum=n_coupon_levels - 1,
            name="action",
        )
        self.coupon_unit = coupon_unit
        self.n_coupon_levels = n_coupon_levels
        # 2. observation spec
        self._observation_spec = {
            "previous_transaction": array_spec.BoundedArraySpec(
                shape=(self.synthesizer.n_product,),
                dtype=integer_dtype,
                minimum=0,
                name="previous_transaction",
            ),
            "product_price": array_spec.BoundedArraySpec(
                shape=(self.synthesizer.n_product,),
                dtype=float_dtype,
                minimum=0,
                name="product_price",
            ),
            "marketing_feature": array_spec.BoundedArraySpec(
                shape=(),
                dtype=float_dtype,
                minimum=0,
                name="marketing_feature",
            ),
            "observed_customer_product_feature": array_spec.BoundedArraySpec(
                shape=(self.synthesizer.n_product,),
                dtype=float_dtype,
                name="observed_customer_product_feature",
            ),
        }

        # 3. internal record to track the generated transaction
        self._transaction = np.zeros(
            (self.synthesizer.n_customer, self.synthesizer.n_product),
            dtype=integer_dtype,
        )
        self._discount_factor = discount_factor
        self._episode_ended = False
        self._current_step = 0
        self._max_steps = max_steps
        self._batch_size = self.synthesizer.n_customer

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batched(self):
        return True

    def action_spec(self):
        """Return the action spec."""
        return self._action_spec

    def observation_spec(self):
        """Return the observation spec."""
        return self._observation_spec

    def reward_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(),
            dtype=float_dtype,
            minimum=0,
            name="reward",
        )

    def _prepare_observation(self, action=None):
        """Environment for evaluating RL agent that targets coupons to retail customers.

        Returns
        -------
            dict: a dictionary contains array for previous transaction, product price and marketing feature.
        """
        if self.marketing_feature.shape == ():
            marketing_feature = np.tile(self.marketing_feature, (self.batch_size,)).astype(float_dtype)
        elif self.marketing_feature.shape == (self.batch_size,):
            marketing_feature = np.asarray(self.marketing_feature, dtype=float_dtype)
        return {
            "previous_transaction": self._transaction,
            "product_price": np.tile(self.product_price, (self.batch_size, 1)).astype(float_dtype),
            "marketing_feature": marketing_feature,
            "observed_customer_product_feature": np.asarray(self.observed_customer_product_feature, dtype=float_dtype),
        }

    def _reset(self):
        """Reset the environment to a clean slate.

        Returns
        -------
        ts.restart
            with initial state witch is observation of all zero (no purchase for all products)
        """
        self.synthesizer.reset()
        # reset the variables of the py environment
        self.store_visit_prob = self.store_visit_prob.at[:].set(1)
        self.store_visit = self.store_visit.at[:].set(1)
        self._transaction = np.zeros(
            (self.synthesizer.n_customer, self.synthesizer.n_product),
            dtype=integer_dtype,
        )
        self._episode_ended = False
        self._current_step = 0
        (
            self.product_price,
            self.marketing_feature,
            self.observed_customer_product_feature,
        ) = self.synthesizer.get_price_info_for_next_step()
        return ts.TimeStep(
            observation=self._prepare_observation(),
            reward=np.zeros((self._batch_size,), dtype=float_dtype),
            discount=(np.ones((self._batch_size,)) * self._discount_factor).astype(np.float32),
            step_type=np.zeros((self._batch_size,), dtype=integer_dtype),  # representing the first step
        )

    def _step(self, action: np.ndarray):
        """Take one step in the environment.

            sequence of activity taken at each call of this method:

            0. check if the simulation is ended or not
            1. record action as recommended coupon into price_action
            2. get observation by calling the synthesizer
            3. record observation
            4. calculate the reward
            5. sample price and marketing feature for the next step
            6. pass the information as tf trajectory

        Parameters
        ----------
        action : array_spec.BoundedArraySpec
            the discount propsed by the agent with shape of (self.n_segment, self.n_product)

        Returns
        -------
        ts.transition
            a ts.transition instance with state, reward and RL discount
        """
        if self._episode_ended:
            return self.reset()

        action = action * self.coupon_unit
        synth_step_output = self.synthesizer.sample_transaction_one_step(
            self.store_visit,
            self.store_visit_prob,
            self.product_price,
            self.marketing_feature,
            self.observed_customer_product_feature,
            action,
            compute_store_prob=(self._current_step > 0),
        )
        self._transaction = np.asarray(synth_step_output["quantity"])
        self.store_visit = synth_step_output["store_visit"]
        self.store_visit_prob = synth_step_output["store_visit_prob"]
        self._current_step += 1
        revenue = (self._transaction * synth_step_output["product_price_with_coupon"]).sum(axis=1)
        self.coupon_redemption_indicator = synth_step_output["coupon_redemption_indicator"]
        # prepare info needed for next step
        (
            self.product_price,
            self.marketing_feature,
            self.observed_customer_product_feature,
        ) = self.synthesizer.get_price_info_for_next_step(coupon=action)

        if self._current_step >= self._max_steps:
            self._episode_ended = True
            step_type = np.ones((self._batch_size,), dtype=integer_dtype) * 2  # representing the last step
        else:
            step_type = np.ones((self._batch_size,), dtype=integer_dtype)  # representing the mid step
        return ts.TimeStep(
            observation=self._prepare_observation(action=action),
            reward=np.asarray(revenue, dtype=float_dtype),
            discount=(np.ones((self._batch_size,)) * self._discount_factor).astype(np.float32),
            step_type=step_type,
        )

"""Classes that collect and update features of the customer's purchase behavior.

Each feature is a scalar and is updated at each time step. The features are used as observations for the contextual bandit environment. 
The update method returns a feature array that is a vector with shape=(n_customers,). 
"""

import numpy as np


class AveragePurchaseQuantity:
    """Collect the average purchase quantity of a customer's positive purchase events."""

    def __init__(self, n_customers):
        self.purchase_event_counts = np.zeros(n_customers)
        self.avg_purchase_qty = np.zeros(n_customers)

    def update(self, transaction, **kwargs):
        # get flag of purchase event
        qty_items_purchased = transaction.sum(axis=1)
        purchase_event = qty_items_purchased > 0

        # Sum the total number of purchases per customer
        self.avg_purchase_qty[purchase_event] = self.purchase_event_counts[purchase_event] * self.avg_purchase_qty[purchase_event] + qty_items_purchased[purchase_event]

        # Update the purchase counts
        self.purchase_event_counts[purchase_event] = self.purchase_event_counts[purchase_event] + 1

        # Normalize the average purchase quantity
        self.avg_purchase_qty[purchase_event] = self.avg_purchase_qty[purchase_event] / self.purchase_event_counts[purchase_event]
        return self.avg_purchase_qty


class AveragePurchaseProbability:
    """Collect the probability of a customer making a purchase in a given timestep."""

    def __init__(self, n_customers):
        self.purchase_event_counts = np.zeros(n_customers)
        self.n_steps = 0

    def update(self, transaction, **kwargs):
        # get flag of purchase event
        qty_items_purchased = transaction.sum(axis=1)
        purchase_event = qty_items_purchased > 0
        self.purchase_event_counts += purchase_event
        self.n_steps += 1
        return self.purchase_event_counts / self.n_steps


class AverageRedeemedCoupon:
    """Collect average coupon of purchased events."""

    def __init__(self, n_customers):
        self.n_redemptions = np.zeros(n_customers)
        self.cumulative_disc_amt = np.zeros(n_customers)
        self.n_customers = n_customers

    def update(self, transaction, action=None, coupon_redemption_indicator=None, **kwargs):
        assert action.shape == (len(transaction),)
        assert coupon_redemption_indicator.shape == transaction.shape

        total_redemptions = coupon_redemption_indicator.sum(axis=1)
        self.n_redemptions += total_redemptions
        self.cumulative_disc_amt += total_redemptions * action

        ever_redeemed = self.n_redemptions > 0
        cum_avg_redeemed_discount = np.zeros(self.n_customers)
        cum_avg_redeemed_discount[ever_redeemed] = self.cumulative_disc_amt[ever_redeemed] / self.n_redemptions[ever_redeemed]

        if sum(ever_redeemed) > 0:
            # Fill in customers who have not redeemed a coupon with the mean of the customers who have
            cum_avg_redeemed_discount[~ever_redeemed] = (cum_avg_redeemed_discount[ever_redeemed]).mean()

        return cum_avg_redeemed_discount


class AveragePurchasePrice:
    """Collect average price of purchased items before coupon applied."""

    def __init__(self, n_customers):
        self.n_purchases = np.zeros(n_customers)
        self.cumulative_purchase_amt = np.zeros(n_customers)
        self.n_customers = n_customers

    def update(self, transaction, product_price=None, **kwargs):
        # Find prices paid this step
        purchase_mask = transaction > 0
        prices_paid = purchase_mask * product_price[np.newaxis,]

        # Count the number of purchases and sum the avg prices paid
        self.n_purchases += purchase_mask.sum(axis=1)
        self.cumulative_purchase_amt += prices_paid.sum(axis=1)
        cum_avg_price_paid = np.zeros(self.n_customers)

        ever_purchased = self.n_purchases > 0
        cum_avg_price_paid[ever_purchased] = self.cumulative_purchase_amt[ever_purchased] / self.n_purchases[ever_purchased]

        # Fill in customers who have not made a purchase with the mean of the customers who have made a purchase
        if sum(ever_purchased) > 0:
            cum_avg_price_paid[~ever_purchased] = (cum_avg_price_paid[ever_purchased]).mean()
        return cum_avg_price_paid


class AveragePurchaseDiscount:
    """Collect average discount of purchased items before coupon applied."""

    # discount from the pricing policy
    def __init__(self, n_customers, product_base_price):
        self.n_purchases = np.zeros(n_customers)
        self.cumulative_purchase_discount = np.zeros(n_customers)
        self.n_customers = n_customers
        self.product_base_price = product_base_price

    def update(self, transaction, product_price=None, **kwargs):
        # Find discount used this step
        purchase_mask = transaction > 0
        discount = 1 - product_price / self.product_base_price
        discount_applied = purchase_mask * discount[np.newaxis,]

        # Count the number of purchases and sum the avg prices paid
        self.n_purchases += purchase_mask.sum(axis=1)
        self.cumulative_purchase_discount += discount_applied.sum(axis=1)
        cum_avg_discount_applied = np.zeros(self.n_customers)

        ever_purchased = self.n_purchases > 0
        cum_avg_discount_applied[ever_purchased] = self.cumulative_purchase_discount[ever_purchased] / self.n_purchases[ever_purchased]

        # Fill in customers who have not made a purchase with the mean of the customers who have made a purchase
        cum_avg_discount_applied[~ever_purchased] = (cum_avg_discount_applied[ever_purchased]).mean()

        return cum_avg_discount_applied

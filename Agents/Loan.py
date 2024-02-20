import uuid

import numpy as np


class LoanObj:
    """
    Represents a loan in the simulation.

    Attributes:
        id (str): Unique identifier for the loan.
        maturity (int): Number of cycles until the loan matures.
        current_cycle (int): The current cycle of the simulation.
        starting_cycle (int): The cycle when the loan was initiated.
        ending_cycle (int): The cycle when the loan will end.
        time_to_maturity (int): Number of cycles until the loan matures.
        pd (float): Probability of default for the loan.
        size (float): Size of the loan.
        interest_rate (float): Interest rate of the loan.
        fair_value (float): Fair value of the loan.
        market_price (float): Market price of the loan.
        current_owner (str): The current owner of the loan.
        maturity_bool (bool): Indicates if the loan has matured.
        fair_value_history (list): History of the loan's fair values.
        market_price_history (list): History of the loan's market prices.
        ownership_history (list): History of the loan's ownership.
    """

    __slots__ = ['id', 'maturity', 'current_cycle', 'starting_cycle', 'ending_cycle', 'time_to_maturity', 'pd', 'size', 'base_interest_rate',
                 'interest_rate', 'fair_value', 'market_price', 'current_owner', 'maturity_bool', 'fair_value_history',
                 'market_price_history', 'ownership_history', 'sale_price_history', 'reserve_price']

    def __init__(self, current_cycle=0, current_owner="no owner", reserve_price=0.80, float_interest=0):
        """
        Initializes the Loan with random values for maturity, pd, size, interest rate, and fair value.
        Sets the starting cycle, calculates the ending cycle and time to maturity based on maturity.
        Initializes histories and sets the current owner.
        """

        self.id = str(uuid.uuid4())
        self.maturity = np.random.randint(3, 8) * 12
        self.current_cycle = current_cycle
        self.starting_cycle = self.current_cycle
        self.ending_cycle = self.starting_cycle + self.maturity
        self.time_to_maturity = self.ending_cycle - self.current_cycle
        self.pd = (np.random.beta(2, 100))
        self.size = np.random.uniform(500_000, 5_000_000)
        self.base_interest_rate = self.generate_interest_rate()
        self.interest_rate = self.base_interest_rate + float_interest
        self.fair_value = self.calculate_price()
        self.market_price = self.fair_value
        self.reserve_price = self.fair_value * reserve_price
        self.current_owner = current_owner
        self.maturity_bool = False

        # tracking attributes
        self.fair_value_history = [self.fair_value]
        self.market_price_history = [self.market_price]
        self.sale_price_history = [None]
        self.ownership_history = [current_owner]

    def generate_interest_rate(self):
        """
        Generates a random interest rate based on some factors and noise.
        The interest rate is influenced by the loan's probability of default.
        """

        base_level_noise = np.random.normal(0.01, 0.005)
        correlation_factor = 0.8
        influence_factor = np.random.uniform(0.05, 0.15)
        correlated_component = correlation_factor * self.pd * influence_factor
        return base_level_noise + correlated_component

    def calculate_price(self):
        """
        Calculates the price of the loan based on its attributes.
        The price is influenced by the probability of default, interest rate, and size.
        """

        # Effect of PD on price with additional penalty for PD > 0.2
        beta1 = np.random.normal(-8, 1)
        pd_effect = beta1 * self.pd
        if self.pd > 0.2:
            pd_effect *= 3  # Double the negative effect for PD > 0.2

        # Stronger Effect of interest rate on price
        beta2 = np.random.normal(5, 1)
        ir_effect = beta2 * self.interest_rate

        # Effect of size on price
        beta3 = np.random.uniform(-0.00000002, 0)  # Smaller coefficient
        size_effect = beta3 * self.size

        # Intermediate price
        intermediate_price = np.random.normal(100, 0.5) + pd_effect + ir_effect + size_effect

        # Maturity effect to bring price closer to par value for lower maturities
        k = 0.5  # decay constant
        maturity_effect = (intermediate_price - 100) * np.exp(-k * self.time_to_maturity)

        final_price = intermediate_price - maturity_effect

        return final_price

    @staticmethod
    def generate():
        return LoanObj()

    def update_owner(self, new_owner):
        self.current_owner = new_owner
        self.ownership_history.append(new_owner.id)

    def update(self, current_cycle, float_interest = 0, new_owner=None, new_market_price=None):
        """
        Updates the loan's attributes for a new cycle.
        If the loan has matured, checks for default and adjusts fair value and market price accordingly.
        Records the changes in the histories.

        Args:
        current_cycle (int): The new current cycle.
        new_owner (str, optional): The new owner of the loan, if it has changed.
        new_market_price (float, optional): The new market price of the loan, if it has changed.
        """

        if current_cycle > self.ending_cycle:
            return

        if new_owner:
            self.update_owner(new_owner)

        self.current_cycle = current_cycle
        self.time_to_maturity = self.ending_cycle - self.current_cycle
        self.interest_rate = self.base_interest_rate + float_interest
        self.fair_value = self.calculate_price()

        if new_market_price:
            self.market_price = new_market_price

        self.market_price_history.append(self.market_price)

        # Loan maturity logic
        if current_cycle == self.ending_cycle:
            self.maturity_bool = True
            default_outcome = np.random.rand() < self.pd
            if default_outcome:
                self.fair_value = np.random.normal(60, 10)
                self.market_price = self.fair_value  # Set to liquidation value
            else:
                self.fair_value = 100  # Set to par value
                self.market_price = 100  # Set to par value
        self.fair_value_history.append(self.fair_value)

    def as_dict(self):
        return {slot: getattr(self, slot, None) for slot in self.__slots__}


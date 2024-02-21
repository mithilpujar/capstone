import uuid

import numpy as np


class LoanInvestorObj:
    def __init__(self, trader=None, capital=None, min_capital=0.15, target_score_param = 0.616):
        """
        Initializes the LoanInvestor object with given or default parameters.

        Parameters:
        - trader (optional): An instance representing the trader for the investor.
        - capital (optional): Initial capital for the investor. If not provided, it's generated randomly.
        - min_capital (float, optional): Minimum capital that the investor should hold. Defaults to 0.15.
        """

        self.id = 'I' + str(uuid.uuid4())
        self.capital = capital if capital else self.generate_initial_capital()
        self.min_capital_pct = min_capital
        self.capital_history = []

        # target score should be negatively correlated with capital
        self.target_score = np.abs(np.random.normal(target_score_param, 0.1)) if self.capital < 10e8 else np.abs(np.random.normal(target_score_param*0.5, 0.05))
        self.current_score = 0
        self.current_cycle = 0

        self.loan_fair_values = []
        self.portfolio_values = []

        self.portfolio = []
        self.matured_loans = []
        self.trader = trader

        self.interest_received = []

        # adding in trading logic
        self.loans_for_sale = []

        # storing sold loans
        self.sold_loans = []
        self.purchased_loans = []

    @staticmethod
    def generate_initial_capital():
        """
        Generates a random initial capital for the investor using a Pareto distribution.

        Returns:
        float: Initial capital.
        """

        random_capital = np.round(np.random.pareto(2) * 50e6 + 50e6)
        return random_capital

    def initialize_portfolio(self, available_loans):

        """
        Initializes the investor's portfolio with available loans.
        :param available_loans: List of available loans already prepared by the loanMarket class.
        :param capital_threshold: Maximum proportion of capital to be invested.
        """

        np.random.shuffle(available_loans)
        total_investment = 0
        for loan in available_loans:
            purchase_value = (loan.market_price / 100) * loan.size
            if total_investment + purchase_value <= self.capital * (self.min_capital_pct + 0.05):
                total_investment += purchase_value
                loan.update_owner(self)
                self.portfolio.append(loan)
            else:
                break

        self.capital -= total_investment
        self.calculate_current_score()
        self.loan_fair_values.append(np.sum([((loan.fair_value/100) * loan.size) for loan in self.portfolio]))
        self.portfolio_values.append(self.loan_fair_values[-1] + self.capital)
        self.capital_history.append(self.capital)

    def tune_target_score(self, target_score_param):
        self.target_score = np.abs(np.random.normal(target_score_param, 0.1)) if self.capital < 10e8 else np.abs(np.random.normal(target_score_param*0.5, 0.05))

    def calculate_value(self, just_matured = []):
        self.loan_fair_values.append(np.sum([((loan.fair_value / 100) * loan.size) for loan in self.portfolio if not loan.maturity_bool]))

        # adding in the value of the matured loans
        self.capital += np.sum([((loan.fair_value / 100) * loan.size) for loan in just_matured])


    def calculate_current_score(self):
        weighted_interest = sum([loan.interest_rate * loan.size for loan in self.portfolio])
        weighted_pd = sum([loan.pd * loan.size for loan in self.portfolio])
        total_size = sum([loan.size for loan in self.portfolio])
        self.current_score = (weighted_interest / total_size) / (
                weighted_pd / total_size) if weighted_interest > 0 else 0

    def receive_interest(self):
        """
        This is the procedure every cycle for the investor to receive interest on their loans.
        :param float_interest: The floating base rate for interest (think SOFR)
        """

        total_interest = 0

        for loan in self.portfolio:
            entered = True
            if loan.maturity_bool:
                self.capital += ((loan.fair_value / 100) * loan.size)
                self.matured_loans.append(loan)
                self.portfolio.remove(loan)

            else:
                cycle_interest = (loan.interest_rate / 12) * loan.size
                total_interest += cycle_interest

        self.capital += total_interest
        self.interest_received.append(total_interest)

    def get_loan_to_sell(self, num_select=1, reserve_price = 0.8):
        # process to find the top loans that contribute to PD in the wrong direction in terms of absolute value
        # returns the top loan to sell which would be passed to their trader

        # start by checking if there are loans in the portfolio, if not return
        if len(self.portfolio) == 0:
            return

        # start by sorting the portfolio by PD
        sorted_portfolio = sorted(self.portfolio, key=lambda x: x.pd / x.interest_rate, reverse=True)

        # removing matured loans from the sorted portfolio
        sorted_portfolio = [loan for loan in sorted_portfolio if not loan.maturity_bool]

        if len(sorted_portfolio) == 0:
            return

        # finding out what the wrong direction is by comparing current pd to target pd
        wrong_direction = self.current_score > self.target_score

        # if the current score is less than the target score, then the wrong direction is positive, meaning we should sell the highest score loans
        # if the current score is greater than the target score, then the wrong direction is negative, meaning we should sell the lowest score loans
        if wrong_direction:
            top_wrong_direction = sorted_portfolio[:num_select]
        else:
            top_wrong_direction = sorted_portfolio[-num_select:]

        loan_to_sell = np.random.choice(top_wrong_direction)
        loan_to_sell.reserve_price = reserve_price*loan_to_sell.sale_price_history[-1] if loan_to_sell.sale_price_history[-1] is not None else reserve_price * loan_to_sell.market_price

        self.loans_for_sale.append(loan_to_sell)
        return loan_to_sell

    def get_bid_price(self, loan, pricing_method = 'portfolio_included'):

        # if the loan has matured, you can't bid on it
        if loan.maturity_bool:
            return 0

        # use portfolio neutral pricing for the bid price
        if pricing_method == 'portfolio_neutral':
            # takes a loan as an input and returns a bid price for the investor
            # breaks if the investor can't purchase the loan
            # start by calculating the total interest received from the loan
            proj_interest = loan.interest_rate / 12 * loan.time_to_maturity
            bid_price = proj_interest + 100 - ((loan.pd * loan.time_to_maturity) / self.target_score)

        # using a portfolio included pricing method to determine the bid price
        # uses projected interest if the new loan is included in the portfolio
        if pricing_method == 'portfolio_included':
            # creating a temporary portfolio to calculate the projected interest
            temp_portfolio = self.portfolio.copy()
            temp_portfolio.append(loan)

            # calculating the projected interest of the entire portfolio
            proj_interest = np.sum([(loan.interest_rate / 12 * loan.time_to_maturity * loan.size) for loan in temp_portfolio])
            portfolio_size = sum([loan.size for loan in temp_portfolio])
            proj_interest = proj_interest / portfolio_size
            bid_price = proj_interest + 100 - ((loan.pd * loan.time_to_maturity) / self.target_score)

        # ensuring the loan doesn't exceed the minimum capital for an investor
        # if it does, then the investor will only bid until their minimum capital threshold
        if (loan.size*(bid_price/100)) > self.capital:
            bid_price = (self.capital * (1-self.min_capital_pct)) / loan.size

        return bid_price

    def buy_loan(self, loan_to_purchase, broker_fee):
        # method for the investor to buy the loan after they have won an auction
        # updates the investor's portfolio and capital
        self.portfolio.append(loan_to_purchase)
        self.purchased_loans.append(loan_to_purchase.id)
        self.capital -= loan_to_purchase.size * (loan_to_purchase.sale_price_history[-1] / 100)
        self.capital -= broker_fee

        # updating the loan's owner
        loan_to_purchase.update_owner(self)

        # updating the investor's values
        self.calculate_value()

    def update(self, cycle=None):
        """
        Updates the investor's state for a given cycle.

        Parameters:
        - float_interest (float, optional): The floating base rate for interest. Defaults to 0.
        - cycle (int, optional): The current cycle. Not currently used.
        """

        self.current_cycle = cycle

        self.receive_interest()

        # removing matured loans from portfolio
        just_matured = [loan for loan in self.portfolio if loan.maturity_bool]
        self.matured_loans.extend(just_matured)
        self.portfolio = [loan for loan in self.portfolio if not loan.maturity_bool]

        self.calculate_value(just_matured)
        self.portfolio_values.append(self.loan_fair_values[-1] + self.capital)

        if self.capital_history[-1] < self.capital:
            self.capital_history.append(self.capital)


        self.calculate_current_score()





import Agents.Loan as Loan
import Agents.LoanInvestor as LoanInvestor
import Agents.LoanTrader as LoanTrader
import numpy as np
import streamlit as st

class loanMarket:
    def __init__(self):

        self.cycle = 0
        self.num_loans = st.slider("Number of loans", 0, 10000, 100)
        self.num_investors = st.slider("Number of investors", 0, 1000, 10)
        self.num_traders = st.slider("Number of traders", 0, 1000, 10)

        # creating the universe of loans
        self.loans = [Loan.LoanObj() for _ in range(self.num_loans)]

        # creating the universe of investors
        self.investors = [LoanInvestor.LoanInvestor() for _ in range(self.num_investors)]

        # creating the universe of traders
        self.traders = [LoanTrader.LoanTrader(max_investors=self.num_investors // self.num_traders) for _ in
                        range(self.num_traders)]

    def initialize(self):





    def update(self):
        for loan in self.loans:
            # loans are updated independent of investors
            loan.update(self.cycle + 1)
        for investor in self.investors:
            # investors are updated based on the loans they hold
            investor.update(cycle=self.cycle + 1)
        self.cycle += 1

    def get_average_portfolio_values(self):
        # Calculate the average portfolio value at the end of each cycle
        avg_portfolio_values = []
        for cycle in range(self.cycle):
            cycle_values = [investor.portfolio_values[cycle] for investor in self.investors if
                            investor.portfolio_values]
            avg_value = sum(cycle_values) / len(cycle_values) if cycle_values else 0
            avg_portfolio_values.append(avg_value)
        return avg_portfolio_values

markettrial = loanMarket()
markettrial.initialize()
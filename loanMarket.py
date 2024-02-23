import matplotlib.pyplot as plt

import Agents.Loan as Loan
import Agents.LoanInvestor as LoanInvestor
import Agents.LoanTrader as LoanTrader
import numpy as np
import streamlit as st
import matplotlib as mpl


class loanMarket:
    def __init__(self):

        self.cycle = 0
        self.num_loans = st.number_input('Number of loans', min_value=1, max_value=100000, value=100, step=10)
        self.num_investors = st.number_input('Number of investors', min_value=1, max_value=1000, value=100, step=10)
        self.num_traders = st.number_input('Number of traders', min_value=1, max_value = self.num_investors, value=10, step=1)
        self.interest_rate = st.slider('Interest Rate', min_value=0.00, max_value=0.15, value=0.05, step=0.01)
        self.broker_fee = st.slider('Broker Fee', min_value=0.01, max_value=1.0, value=0.15, step=0.01)
        self.min_capital = st.slider('Minimum Capital Percent', min_value=0.01, max_value=0.30, value=0.05, step=0.01)
        self.default_rate = st.slider('Default Rate', min_value=1, max_value=300, value=100, step=10)
        self.reserve_price = st.slider('Reserve Price', min_value=0.01, max_value=1.0, value=0.8, step=0.01)


        # creating the universe of loans
        self.loans = [Loan.LoanObj(float_interest=self.interest_rate, default_rate=301 - self.default_rate, reserve_price=self.reserve_price) for _ in range(self.num_loans)]

        # creating the universe of investors
        self.investors = [LoanInvestor.LoanInvestorObj(min_capital=self.min_capital, target_score_param=100) for _ in range(self.num_investors)]

        # creating the universe of traders
        self.traders = [LoanTrader.LoanTraderObj(max_investors=self.num_investors // self.num_traders, broker_fee=self.broker_fee) for _ in
                        range(self.num_traders)]

    def initialize(self):

        # Assigning loans to investors
        for investor in self.investors:
            available_loans = [loan for loan in self.loans if loan.current_owner == 'no owner']
            investor.initialize_portfolio(available_loans)

        # Assigning a trader to the investors
        for trader in self.traders:
            trader.add_investors(self.investors)

    def update(self):

        # creating new loans
        new_loans = [Loan.LoanObj(float_interest=self.interest_rate, default_rate=301 - self.default_rate) for _ in range(self.num_loans//100)]
        self.loans.extend(new_loans)

        for loan in self.loans:
            # loans are updated independent of investors
            loan.update(self.cycle + 1, float_interest=self.interest_rate)
        for investor in self.investors:
            # investors are updated based on the loans they hold
            investor.update(cycle=self.cycle + 1)
        for trader in self.traders:
            # traders are updated based on the investors they hold
            trader.update(cycle=self.cycle + 1)
            trader.update_loans_for_sale(new_loans)

        self.cycle += 1

    def get_average_portfolio_values(self, plot_values=False):
        # Calculate the average portfolio value at the end of each cycle
        avg_portfolio_values = []
        for cycle in range(self.cycle):
            cycle_values = [investor.portfolio_values[cycle] for investor in self.investors if
                            investor.portfolio_values]
            avg_value = sum(cycle_values) / len(cycle_values) if cycle_values else 0
            avg_portfolio_values.append(avg_value)

        if plot_values:
            plt.plot(avg_portfolio_values)
            plt.title('Average Portfolio Value')
            plt.xlabel('Cycle')
            plt.ylabel('Value')
            plt.show()

        return avg_portfolio_values

    def plot_portfolio_values(self):
        ax = plt.figure()
        # Plot the portfolio value for each investor
        for investor in self.investors:
            plt.plot(investor.portfolio_values)
        plt.title('Portfolio Value')
        plt.xlabel('Cycle')
        plt.ylabel('Value')
        st.pyplot(ax)

    def plot_capital_values(self):
        ax = plt.figure()
        for investor in self.investors:
            plt.plot(investor.capital_history)
        plt.title('Capital Value')
        plt.xlabel('Cycle')
        plt.ylabel('Value')
        st.pyplot(ax)

    def plot_sale_prices(self):
        ax = plt.figure()

        plotting_bar = st.progress(0, text="Plotting sale prices")

        # loans that sold
        loans_that_sold = [loan for loan in self.loans if loan.sale_price_history[-1] is not None]

        for loan in loans_that_sold:
            plotting_bar.progress(loans_that_sold.index(loan) / len(loans_that_sold), text="Plotting sale prices")
            if loan.sale_price_history[-1] > 10:
                plt.plot(loan.sale_price_history)

        plt.title('Sale Price')
        plt.xlabel('Cycle')
        plt.ylabel('Value')
        st.pyplot(ax)

        plotting_bar.empty()

    def print_parameter_values(self):

        col1, col2, col3 = st.columns(3)
        col1.metric(label='Number of loans', value=self.num_loans)
        col1.metric(label='Number of investors', value=self.num_investors)
        col1.metric(label = 'Number of Defaulted Loans', value = len([loan for loan in self.loans if loan.defaulted]))
        col2.metric(label='Number of traders', value=self.num_traders)
        col2.metric(label='Interest Rate', value=self.interest_rate)
        col3.metric(label='Broker Fee', value=self.broker_fee)
        col3.metric(label='Minimum Capital Percent', value=self.min_capital)

    def get_winners_losers_capital(self, plot_values=False):
        # Calculate the winners and losers as the highest capital difference between beginning and end
        winners = []
        losers = []

        for investor in self.investors:
            if investor.capital_history:
                capital_diff = investor.capital_history[-1] - investor.capital_history[0]
                if capital_diff > 0:
                    winners.append(investor)
                else:
                    losers.append(investor)

        ax1 = plt.figure()
        plt.scatter([investor.id for investor in winners], sorted([investor.capital_history[-1] - investor.capital_history[0] for investor in winners]), label='Winners', color='g')
        plt.title('Plot of {} Capital Winners'.format(len(winners)))
        plt.xticks(visible=False)
        plt.legend()
        st.pyplot(ax1)

        ax2 = plt.figure()
        plt.scatter([investor.id for investor in losers], sorted([investor.capital_history[-1] - investor.capital_history[0] for investor in losers]), label='Losers', color='r')
        plt.title('Plot of {} Capital Losers'.format(len(losers)))
        plt.xticks(visible=False)
        plt.legend()
        st.pyplot(ax2)

    def get_winners_losers_portfolio_value(self):
        winners = []
        losers = []

        for investor in self.investors:
            if investor.portfolio_values:
                portfolio_diff = investor.portfolio_values[-1] - investor.portfolio_values[0]
                if portfolio_diff > 0:
                    winners.append(investor)
                else:
                    losers.append(investor)

        ax1 = plt.figure()
        plt.scatter([investor.id for investor in winners], sorted([investor.portfolio_values[-1] - investor.portfolio_values[0] for investor in winners]), label='Winners', color='g')
        plt.title('Plot of {} Portfolio Value Winners'.format(len(winners)))
        plt.xticks(visible=False)
        plt.legend()
        st.pyplot(ax1)

        ax2 = plt.figure()
        plt.scatter([investor.id for investor in losers], sorted([investor.portfolio_values[-1] - investor.portfolio_values[0] for investor in losers]), label='Losers', color='r')
        plt.title('Plot of {} Portfolio Value Losers'.format(len(losers)))
        plt.xticks(visible=False)
        plt.legend()
        st.pyplot(ax2)


    def plot_trader_revenue(self):
        # plotting the trader revenue

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        for trader in self.traders:
            ax1.plot(trader.revenue_history)
            ax2.plot(trader.broker_revenue_history)

        ax1.set_title('Trader Revenue')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('Revenue')

        ax2.set_title('Broker Revenue')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Revenue')

        # Displaying the plots using Streamlit
        st.pyplot(fig1)
        st.pyplot(fig2)

        return

    def analyze_capital_history(self):
        # Analyze the capital history of the investors

        ax1 = plt.figure()

        for investor in self.investors[:1]:
            st.write("Matured loans: ", [loan.fair_value_history[-2] for loan in investor.matured_loans])
            st.write("Portfolio: ", investor.portfolio)

            if investor.matured_loans == []:

                # comparing the purchased_loans and sold_loans lists
                if investor.purchased_loans == investor.sold_loans:
                    st.write("Investor {} has no matured loans and has sold all purchased loans".format(investor.id[:5]))
                else:
                    st.write("Investor should be holding loans")
                    st.write("Investor Portfolio: ", investor.portfolio)
                    st.write("Investor Matured Loans: ", investor.matured_loans)

                st.write("Purchased loans: ", investor.purchased_loans)
                st.write("Sold loans: ", investor.sold_loans)


        for investor in self.investors[:1]:
            plt.plot(investor.capital_history, color='black', label=investor.id, linestyle='solid')

        plt.title('Capital History')
        plt.xlabel('Cycle')
        plt.ylabel('Value')
        plt.legend()

        st.pyplot(ax1)

        return

    def analyze_loan_market(self):

        # Sort the loans based on the change in fair value
        loan_sale_sorted = sorted(self.loans, key=lambda x: x.sale_price_history[-1] - x.sale_price_history[0])
        winners_sale = loan_sale_sorted[-3:]  # Top 3 loans with the highest increase in market value
        # filtering losers that defaulted
        losers_sale = [loan for loan in loan_sale_sorted if loan.defaulted == False][:3]

        # Create a figure and two subplots (axes)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))  # Adjust figsize as needed

        # Plot winners on the first subplot
        for loan in winners_sale:
            ax1.plot(loan.sale_price_history, label=f"Loan {loan.id[:5]} Sale", linestyle='dashed')

        ax1.set_title('Winning Loans (by sale)')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('Value')
        ax1.legend()

        # Plot losers on the second subplot
        for loan in losers_sale:
            ax2.plot(loan.sale_price_history, label=f"Loan {loan.id[:5]} Sale", linestyle='dashed')

        ax2.set_title('Losing Loans (by sale)')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Value')
        ax2.legend()

        # plotting winners and losers by market price
        # Plot winners on the first subplot
        loans_market = sorted(self.loans, key=lambda x: x.market_price_history[-1] - x.market_price_history[0])
        winners_market = loans_market[-3:]  # Top 3 loans with the highest increase in market value
        losers_market = loans_market[:3]

        for loan in winners_market:
            ax3.plot(loan.market_price_history, label=f"Loan {loan.id[:5]} Market Winner", linestyle='dashed')
        for loan in losers_market:
            ax3.plot(loan.market_price_history, label=f"Loan {loan.id[:5]} Market Loser", linestyle='dashed')

        ax3.set_title('Winning and Losing Loans (by market price)')
        ax3.set_xlabel('Cycle')
        ax3.set_ylabel('Value')
        ax3.legend()

        plt.tight_layout()  # Adjust the layout to make sure there's no overlap
        st.pyplot(fig)

    def plot_total_market_value(self, cycles):

        # Initialize lists to store total market values and capitals for each cycle
        total_market_values = []
        total_capitals = []
        total_portfolio_values = []

        # Calculate the total market value for each cycle
        for cycle in range(self.cycle):
            cycle_market_value = 0
            for loan in self.loans:
                # Check if the loan has a fair value for the current cycle
                if cycle < len(loan.fair_value_history):
                    cycle_market_value += ((loan.fair_value_history[cycle])/100) * loan.size

            total_market_values.append(cycle_market_value)

        # Calculate the total capital for each cycle
        # Assuming all investors have a capital history for each cycle
        for cycle in range(self.cycle):
            cycle_total_capital = sum(investor.capital_history[cycle] for investor in self.investors)
            total_capitals.append(cycle_total_capital)


        # calculate the total portfolio value for each cycle
        for cycle in range(self.cycle):
            cycle_portfolio_value = sum(investor.portfolio_values[cycle] for investor in self.investors)
            total_portfolio_values.append(cycle_portfolio_value)

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))  # Create a figure and two subplots

        # Plot total market value
        axs[0].plot(total_market_values)
        axs[0].set_title('Total Market Value')
        axs[0].set_xlabel('Cycle')
        axs[0].set_ylabel('Value')

        # Plot total capital
        axs[1].plot(total_capitals)
        axs[1].set_title('Total Capital')
        axs[1].set_xlabel('Cycle')
        axs[1].set_ylabel('Value')

        # Plot total portfolio value
        axs[2].plot(total_portfolio_values)
        axs[2].set_title('Total Portfolio Value')
        axs[2].set_xlabel('Cycle')
        axs[2].set_ylabel('Value')

        plt.tight_layout()
        plt.ticklabel_format(style='plain', axis='y')
        axs[2].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.show()

        st.pyplot(fig)

markettrial = loanMarket()
markettrial.initialize()

cycles = st.slider("Number of cycles", 0, 500, 100)

progress_text = "Simulation in progress. Please wait."
cycle_bar = st.progress(0, text=progress_text)

for _ in range(cycles):
    cycle_bar.progress(_/cycles, text=progress_text)
    markettrial.update()

cycle_bar.empty()

markettrial.print_parameter_values()
markettrial.plot_portfolio_values()
markettrial.plot_capital_values()
markettrial.plot_sale_prices()
markettrial.get_winners_losers_capital(plot_values=True)
markettrial.get_winners_losers_portfolio_value()
markettrial.plot_trader_revenue()

markettrial.analyze_capital_history()
markettrial.analyze_loan_market()
markettrial.plot_total_market_value(cycles)
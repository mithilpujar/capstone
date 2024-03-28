import matplotlib.pyplot as plt

import Agents.Loan as Loan
import Agents.LoanInvestor as LoanInvestor
import Agents.LoanTrader as LoanTrader
import numpy as np
import streamlit as st
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import warnings
import time
import pickle

# clearing the cache
loanMarket = None

class loanMarket:
    def __init__(self, cycles):

        warnings.filterwarnings("ignore")

        self.cycle = 0
        self.num_loans = st.number_input('Number of loans', min_value=1, max_value=100000, value=1000, step=10)
        self.num_investors = st.number_input('Number of investors', min_value=1, max_value=1000, value=100, step=10)
        self.num_traders = st.number_input('Number of traders', min_value=1, max_value = self.num_investors, value=10, step=1)
        self.interest_rate = st.slider('Interest Rate', min_value=0.00, max_value=0.15, value=0.05, step=0.01)
        self.broker_fee = st.slider('Broker Fee (bps)', min_value=0, max_value=100, value=15, step=1)/100
        self.min_capital = st.slider('Minimum Capital Percent', min_value=0.01, max_value=0.30, value=0.05, step=0.01)
        self.default_rate = st.slider('Default Rate', min_value=1, max_value=300, value=100, step=10)
        self.reserve_price = st.slider('Reserve Price', min_value=0.01, max_value=1.0, value=0.8, step=0.01)
        self.recovery_value = st.slider('Recovery Value', min_value=1, max_value=100, value=40, step=1)
        self.new_loans_slider = st.slider('Number of new loans', min_value=1, max_value=10, value=5, step=1)


        # clearing cache
        self.loans = None
        self.new_loans = None
        self.investors = None
        self.traders = None


        # creating the universe of loans
        self.loans = [Loan.LoanObj(float_interest=self.interest_rate, default_rate=301 - self.default_rate, reserve_price=self.reserve_price, recovery_value=self.recovery_value) for _ in range(self.num_loans)]

        # creating all new loans that will be used in the simulation
        self.new_loans = [[Loan.LoanObj(float_interest=self.interest_rate, default_rate=301 - self.default_rate, reserve_price=self.reserve_price, recovery_value=self.recovery_value) for _ in range(int(self.new_loans_slider/100 * self.num_loans))] for _ in range(cycles)]

        # creating the universe of investors
        self.investors = [LoanInvestor.LoanInvestorObj(min_capital=self.min_capital, target_score_param=0.250) for _ in range(self.num_investors)]

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

    def update(self, last_issue_cycle):

        # getting this cycles new loans
        new_loans = self.new_loans[self.cycle]

        # splitting the list of new loans into sets to be allocated to individual traders
        split_new_loans = [new_loans[i:i + len(new_loans) // self.num_traders] for i in range(0, len(new_loans), len(new_loans) // self.num_traders)]

        for loan in new_loans:
            loan.starting_cycle = self.cycle
            loan.ending_cycle = loan.starting_cycle + loan.maturity


        self.loans.extend(new_loans)

        start_time_loans = time.time()
        for loan in self.loans:
            # loans are updated independent of investors
            loan.update(self.cycle + 1, float_interest=self.interest_rate)
        end_time_loans = time.time()
        loan_update_time = end_time_loans - start_time_loans
        loan_time_display.text(f"Last loans update took: {loan_update_time:.2f} seconds")

        start_time_investors = time.time()
        for investor in self.investors:
            # investors are updated based on the loans they hold
            investor.update(cycle=self.cycle + 1, float_interest=self.interest_rate)
        end_time_investors = time.time()
        investors_update_time = end_time_investors - start_time_investors
        investor_time_display.text(f"Last investors update took: {investors_update_time:.2f} seconds")

        start_time_traders = time.time()
        for trader in self.traders:
            # traders are updated based on the investors they hold
            trader.update(cycle=self.cycle + 1)
            if self.cycle < last_issue_cycle:
                trader.update_loans_for_sale(split_new_loans[self.traders.index(trader)])
        end_time_traders = time.time()
        traders_update_time = end_time_traders - start_time_traders
        trader_time_display.text(f"Last traders update took: {traders_update_time:.2f} seconds")

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

        # plotting portfolio values and interest received each as subplots
        fig, axs = plt.subplots(4, 1, figsize=(8, 16))

        # Plot the portfolio values
        for investor in self.investors:
            axs[0].plot(investor.portfolio_values)
        axs[0].set_title('Portfolio Value')
        axs[0].set_xlabel('Cycle')
        axs[0].set_ylabel('Value')

        # Plot the capital values
        for investor in self.investors:
            axs[1].plot(investor.capital_history)
        axs[1].set_title('Capital Value')
        axs[1].set_xlabel('Cycle')
        axs[1].set_ylabel('Value')


        # Plot the interest received
        for investor in self.investors:
            axs[2].plot(investor.interest_received)
        axs[2].set_title('Interest Received')
        axs[2].set_xlabel('Cycle')
        axs[2].set_ylabel('Value')

        # Plot loan fair values
        for investor in self.investors:
            axs[3].plot(investor.loan_fair_values)
        axs[3].set_title('Loan Fair Values')
        axs[3].set_xlabel('Cycle')
        axs[3].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

        st.pyplot(fig)


    def plot_fair_values(self):

        ax1 = plt.figure()
        # plotting the average fair value of all loans in the market
        for loan in self.loans:
            plt.plot(loan.fair_value_history)
        plt.title('Loan Fair Values')

        st.pyplot(ax1)

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
        col2.metric(label='Value of Defaulted Loans', value=round(np.mean([loan.fair_value_history[-1] for loan in self.loans if loan.defaulted]), 2))

        col3.metric(label='Broker Fee', value=self.broker_fee)
        col3.metric(label='Minimum Capital Percent', value=self.min_capital)

    def get_winners_losers_capital(self, plot_values=False):
        # Calculate the winners and losers as the highest capital difference between beginning and end
        winners = []
        losers = []

        for investor in self.investors:
            if investor.capital_history:
                capital_diff = investor.capital_history[-1] - investor.capital_history[1]
                if capital_diff > 0:
                    winners.append(investor)
                else:
                    losers.append(investor)

        ax1 = plt.figure()
        plt.scatter([investor.id for investor in winners], sorted([investor.capital_history[-1] - investor.capital_history[1] for investor in winners]), label='Winners', color='g')
        plt.title('Plot of {} Capital Winners'.format(len(winners)))
        plt.xticks(visible=False)
        plt.legend()
        st.pyplot(ax1)

        ax2 = plt.figure()
        plt.scatter([investor.id for investor in losers], sorted([investor.capital_history[-1] - investor.capital_history[1] for investor in losers]), label='Losers', color='r')
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

    def analyze_top_bottom_investor(self):

        # Formatter function to add commas
        def comma_formatter(x, pos):
            return "{:,}".format(int(x))

        # Analyze the capital history of the investors

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Adjusted to create a 2x2 grid of plots

        # First find the top investor by capital increase
        top_investor = max(self.investors, key=lambda x: x.capital_history[-1] - x.capital_history[0])
        bottom_investor = min(self.investors, key=lambda x: x.capital_history[-1] - x.capital_history[0])

        # Plot for Top Investor (Financial Metrics)
        axs[0, 0].plot(top_investor.capital_history, label='Capital', color='g')
        axs[0, 0].plot(top_investor.portfolio_values, label='Portfolio Value', color='r')
        axs[0, 0].set_title(f'Top Investor {top_investor.id[:5]} Financial History')
        axs[0, 0].set_ylabel('Value')
        axs[0, 0].legend(loc='upper left')
        axs[0, 0].yaxis.set_major_formatter(FuncFormatter(comma_formatter))

        # Plot for Top Investor (Score Metrics)
        axs[0, 1].plot(top_investor.current_score_history, label='Current Score', color='c')
        axs[0, 1].axhline(y=top_investor.target_score, color='m', linestyle=':', label='Target Score')
        axs[0, 1].set_title(f'Top Investor {top_investor.id[:5]} Score History')
        axs[0, 1].set_ylabel('Score')
        axs[0, 1].legend(loc='upper left')

        # Plot for Bottom Investor (Financial Metrics)
        axs[1, 0].plot(bottom_investor.capital_history, label='Capital', color='g')
        axs[1, 0].plot(bottom_investor.portfolio_values, label='Portfolio Value', color='r')
        axs[1, 0].set_title(f'Bottom Investor {bottom_investor.id[:5]} Financial History')
        axs[1, 0].set_ylabel('Value')
        axs[1, 0].legend(loc='upper left')
        axs[1, 0].yaxis.set_major_formatter(FuncFormatter(comma_formatter))

        # Plot for Bottom Investor (Score Metrics)
        axs[1, 1].plot(bottom_investor.current_score_history, label='Current Score', color='c')
        axs[1, 1].axhline(y=bottom_investor.target_score, color='m', linestyle=':', label='Target Score')
        axs[1, 1].set_title(f'Bottom Investor {bottom_investor.id[:5]} Score History')
        axs[1, 1].set_ylabel('Score')
        axs[1, 1].legend(loc='upper left')

        # Adjust layout for better readability
        plt.tight_layout()

        # Use Streamlit's pyplot function to display the matplotlib figure
        st.pyplot(fig)

        return

    def analyze_loan_market(self):

        # Sort the loans based on the change in fair value
        loan_sale_sorted = sorted(self.loans, key=lambda x: x.sale_price_history[-1] - x.sale_price_history[0])
        winners_sale = loan_sale_sorted[-3:]  # Top 3 loans with the highest increase in market value
        # filtering losers that defaulted
        losers_sale = [loan for loan in loan_sale_sorted if loan.defaulted == False][:3]

        # Create a figure and two subplots (axes)
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 30))  # Adjust figsize as needed

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
            ax4.plot(loan.market_price_history, label=f"Loan {loan.id[:5]} Market Loser", linestyle='dashed')

        ax3.set_title('Winning Loans (by market price)')
        ax3.set_xlabel('Cycle')
        ax3.set_ylabel('Value')
        ax3.legend()

        ax4.set_title('Losing Loans (by market price)')
        ax4.set_xlabel('Cycle')
        ax4.set_ylabel('Value')
        ax4.legend()

        # plotting winners and losers by fair value
        # Plot winners on the first subplot
        loans_fair = sorted(self.loans, key=lambda x: x.fair_value_history[-1] - x.fair_value_history[0])
        winners_fair = loans_fair[-3:]  # Top 3 loans with the highest increase in market value
        losers_fair = loans_fair[:3]

        for loan in winners_fair:
            ax5.plot(loan.fair_value_history, label=f"Loan {loan.id[:5]} Fair Winner", linestyle='dashed')
        for loan in losers_fair:
            ax6.plot(loan.fair_value_history, label=f"Loan {loan.id[:5]} Fair Loser", linestyle='dashed')

        ax5.set_title('Winning Loans (by fair value)')
        ax5.set_xlabel('Cycle')
        ax5.set_ylabel('Value')
        ax5.legend()

        ax6.set_title('Losing Loans (by fair value)')
        ax6.set_xlabel('Cycle')
        ax6.set_ylabel('Value')
        ax6.legend()

        plt.tight_layout()  # Adjust the layout to make sure there's no overlap
        st.pyplot(fig)

    def plot_total_market_value(self, cycles):

        # Initialize lists to store total market values and capitals for each cycle
        total_market_values = []
        total_capitals = []
        total_portfolio_values = []
        num_loans_over_time = []
        total_size_loans = []


        # Calculate the total market value for each cycle and number of loans in each cycle
        for cycle in range(self.cycle):
            cycle_market_value = 0
            num_loans = self.num_loans
            for loan in self.loans:
                # Check if the loan has a fair value for the current cycle
                if cycle < len(loan.fair_value_history):
                    cycle_market_value += ((loan.fair_value_history[cycle])/100) * loan.size
                if loan.maturity_bool == False:
                    num_loans += 1

            # calculating the total market value each cycle
            total_market_values.append(cycle_market_value)
            num_loans_over_time.append(num_loans)

            # calculating the total capital for each cycle
            cycle_total_capital = sum(investor.capital_history[cycle] for investor in self.investors)
            total_capitals.append(cycle_total_capital)

            # calculating total portfolio value each cycle
            cycle_portfolio_value = sum(investor.portfolio_values[cycle] for investor in self.investors)
            total_portfolio_values.append(cycle_portfolio_value)

            # calculating total size of loans
            cycle_total_size_loans = sum(loan.size for loan in self.loans)
            total_size_loans.append(cycle_total_size_loans)



        # Plotting
        fig, axs = plt.subplots(5, 1, figsize=(12, 8))  # Create a figure and two subplots

        # Plot number of loans over time
        axs[0].plot(num_loans_over_time)
        axs[0].set_title('Number of Loans Over Time')
        axs[0].set_xlabel('Cycle')
        axs[0].set_ylabel('Number of Loans')

        # Plot total market value
        axs[1].plot(total_market_values)
        axs[1].set_title('Total Market Fair Values')
        axs[1].set_xlabel('Cycle')
        axs[1].set_ylabel('Value')

        # Plot total capital
        axs[2].plot(total_capitals)
        axs[2].set_title('Total Capital')
        axs[2].set_xlabel('Cycle')
        axs[2].set_ylabel('Value')

        # Plot total portfolio value
        axs[3].plot(total_portfolio_values)
        axs[3].set_title('Total Portfolio Value')
        axs[3].set_xlabel('Cycle')
        axs[3].set_ylabel('Value')

        # Plot total size of loans
        axs[4].plot(total_size_loans)
        axs[4].set_title('Total Size of Loans')
        axs[4].set_xlabel('Cycle')
        axs[4].set_ylabel('Value')

        plt.tight_layout()
        plt.ticklabel_format(style='plain', axis='y')
        axs[2].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        axs[3].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.show()

        st.pyplot(fig)

    def plot_score_allocation(self, before_after):
        num_investors = len(self.investors)

        # Extracting the capital, current scores, and target scores
        capital = np.array([investor.portfolio_values[-1] for investor in self.investors])
        current_scores = np.array([investor.current_score for investor in self.investors])
        target_scores = np.array([investor.target_score for investor in self.investors])

        # Sorting the scores and capital for a more organized visualization
        indices = np.argsort(target_scores)
        current_scores_sorted = current_scores[indices]
        target_scores_sorted = target_scores[indices]
        capital_sorted = capital[indices]

        # Calculating the total capital above and below target score
        total_capital_above_target = sum(
            capital[i] for i in range(num_investors) if current_scores[i] > target_scores[i])
        total_capital_below_target = sum(
            capital[i] for i in range(num_investors) if current_scores[i] < target_scores[i])

        # Normalize capital for scatter plot size (adjust scale factor as needed)
        size_factor = 0.000001  # Adjust this factor to scale the sizes appropriately
        sizes = capital_sorted * size_factor

        # Creating the scatter plot with connected lines
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size to create space for the text above
        for i in range(len(self.investors)):
            plt.plot([i, i], [current_scores_sorted[i], target_scores_sorted[i]], color='purple')  # lines
        plt.scatter(range(num_investors), current_scores_sorted, s=sizes, color='blue', label='Current Score', zorder=5)
        plt.scatter(range(num_investors), target_scores_sorted, color='red', label='Target Score', zorder=5)

        # Adjust the subplot parameters to make room for the text
        plt.subplots_adjust(top=0.85)

        # Plotting the annotations above the plot
        y_texts = [0.95, 0.92, 0.89, 0.86]  # Adjust these positions as needed
        texts = [
            f'Number of investors above target score: {sum(current_scores > target_scores)}',
            f'Number of investors below target score: {sum(current_scores < target_scores)}',
            f'Total capital above target score: {total_capital_above_target:,.2f}',
            f'Total capital below target score: {total_capital_below_target:,.2f}'
        ]
        for y, text in zip(y_texts, texts):
            fig.text(0.02, y, text, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                     transform=fig.transFigure)

        plt.ylabel("Score")
        plt.xlabel("Investor")
        plt.title("Investor Current Score and Target Score " + before_after)
        plt.legend()
        plt.grid(True)
        plt.show()

        # If you're using Streamlit, uncomment the following line to display the plot
        st.pyplot(fig)


# clearing the cache
markettrial = None

# creating sliders for the user to input the parameterr
cycles = st.slider("Number of cycles", 0, 500, 200)
last_issue_cycle = st.number_input("Last issue cycle", 0, cycles, cycles - 100)

#initializing the market
markettrial = loanMarket(cycles=cycles)
markettrial.initialize()

# creating a proceed button to run the simulation when we want
proceed = st.button("Proceed")

if proceed:
    import streamlit as st
    from datetime import datetime, timezone

    # Initialize the progress bar
    progress_bar = st.progress(0)

    # Dynamic text
    cycle_display = st.empty()
    time_display = st.empty()
    loan_time_display = st.empty()
    investor_time_display = st.empty()
    trader_time_display = st.empty()

    # Placeholder for progress text
    progress_text_display = st.empty()
    progress_text = "Simulation in progress. Please wait."
    progress_text_display.text(progress_text)

    markettrial.plot_score_allocation(before_after="Before")

    # For loop to simulate progress
    for i in range(cycles):
        start_time = time.time()  # Start time of the update
        markettrial.update(last_issue_cycle=last_issue_cycle)
        end_time = time.time()  # End time of the update

        update_duration = end_time - start_time  # Calculate the duration of the update

        # Update the progress bar
        progress_bar.progress((i + 1) / cycles)

        # updating the cycle display
        cycle_display.text(f"Cycle: {i + 1} of {cycles}")

        # Update the time display with the duration of the last update
        time_display.text(f"Last update took: {update_duration:.2f} seconds")

    # Clear the progress bar and text once the loop is complete
    progress_bar.empty()
    progress_text_display.empty()
    cycle_display.empty()
    time_display.empty()
    loan_time_display.empty()
    trader_time_display.empty()
    investor_time_display.empty()

    markettrial.plot_score_allocation(before_after="After")
    markettrial.print_parameter_values()
    markettrial.plot_portfolio_values()
    markettrial.plot_fair_values()
    markettrial.get_winners_losers_capital(plot_values=True)
    markettrial.get_winners_losers_portfolio_value()
    markettrial.plot_trader_revenue()

    markettrial.analyze_top_bottom_investor()
    markettrial.analyze_loan_market()

    markettrial.plot_total_market_value(cycles)

    markettrial.plot_sale_prices()

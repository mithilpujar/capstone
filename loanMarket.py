import Agents.Loan as Loan
import Agents.LoanInvestor as LoanInvestor
import Agents.LoanTrader as LoanTrader
import numpy as np

class loanMarket:
    def __init__(self, num_loans, num_investors, num_traders):

        self.cycle = 0
        self.num_loans = num_loans
        self.num_investors = num_investors
        self.num_traders = num_traders

        # creating the universe of loans
        self.loans = [Loan.Loan() for _ in range(self.num_loans)]

        # creating the universe of investors
        self.investors = [LoanInvestor.LoanInvestor() for _ in range(self.num_investors)]

        # creating the universe of traders
        self.traders = [LoanTrader.LoanTrader(max_investors=self.num_investors // self.num_traders) for _ in
                        range(self.num_traders)]

    def initialize(self):

        for investor in self.investors:
            # regenerating list of available loans
            available_loans = [loan for loan in self.loans if loan.current_owner == "no owner"]
            investor.initialize_portfolio(available_loans)

        # Assign unsold loans to traders
        unsold_loans = [loan for loan in self.loans if loan.current_owner == "no owner"]
        unsold_loans_count = len(unsold_loans)
        loans_per_trader = unsold_loans_count // self.num_traders

        for trader_ in self.traders:
            trader_loans = unsold_loans[:loans_per_trader]
            unsold_loans = unsold_loans[loans_per_trader:]  # Update the unsold_loans list
            trader_.initialize_book(trader_loans)

            # Assign investors to the trader if they haven't reached their maximum
            if not trader_.max_investors_reached:
                trader_.add_investor(self.investors)

        # If there are any leftover loans, assign them randomly to the traders
        for loan in unsold_loans:
            trader = np.random.choice(self.traders)
            loan.update_owner(trader.id)
            trader.loans_for_sale.append(loan)

        # Assign any unassigned investors to traders randomly
        unassigned_investors = [investor for investor in self.investors if investor.trader is None]
        for investor in unassigned_investors:
            trader = np.random.choice(self.traders)
            trader.investors.append(investor)
            investor.trader = trader

        # Assign the partner trader to each trader
        for trader in self.traders:
            partner_trader = np.random.choice([t for t in self.traders if t != trader], replace=False)
            trader.partner_trader = partner_trader

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

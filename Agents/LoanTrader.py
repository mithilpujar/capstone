import uuid
import numpy as np
import streamlit as st

class LoanTraderObj:
    def __init__(self, max_investors=10, broker_fee=0.0, partner_fee=0.25):
        self.id = 'T' + str(uuid.uuid4())
        self.partner_trader = None
        self.broker_fee = broker_fee  # fee for trading through the broker in bps
        self.friction_fee_partner = partner_fee  # frictional fee for trading through the partner trader in bps
        self.max_investors = max_investors
        self.max_investors_reached = False
        self.current_cycle = 0
        self.loans_for_sale = []
        self.investors = []
        self.broker_revenue_history = []
        self.interest_revenue_history = []
        self.revenue_history = []
        self.cycle_broker_revenue = 0
        self.num_sales = 0

    def add_investors(self, investors):
        for investor in investors:
            if investor.trader is None and len(self.investors) < self.max_investors:
                self.investors.append(investor)
                investor.trader = self
            else:
                self.max_investors_reached = True
                pass

    def update_loans_for_sale(self, new_loans):
        for loan in new_loans:
            self.loans_for_sale.append(loan)
            loan.update_owner(self)

    def collect_loans_for_sale(self, print_outputs = False, num_investors = 1):


        # method to collect the loans for sale from the investors
        investors_with_loans_for_sale = [investor for investor in self.investors if investor.get_loan_to_sell() is not None]

        for investor in investors_with_loans_for_sale:
            # check the loan isn't already for sale
            loan = investor.get_loan_to_sell()
            if loan is not None and loan not in self.loans_for_sale:
                # dropping none from the list of loans for sale
                self.loans_for_sale.append(loan)


        if print_outputs:
            print('Trader {} has {} loans for sale.'.format(self.id[:5], len(self.loans_for_sale)))
            print('Investors with loans listed: ', [loan.current_owner.id[:5] for loan in self.loans_for_sale])
            print('Loans for sale: ', [loan.id[:5] for loan in self.loans_for_sale])

        return

    def run_auction(self, show_bids = False):

        '''
        The following method runs the auction amongst the investors that it services along with clearing the sale of the loans and transfering ownership.
        :param show_bids:
        :return:
        '''

        if self.loans_for_sale == []:
            return

        # method to run the auction for the loans by collecting bid prices and choosing the highest bid

        # purging the loans for sale book if the loan has matured
        self.loans_for_sale = [loan for loan in self.loans_for_sale if not loan.maturity_bool]

        loans_sold = []

        for loan in self.loans_for_sale:

            purchased = False
            top_bidder = {'investor': None, 'bid_price': 0}
            # the potential bidders are those who don't already own the loan
            available_bidders = [investor for investor in self.investors if investor.id != loan.current_owner.id]

            for investor in available_bidders:
                bid = investor.get_bid_price(loan)

                if bid > top_bidder['bid_price']:
                    top_bidder['investor'] = investor
                    top_bidder['bid_price'] = bid

                if show_bids:
                    print('Investor {} bids {} for loan {}'.format(investor.id[:5], bid, loan.id[:5]))



            # updating the loan market price history
            loan.market_price_history.append(top_bidder['bid_price'])
            loan.market_price = top_bidder['bid_price']

            # Now we check if the top bid is higher than the loan's reserve price
            # if it is, we clear the sale

            if top_bidder['bid_price'] >= loan.reserve_price:
                # removing the loan from the seller's portfolio or trader's loans for sale
                if loan.current_owner.id[0] == 'I':
                    loan.current_owner.sold_loans.append(loan)
                    loan.current_owner.portfolio.remove(loan)


                # updating the loan's owner
                loan.sale_price_history.append(top_bidder['bid_price'])
                broker_fee_amt = (self.broker_fee/100)*(top_bidder['bid_price']/100)*loan.size
                self.cycle_broker_revenue += broker_fee_amt

                top_bidder['investor'].buy_loan(loan, broker_fee_amt)

                loans_sold.append(loan)
                purchased = True
                self.num_sales += 1

            if show_bids:
                print("Purchased: ", purchased)
                print('Top bidder is {} with bid price {} for ${}'.format(top_bidder['investor'].id[:5], top_bidder['bid_price'], top_bidder['bid_price']/100*loan.size))
                print("Loan Ownership History: ", loan.ownership_history)
                #print("\n Top Bidder Attributes: ", vars(top_bidder['investor']))

        # removing the loans that have been sold from the trader's loans for sale
        for loan in loans_sold:
            self.loans_for_sale.remove(loan)

        return

    def receive_interest_payments(self):
        total_interest = 0
        for loan in self.loans_for_sale:
            if not loan.maturity_bool and loan.current_owner.id[0] == 'T':
                total_interest += (loan.interest_rate / 12) * loan.size

        self.interest_revenue_history.append(total_interest)

    def update(self, cycle, num_investors=3):
        # consolidating trader actions into a unified method
        self.cycle_broker_revenue = 0

        self.current_cycle = cycle
        self.collect_loans_for_sale(num_investors=num_investors)
        self.run_auction()
        self.receive_interest_payments()
        self.broker_revenue_history.append(self.cycle_broker_revenue)
        self.revenue_history.append(sum(self.broker_revenue_history) + sum(self.interest_revenue_history))

        return

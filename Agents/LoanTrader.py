import uuid
import numpy

class LoanTrader:
    def __init__(self, max_investors=10, broker_fee=0.15, partner_fee=0.25):
        self.id = 'T' + str(uuid.uuid4())
        self.partner_trader = None
        self.broker_fee = broker_fee  # fee for trading through the broker in bps
        self.friction_fee_partner = partner_fee  # frictional fee for trading through the partner trader in bps
        self.max_investors = max_investors
        self.max_investors_reached = False
        self.loans_for_sale = []
        self.investors = []

    def add_investors(self, investors):
        for investor in investors:
            if investor.trader == None and len(self.investors) < self.max_investors:
                self.investors.append(investor)
                investor.trader = self
            else:
                self.max_investors_reached = True
                pass

    def update_loans_for_sale(self, available_loans):
        for loan in available_loans:
            self.loans_for_sale.append(loan)
            loan.update_owner(self.id)


    def run_auction(self, show_bids = False):
        # method to run the auction for the loans by collecting bid prices and choosing the highest bid
        for loan in self.loans_for_sale:
            top_bidder = {'id': None, 'bid_price': 0}
            for investor in self.investors:
                bid = investor.get_bid_price(loan)
                if bid > top_bidder['bid_price']:
                    top_bidder['id'] = investor.id
                    top_bidder['bid_price'] = bid
                if show_bids:
                    print('Investor {} bids {} for loan {}'.format(investor.id[:5], bid, loan.id[:5]))

            if show_bids:
                print('Top bidder is {} with bid price {} \n'.format(top_bidder['id'][:5], top_bidder['bid_price']))
        return

    def sell_loan(self):
        return
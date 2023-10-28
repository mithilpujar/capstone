import uuid
import numpy

class Trader:
    def __init__(self, max_investors, broker_fee=0.15, partner_fee=0.25):
        self.id = 'T' + str(uuid.uuid4())
        self.partner_trader = None
        self.broker_fee = broker_fee  # fee for trading through the broker in bps
        self.friction_fee_partner = partner_fee  # frictional fee for trading through the partner trader in bps
        self.max_investors = max_investors
        self.max_investors_reached = False
        self.loans_for_sale = []
        self.investors = []

    def add_investor(self, investors):
        for investor in investors:
            if investor.trader == None and len(self.investors) < self.max_investors:
                self.investors.append(investor)
                investor.trader = self
            else:
                self.max_investors_reached = True
                pass

    def initialize_book(self, available_loans):
        self.loans_for_sale.append(available_loans)
        for loan in available_loans:
            loan.update_owner(self.id)

    def run_auction(self):
        return
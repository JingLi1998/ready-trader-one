import asyncio
import itertools

from typing import List, Tuple

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side


class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """Initialise a new instance of the AutoTrader class."""
        super(AutoTrader, self).__init__(loop)
        self.order_ids = itertools.count(1)
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = 0

        self.position = 0

        # Future
        self.best_future_bid = 0
        self.best_future_ask = 0

        # ETF
        self.previous_ask_price_diffs = []
        self.previous_bid_price_diffs = []

        # Max Min Price Diffs
        self.max_buy_price_diff = 0
        self.min_sell_price_diff = 0

        # Max Min Prices
        self.max_buy_price = 0
        self.min_sell_price = 0

        # Volumes
        self.current_bid_volume = 0
        self.current_ask_volume = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning(
            "error with order %d: %s", client_order_id, error_message.decode()
        )
        self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_order_book_update_message(
        self,
        instrument: int,
        sequence_number: int,
        ask_prices: List[int],
        ask_volumes: List[int],
        bid_prices: List[int],
        bid_volumes: List[int],
    ) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        current_ask_prices = reversed(ask_prices)
        current_bid_prices = bid_prices
        current_ask_volumes = reversed(ask_volumes)
        current_bid_volumes = bid_volumes
        if instrument == Instrument.FUTURE:
            self.best_future_bid = bid_prices[-1]
            self.best_future_ask = ask_prices[0]
        else:
            if self.best_future_ask == 0 or self.best_future_bid == 0:
                return
            for price in current_ask_prices:
                self.min_sell_price_diff = self.updateMinSellPriceDiffFromAskAdd(
                    price - self.best_future_ask, self.min_sell_price_diff
                )
                self.max_buy_price_diff = self.updateMaxBuyPriceDiffFromAskAdd(
                    price - self.best_future_bid, self.max_buy_price_diff
                )
            for price in current_bid_prices:
                self.min_sell_price_diff = self.updateMinSellPriceDiffFromBidAdd(
                    price - self.best_future_ask, self.min_sell_price_diff
                )
                self.max_buy_price_diff = self.updateMaxBuyPriceDiffFromBidAdd(
                    price - self.best_future_bid, self.max_buy_price_diff
                )
            if self.previous_ask_price_diffs:
                for price in self.previous_ask_price_diffs:
                    self.max_buy_price_diff = self.updateMaxBuyPriceDiffFromAskRemove(
                        price, self.max_buy_price_diff
                    )
            if self.previous_bid_price_diffs:
                for price in self.previous_bid_price_diffs:
                    self.min_sell_price_diff = self.updateMinSellPriceDiffFromBidRemove(
                        price, self.min_sell_price_diff
                    )

            self.previous_ask_price_diffs = [
                x - self.best_future_ask for x in current_ask_prices
            ]
            self.previous_bid_price_diffs = [
                x - self.best_future_bid for x in current_bid_prices
            ]

            self.min_sell_price = (
                max(0, self.min_sell_price_diff) + self.best_future_ask
            )
            self.max_buy_price = min(0, self.max_buy_price_diff) + self.best_future_bid

            self.logger.warning(f"Maxdiff: {self.max_buy_price_diff}")
            self.logger.warning(f"Mindiff: {self.min_sell_price_diff}")
            self.logger.warning(f"MaxBuy: {self.max_buy_price}")
            self.logger.warning(f"MinSell: {self.min_sell_price}")

            if self.position < 100:
                if self.current_bid_volume < 100:
                    self.bid_id = next(self.order_ids)
                    self.bid_price = self.max_buy_price
                    self.send_insert_order(
                        self.bid_id,
                        Side.BUY,
                        self.bid_price,
                        min(100 - self.position, 100 - self.current_bid_volume),
                        Lifespan.GOOD_FOR_DAY,
                    )
                    self.current_bid_volume += 100 - self.current_bid_volume
                # else:
                # amend

            if self.position > -100:
                if self.current_ask_volume < 100:
                    self.ask_id = next(self.order_ids)
                    self.ask_price = self.min_sell_price
                    self.send_insert_order(
                        self.ask_id,
                        Side.SELL,
                        self.ask_price,
                        min(100 + self.position, 100 - self.current_ask_volume),
                        Lifespan.GOOD_FOR_DAY,
                    )
                    self.current_ask_volume += 100 - self.current_ask_volume

    def on_order_status_message(
        self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int
    ) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        if client_order_id == self.ask_id:
            self.current_ask_volume = remaining_volume
        else:
            self.current_bid_volume = remaining_volume

    def on_position_change_message(
        self, future_position: int, etf_position: int
    ) -> None:
        """Called when your position changes.

        Since every trade in the ETF is automatically hedged in the future,
        future_position and etf_position will always be the inverse of each
        other (i.e. future_position == -1 * etf_position).
        """
        self.position = etf_position

    def on_trade_ticks_message(
        self, instrument: int, trade_ticks: List[Tuple[int, int]]
    ) -> None:
        """Called periodically to report trading activity on the market.

        Each trade tick is a pair containing a price and the number of lots
        traded at that price since the last trade ticks message.
        """
        pass

    def updateMinSellPriceDiffFromBidAdd(self, bidPriceDiff, oldPriceDiff):
        return max(bidPriceDiff, oldPriceDiff)

    def updateMinSellPriceDiffFromAskAdd(self, askPriceDiff, oldPriceDiff):
        return min(askPriceDiff, oldPriceDiff)

    def updateMinSellPriceDiffFromBidRemove(self, bidPriceDiff, oldPriceDiff):
        return min(bidPriceDiff, oldPriceDiff)

    def updateMaxBuyPriceDiffFromBidAdd(self, bidPriceDiff, oldPriceDiff):
        return max(bidPriceDiff, oldPriceDiff)

    def updateMaxBuyPriceDiffFromAskAdd(self, askPriceDiff, oldPriceDiff):
        return min(askPriceDiff, oldPriceDiff)

    def updateMaxBuyPriceDiffFromAskRemove(self, askPriceDiff, oldPriceDiff):
        return max(askPriceDiff, oldPriceDiff)

    def updateMinSellPrice(self, sellPriceDiff, futureAsk):
        return max(0, sellPriceDiff) + futureAsk

    def updateMaxBuyPrice(self, buyPriceDiff, futureBid):
        return min(0, buyPriceDiff) + futureBid

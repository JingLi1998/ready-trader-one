import asyncio
import itertools
import numpy as np

from typing import List, Tuple

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side


class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        """Initialise a new instance of the AutoTrader class."""
        super(AutoTrader, self).__init__(loop)
        self.order_ids = itertools.count(1)
        self.ask_id = (
            self.ask_price
        ) = self.bid_id = self.bid_price = self.bid_volume = self.ask_volume = 0

        self.position = 0

        self.etf_seq_number = -1
        self.future_seq_number = -1

        ob_depth = 5
        self.future_ask_prices = np.zeros(ob_depth)
        self.future_ask_volumes = np.zeros(ob_depth)
        self.future_bid_prices = np.zeros(ob_depth)
        self.future_bid_volumes = np.zeros(ob_depth)

        self.etf_ask_prices = np.zeros(ob_depth)
        self.etf_ask_volumes = np.zeros(ob_depth)
        self.etf_bid_prices = np.zeros(ob_depth)
        self.etf_bid_volumes = np.zeros(ob_depth)

        self.ones_arr = np.ones(ob_depth)

        self.max_bid_volume = 100
        self.max_ask_volume = 100

        self.price_bw = 0
        self.price_bw_prev = 0
        self.volume_im_etf = 0
        self.volume_im_etf_prev = 0

        self.mean_est = 0
        self.var_est = 0
        self.est_index = 1

        self.order_count = 0
        self.max_order_count = 20

        self.tick_size = 1

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
        if (
            sequence_number < self.future_seq_number
            or sequence_number < self.etf_seq_number
        ):
            return

        if instrument == Instrument.FUTURE:
            if sequence_number >= self.future_seq_number:
                self.future_ask_prices = np.array(ask_prices, dtype="int64")
                self.future_ask_volumes = np.array(ask_volumes, dtype="int64")
                self.future_bid_prices = np.array(bid_prices, dtype="int64")
                self.future_bid_volumes = np.array(bid_volumes, dtype="int64")
                self.future_seq_number = sequence_number
        elif sequence_number >= self.etf_seq_number:
            self.etf_ask_prices = np.array(ask_prices, dtype="int64")
            self.etf_ask_volumes = np.array(ask_volumes, dtype="int64")
            self.etf_bid_prices = np.array(bid_prices, dtype="int64")
            self.etf_bid_volumes = np.array(bid_volumes, dtype="int64")
            self.etf_seq_number = sequence_number

        if self.etf_seq_number != self.future_seq_number:
            return

        best_bid = self.future_bid_prices[0]
        best_ask = self.future_ask_prices[0]
        if best_bid == 0 or best_ask == 0:
            return

        bids_on_ob = self.bid_volume * (self.bid_price >= self.etf_bid_prices[-1])
        asks_on_ob = self.ask_volume * (self.ask_price <= self.etf_ask_prices[-1])
        weight_total_etf = np.dot(
            self.etf_bid_volumes + self.etf_ask_volumes - bids_on_ob - asks_on_ob,
            self.ones_arr,
        )
        price_bw_etf = 0
        if weight_total_etf != 0:
            price_bw_etf = (
                np.dot(self.etf_bid_prices, self.etf_bid_volumes)
                + np.dot(self.etf_ask_prices, self.etf_ask_volumes)
                - self.bid_price * bids_on_ob
                - self.ask_price * asks_on_ob
            ) / weight_total_etf
            self.volume_im_etf_prev = self.volume_im_etf
            self.volume_im_etf = (
                np.dot(
                    self.etf_bid_volumes
                    - self.etf_ask_volumes
                    - bids_on_ob
                    + asks_on_ob,
                    self.ones_arr,
                )
                / weight_total_etf
            )

        weight_total_future = np.dot(
            self.future_bid_volumes + self.future_ask_volumes, self.ones_arr
        )
        price_bw_future = 0
        if weight_total_future != 0:
            price_bw_future = (
                np.dot(self.future_bid_prices, self.future_bid_volumes)
                + np.dot(self.future_ask_prices, self.future_ask_volumes)
            ) / weight_total_future

        self.price_bw_prev = self.price_bw
        self.price_bw = price_bw_etf
        if self.price_bw == 0 or self.price_bw_prev == 0:
            self.est_index += 1
            return

        log_return = np.log(self.price_bw / self.price_bw_prev)
        if self.est_index >= 3:
            self.var_est = (
                (self.est_index - 3) * self.var_est
                + (log_return - self.volume_im_etf_prev - self.mean_est) ** 2
            ) / (self.est_index - 2)
        if self.est_index >= 2:
            self.mean_est = (
                (self.est_index - 2) * self.mean_est
                + log_return
                - self.volume_im_etf_prev
            ) / (self.est_index - 1)
        self.est_index += 1
        if self.est_index <= 3:
            return

        log_price_bw_expected_return = (
            self.volume_im_etf + self.mean_est + self.var_est / 2
        )

        self.logger.warning(f"WeightTotal: {weight_total_etf}")
        self.logger.warning(f"PriceBW: {self.price_bw}")
        self.logger.warning(f"VolumeIm: {self.volume_im_etf}")
        self.logger.warning(f"LogReturn: {log_return}")
        self.logger.warning(f"ExpReturn: {log_price_bw_expected_return}")
        self.logger.warning(f"MeanEst: {self.mean_est}")
        self.logger.warning(f"VarEst: {self.var_est}")

        if log_price_bw_expected_return > 0:
            insert_volume = min(self.max_bid_volume - self.position, 1)
            insert_price = np.floor(
                np.exp(log_price_bw_expected_return) * self.price_bw
                + best_bid
                - price_bw_future
            )
            if (
                insert_volume > 0
                and insert_price != self.bid_price
                and self.order_count < self.max_order_count - 2
            ):
                if self.bid_id != 0:
                    self.send_cancel_order(self.bid_id)
                    self.bid_id = 0
                    self.order_count += 1
                self.bid_id = next(self.order_ids)
                self.bid_price = insert_price
                self.bid_volume = insert_volume
                self.send_insert_order(
                    self.bid_id,
                    Side.BUY,
                    self.bid_price,
                    self.bid_volume,
                    Lifespan.GOOD_FOR_DAY,
                )
                self.order_count += 1
        elif log_price_bw_expected_return < 0:
            insert_volume = min(self.max_ask_volume + self.position, 1)
            insert_price = np.ceil(
                np.exp(log_price_bw_expected_return) * self.price_bw
                + best_ask
                - price_bw_future
            )
            if (
                insert_volume > 0
                and insert_price != self.ask_price
                and self.order_count < self.max_order_count - 2
            ):
                if self.ask_id != 0:
                    self.send_cancel_order(self.ask_id)
                    self.ask_id = 0
                    self.order_count += 1
                self.ask_id = next(self.order_ids)
                self.ask_price = insert_price
                self.ask_volume = insert_volume
                self.send_insert_order(
                    self.ask_id,
                    Side.SELL,
                    self.ask_price,
                    self.ask_volume,
                    Lifespan.GOOD_FOR_DAY,
                )
                self.order_count += 1

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

        if client_order_id == self.bid_id:
            self.bid_volume = remaining_volume
            if remaining_volume == 0:
                self.bid_id = 0
        else:
            self.ask_volume = remaining_volume
            if remaining_volume == 0:
                self.ask_id = 0

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
        self.order_count = 0

"""
Round 5 Market Maker
====================
Targets 4 mean-reverting products (AC1 < -0.08, kurtosis > 8):
  ROBOT_DISHES            AC1=-0.22  Kurt=20  spread=0.07%
  ROBOT_IRONING           AC1=-0.12  Kurt=8.8 spread=0.07%
  OXYGEN_SHAKE_EVENING_BREATH AC1=-0.12  Kurt=10.5 spread=0.13%
  OXYGEN_SHAKE_CHOCOLATE  AC1=-0.08  Kurt=10.8 spread=0.13%

Strategy (adapted from FrankfurtHedgehogs champion approach):
  Fair price  = Wall Mid = (min_bid + max_ask) / 2
                Uses the outermost liquidity walls — more stable than simple
                best-bid/best-ask mid, which is skewed by thin top-of-book.

  Phase 1 (Take):  Immediately cross any order ≥ TAKE_EDGE ticks from wall_mid.
                   At wall_mid exactly, cross only to reduce open inventory.
  Phase 2 (Make):  Penny-jump the deepest qualifying bid/ask (volume > MIN_VOL)
                   that is still on the correct side of wall_mid.
                   Post remaining capacity as passive limit orders.

Key Round-5 adaptations vs Round-1 champion code:
  - Position limit = 10 (not 50), so sizing is tight.
  - Wall mid recalculated every tick (price is not fixed at 10,000).
  - No informed-trader detection needed (no KELP-style alpha signals here).
  - TAKE_EDGE = 1 tick is the minimum profitable cross given a spread of ~7-13 units.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

POSITION_LIMIT = 10          # all Round 5 products

# How many ticks from wall_mid an order must be to get crossed immediately.
# 1 = take anything even 1 tick inside fair value.
TAKE_EDGE = 1

# Only penny-jump an existing order if its volume exceeds this.
# Blocks "bait" 1-lot quotes placed by competitors to manipulate spread.
MIN_VOL = 1

# Minimum distance our passive quote must keep from wall_mid.
# Keeps us from posting AT wall_mid and eating our own spread.
MIN_QUOTE_DIST = 1

# Product-specific parameters: (TAKE_EDGE, MIN_VOL, MIN_QUOTE_DIST)
PRODUCT_PARAMS: Dict[str, Tuple[int, int, int]] = {
    "ROBOT_DISHES":               (1, 3, 1),
    "OXYGEN_SHAKE_EVENING_BREATH": (1, 13, 1),
    "OXYGEN_SHAKE_CHOCOLATE":     (1, 12, 1),
    "OXYGEN_SHAKE_GARLIC":          (1, 12, 1)
}

# Products to trade — must have negative autocorrelation and high kurtosis.
PRODUCTS = [
    "ROBOT_DISHES",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC"
]


# ── Per-product market maker ──────────────────────────────────────────────────

class MMTrader:
    """
    Single-product market maker modelled on the FrankfurtHedgehogs StaticTrader.
    Wall-mid replaces simple mid as the fair-price anchor.
    """

    def __init__(self, symbol: str, state: TradingState):
        self.symbol = symbol
        self.pos     = state.position.get(symbol, 0)

        self.take_edge, self.min_vol, self.min_quote_dist = PRODUCT_PARAMS.get(
            symbol,
            (TAKE_EDGE, MIN_VOL, MIN_QUOTE_DIST)
        )

        od = state.order_depths.get(symbol)
        if od and od.buy_orders and od.sell_orders:
            # Sort: bids descending, asks ascending
            self.bids = dict(sorted(od.buy_orders.items(),  reverse=True))
            self.asks = dict(sorted(od.sell_orders.items(), reverse=False))
            self.bids = {p: abs(v) for p, v in self.bids.items()}
            self.asks = {p: abs(v) for p, v in self.asks.items()}
        else:
            self.bids = {}
            self.asks = {}

        # Wall mid: centre of the FULL order-book range (outermost bid + ask).
        # More stable than best-bid/ask mid; resistant to thin top-of-book.
        self.bid_wall  = min(self.bids) if self.bids else None
        self.ask_wall  = max(self.asks) if self.asks else None
        if self.bid_wall is not None and self.ask_wall is not None:
            self.wall_mid = (self.bid_wall + self.ask_wall) / 2.0
        else:
            self.wall_mid = None

        # Remaining buy/sell capacity this tick
        self.buy_cap  = POSITION_LIMIT - self.pos
        self.sell_cap = POSITION_LIMIT + self.pos

        self._orders: List[Order] = []

    # ── Order helpers ─────────────────────────────────────────────────────────

    def _buy(self, price: int, qty: int):
        qty = min(qty, self.buy_cap)
        if qty > 0:
            self._orders.append(Order(self.symbol, price, qty))
            self.buy_cap -= qty

    def _sell(self, price: int, qty: int):
        qty = min(qty, self.sell_cap)
        if qty > 0:
            self._orders.append(Order(self.symbol, price, -qty))
            self.sell_cap -= qty

    # ── Main logic ────────────────────────────────────────────────────────────

    def get_orders(self) -> List[Order]:
        if self.wall_mid is None:
            return []

        wm = self.wall_mid

        # ── PHASE 1: TAKING ───────────────────────────────────────────────────
        # Cross any sell order that is at least TAKE_EDGE below wall_mid.
        for ask_px, ask_vol in self.asks.items():
            if ask_px <= wm - self.take_edge:
                # Clear profit: buy below fair value
                self._buy(ask_px, ask_vol)
            elif ask_px <= wm and self.pos < 0:
                # At fair price but we're short → buy to flatten inventory
                self._buy(ask_px, min(ask_vol, abs(self.pos)))

        # Cross any bid order that is at least TAKE_EDGE above wall_mid.
        for bid_px, bid_vol in self.bids.items():
            if bid_px >= wm + self.take_edge:
                # Clear profit: sell above fair value
                self._sell(bid_px, bid_vol)
            elif bid_px >= wm and self.pos > 0:
                # At fair price but we're long → sell to flatten inventory
                self._sell(bid_px, min(bid_vol, self.pos))

        # ── PHASE 2: MAKING ───────────────────────────────────────────────────
        if self.buy_cap <= 0 and self.sell_cap <= 0:
            return self._orders   # fully consumed by taking, no passive needed

        # Base passive prices: one tick inside the walls
        bid_price = int(self.bid_wall) + 1
        ask_price = int(self.ask_wall) - 1

        # Inventory skew: when near the position limit, widen the quote on the
        # side that would add more risk, and tighten the side that unwinds it.
        # This makes it cheaper for the market to fill us out of a large position.
        # skew = 0 at flat, ±2 ticks at ±8 lots (80% of limit).
        skew = round(self.pos / POSITION_LIMIT * 2)
        bid_price -= skew   # long → lower bid (less eager to buy more)
        ask_price -= skew   # long → lower ask (more eager to sell)

        # Penny-jump the best qualifying bid (below wall_mid, vol > MIN_VOL)
        for bp, bv in self.bids.items():
            if bp >= wm:
                continue                        # above/at fair — skip
            if bv > self.min_vol:
                candidate = bp + 1
                if candidate < wm:              # must stay below fair
                    bid_price = max(bid_price, candidate)
            else:
                bid_price = max(bid_price, bp)  # match the 1-lot (don't jump)
            break                               # only look at best qualifying

        # Penny-jump the best qualifying ask (above wall_mid, vol > MIN_VOL)
        for ap, av in self.asks.items():
            if ap <= wm:
                continue                        # below/at fair — skip
            if av > self.min_vol:
                candidate = ap - 1
                if candidate > wm:              # must stay above fair
                    ask_price = min(ask_price, candidate)
            else:
                ask_price = min(ask_price, ap)  # match the 1-lot
            break

        # Safety: never quote AT or better than wall_mid (no negative spread)
        bid_price = min(bid_price, int(wm) - self.min_quote_dist)
        ask_price = max(ask_price, int(wm) + self.min_quote_dist)

        if self.buy_cap > 0:
            self._buy(bid_price, self.buy_cap)
        if self.sell_cap > 0:
            self._sell(ask_price, self.sell_cap)

        return self._orders


# ── Main Trader class ─────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}

        for product in PRODUCTS:
            if product not in state.order_depths:
                continue
            trader = MMTrader(product, state)
            result = trader.get_orders()
            if result:
                orders[product] = result

        return orders, 0, ""